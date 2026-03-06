"""
Watermarked Text Generator for SAE-OLS.

Implements the white-box Orthogonal Latent Steering generation pipeline.

At each token generation step t:
  1. PRF selects a target SAE feature v_target based on (key, context)
  2. LLM computes hidden state h_t and logits z_t
  3. Top-K tokens define the semantic subspace S
  4. v_target is projected onto S^perp -> delta_h
  5. Hidden state is steered: h'_t = h_t + alpha * delta_h
  6. New logits z'_t preserve top-K rankings exactly (distortion-free)
  7. Token is sampled from softmax(z'_t)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from sae import JumpReLUSAE, load_sae
from prf import get_context_window, select_target_feature_index
from orthogonal import compute_orthogonal_steering_vector
from hooks import HiddenStateInterceptor


class WatermarkedGenerator:
    """
    Generates watermarked text using Orthogonal Latent Steering.

    The watermark is embedded by subtly steering the LLM's hidden states
    towards SAE feature directions that are orthogonal to the semantic
    subspace, ensuring zero distortion on top-K token probabilities.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.model.device

        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name_or_path,
            device_map=self.device,
            dtype=getattr(torch, config.model.torch_dtype),
        )
        self.model.eval()

        # Get the unembedding matrix W_U: [V, d]
        # For most models this is the lm_head weight or the tied embedding weight
        self._W_U = self._get_unembedding_matrix()

        # Load SAE
        self.sae = load_sae(config.sae, device=self.device)

        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_unembedding_matrix(self) -> torch.Tensor:
        """
        Extract the unembedding (output projection) matrix W_U from the model.
        W_U maps hidden states to logits: z = W_U @ h

        Returns:
            W_U of shape [vocab_size, d_model]
        """
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head.weight.detach()  # [V, d]
        elif hasattr(self.model, 'embed_out'):
            return self.model.embed_out.weight.detach()
        else:
            raise ValueError("Cannot find unembedding matrix in model architecture")

    def _get_top_k_unembeddings(self, logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the unembedding vectors for the top-K tokens by logit value.

        Args:
            logits: Logit scores for all tokens, shape [V]
            k: Number of top tokens

        Returns:
            W_topK: Unembedding vectors, shape [K, d]
            top_indices: Token indices, shape [K]
        """
        top_indices = torch.topk(logits, k).indices  # [K]
        W_topK = self._W_U[top_indices]  # [K, d]
        return W_topK, top_indices

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        secret_key: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate watermarked text from a prompt.

        This implements the full SAE-OLS pipeline:
        For each new token, we intercept the hidden state, compute the
        orthogonal steering vector, and modify the hidden state before
        the model computes final logits.

        Args:
            prompt: Input text prompt
            secret_key: Watermark secret key K
            max_new_tokens: Max tokens to generate (overrides config)

        Returns:
            Generated watermarked text (prompt + continuation)
        """
        max_tokens = max_new_tokens or self.config.model.max_new_tokens
        wm_cfg = self.config.watermark
        sae_cfg = self.config.sae

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]  # [1, prompt_len]
        generated_ids = input_ids[0].tolist()  # flat list for tracking

        # Token-by-token generation with steering
        for step in range(max_tokens):
            # Current full sequence as tensor
            cur_ids = torch.tensor([generated_ids], device=self.device)

            # --- Step 1: Forward pass to get hidden state and logits ---
            # We need both the hidden state at target_layer AND the final logits.
            # Use a hook to capture hidden state at the SAE's target layer.
            interceptor = HiddenStateInterceptor(self.model, sae_cfg.target_layer)

            # We'll do a two-phase approach:
            # Phase A: Forward pass to get original logits and hidden state
            interceptor.register()
            outputs = self.model(cur_ids)
            h_t = interceptor.captured  # [1, seq_len, d_model]
            interceptor.remove()

            # Get logits for the LAST token position
            original_logits = outputs.logits[0, -1, :]  # [V]
            h_last = h_t[0, -1, :]  # [d_model] — hidden state of last token

            # --- Step 2: PRF selects target feature ---
            ctx = get_context_window(
                generated_ids, len(generated_ids), wm_cfg.context_window
            )
            target_idx = select_target_feature_index(
                secret_key, ctx, self.sae.d_sae, wm_cfg.hash_algorithm
            )
            v_target = self.sae.get_feature_vector(target_idx)  # [d_model]

            # --- Step 3: Build semantic subspace from top-K tokens ---
            W_topK, _ = self._get_top_k_unembeddings(original_logits, wm_cfg.top_k)

            # --- Step 4: Orthogonal projection ---
            delta_h = compute_orthogonal_steering_vector(
                v_target, W_topK, eps=wm_cfg.projection_eps
            )

            # --- Step 5: Steer hidden state ---
            # h'_t = h_t + alpha * delta_h
            h_steered = h_last + wm_cfg.alpha * delta_h  # [d_model]

            # --- Step 6: Recompute logits with steered hidden state ---
            # z'_t = W_U @ h'_t
            steered_logits = self._W_U @ h_steered  # [V]

            # --- Step 7: Sample from steered distribution ---
            if self.config.model.do_sample:
                # Apply temperature
                steered_logits = steered_logits / self.config.model.temperature
                # Apply top-p (nucleus) sampling
                probs = F.softmax(steered_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens with cumulative probability above top_p
                mask = cumsum - sorted_probs > self.config.model.top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                # Sample
                idx_in_sorted = torch.multinomial(sorted_probs, 1)
                next_token = sorted_indices[idx_in_sorted].item()
            else:
                next_token = steered_logits.argmax().item()

            generated_ids.append(next_token)

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_unwatermarked(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text WITHOUT watermark (baseline for comparison).
        Uses standard model.generate().
        """
        max_tokens = max_new_tokens or self.config.model.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
            do_sample=self.config.model.do_sample,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
