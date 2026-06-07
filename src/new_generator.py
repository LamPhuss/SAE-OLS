"""
Watermarked Text Generator for SAE-OLS.
Implements the white-box Orthogonal Latent Steering generation pipeline.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from sae import JumpReLUSAE, load_sae
from prf import get_prompt_concept_pool, generate_concept_sequence
from orthogonal import compute_orthogonal_steering_vector
from hooks import HiddenStateInterceptor

class WatermarkedGenerator:
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
        self._W_U = self._get_unembedding_matrix()
        self.sae = load_sae(config.sae, device=self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_unembedding_matrix(self) -> torch.Tensor:
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head.weight.detach()
        elif hasattr(self.model, 'embed_out'):
            return self.model.embed_out.weight.detach()
        else:
            raise ValueError("Cannot find unembedding matrix in model architecture")

    def _get_top_k_unembeddings(self, logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        top_indices = torch.topk(logits, k).indices
        W_topK = self._W_U[top_indices]
        return W_topK, top_indices

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        secret_key: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        max_tokens = max_new_tokens or self.config.model.max_new_tokens
        wm_cfg = self.config.watermark
        sae_cfg = self.config.sae

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        generated_ids = input_ids[0].tolist()
        prompt_len = input_ids.shape[1]
        
        # ==========================================================
        # SOTA: PROMPT-ANCHORED CONCEPT POOL (Thay thế I-CDC)
        # ==========================================================
        with torch.no_grad():
            prompt_outputs = self.model(input_ids, output_hidden_states=True)
            prompt_hidden = prompt_outputs.hidden_states[sae_cfg.target_layer][0]
            
        pool_size = getattr(wm_cfg, 'pool_size', 200)
        pool = get_prompt_concept_pool(self.sae, prompt_hidden, top_k=pool_size)
        concept_seq = generate_concept_sequence(secret_key, prompt, pool, length=max_tokens + 50)
                
        # Token-by-token generation with steering
        for step in range(max_tokens):
            cur_ids = torch.tensor([generated_ids], device=self.device)

            interceptor = HiddenStateInterceptor(self.model, sae_cfg.target_layer)
            interceptor.register()
            outputs = self.model(cur_ids)
            h_t = interceptor.captured  
            interceptor.remove()

            original_logits = outputs.logits[0, -1, :]  
            
            # ==========================================================
            # LẤY TARGET CONCEPT THEO TRÌNH TỰ (Không băm context)
            # ==========================================================
            target_idx = concept_seq[step]
            v_target = self.sae.get_feature_vector(target_idx).float()
            v_target = v_target - v_target.mean()

            # --- Giữ nguyên logic OLS tinh xảo của bạn ---
            W_topK, _ = self._get_top_k_unembeddings(original_logits, k=1)

            probs = torch.softmax(original_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            
            if entropy.item() < 2.0: 
                steered_logits = original_logits.float()
            else:
                delta_h = compute_orthogonal_steering_vector(
                    v_target.float(), W_topK.float(), eps=wm_cfg.projection_eps
                )
                
                delta_h_norm = delta_h.norm()
                if delta_h_norm > 1e-8:
                    delta_h = delta_h / delta_h_norm

                logit_delta_raw = (self._W_U @ delta_h.to(self._W_U.dtype)).float()

                V_c = 10 
                top_Vc_vals, top_Vc_indices = torch.topk(original_logits, V_c)
                delta_top_c = logit_delta_raw[top_Vc_indices]
                max_delta = delta_top_c.abs().max()
                
                if max_delta > 1e-8:
                    logit_delta = (logit_delta_raw / max_delta) * 2.0
                else:
                    logit_delta = logit_delta_raw
                    
                steered_logits = original_logits.float() + (wm_cfg.alpha * logit_delta)

                mask = torch.ones_like(original_logits, dtype=torch.bool, device=self.device)
                mask[top_Vc_indices] = False
                steered_logits[mask] = original_logits[mask].float()

                max_allowed_logit = top_Vc_vals[0].float() + 0.5
                steered_logits = torch.clamp(steered_logits, max=max_allowed_logit)   
                
            if self.config.model.do_sample:
                steered_logits = steered_logits / self.config.model.temperature
                probs = F.softmax(steered_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > self.config.model.top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                idx_in_sorted = torch.multinomial(sorted_probs, 1)
                next_token = sorted_indices[idx_in_sorted].item()
            else:
                next_token = steered_logits.argmax().item()

            generated_ids.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_unwatermarked(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        max_tokens = max_new_tokens or self.config.model.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
            do_sample=self.config.model.do_sample
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)