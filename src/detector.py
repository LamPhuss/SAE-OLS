"""
Watermark Detector for SAE-OLS.

Detection algorithm:
  1. Reconstruct target feature sequence using (key, context) at each position
  2. Pass text through LLM to extract hidden states at the SAE target layer
  3. Compute dot-product score s_t = h_t^T @ v_target_t at each position
  4. Aggregate scores and perform Z-score hypothesis testing

Key insight:
  - If text is human-written: h_t has no correlation with v_target_t => s_t ≈ 0
  - If text is watermarked: h_t was steered towards v_target_t => s_t > 0 consistently
  - CLT guarantees the aggregated score follows a normal distribution under H0
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from sae import JumpReLUSAE, load_sae
from prf import get_context_window, select_target_feature_index
from hooks import gather_residual_activations


@dataclass
class DetectionResult:
    """Result of watermark detection on a piece of text."""
    is_watermarked: bool
    z_score: float
    p_value: float
    total_score: float
    num_tokens: int
    mean_score: float
    per_token_scores: Optional[List[float]] = None


class WatermarkDetector:
    """
    Detects SAE-OLS watermarks in text using dot-product scoring
    and Z-score hypothesis testing.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.model.device

        # Load LLM (same model used for generation)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name_or_path,
            device_map=self.device,
            dtype=getattr(torch, config.model.torch_dtype),
        )
        self.model.eval()

        # Load SAE
        self.sae = load_sae(config.sae, device=self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def detect(
        self,
        text: str,
        secret_key: str,
        return_per_token: bool = False
    ) -> DetectionResult:
        """
        Detect whether text contains a SAE-OLS watermark.

        Algorithm:
          1. Tokenize text
          2. For each token t, reconstruct v_target_t from (key, context)
          3. Forward pass to get hidden states h_t at target layer
          4. Compute s_t = h_t^T @ v_target_t (dot product)
          5. Aggregate and Z-test

        Args:
            text: Text to check for watermark
            secret_key: The watermark key to test against
            return_per_token: If True, include per-token scores in result

        Returns:
            DetectionResult with z_score and watermark decision
        """
        wm_cfg = self.config.watermark
        sae_cfg = self.config.sae

        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=True
        ).to(self.device)
        input_ids = inputs["input_ids"]  # [1, N]
        token_ids = input_ids[0].tolist()
        N = len(token_ids)

        if N < 2:
            return DetectionResult(
                is_watermarked=False, z_score=0.0, p_value=1.0,
                total_score=0.0, num_tokens=N, mean_score=0.0
            )

        # --- Step 1: Get hidden states at target layer ---
        hidden_states = gather_residual_activations(
            self.model, sae_cfg.target_layer, input_ids,
            attention_mask=inputs.get("attention_mask")
        )  # [1, N, d_model]
        hidden_states = hidden_states[0]  # [N, d_model]

        # --- Step 2: Reconstruct target features and compute scores ---
        scores = []
        for t in range(N):
            # Get context window for position t
            ctx = get_context_window(token_ids, t, wm_cfg.context_window)

            # Reconstruct target feature index
            target_idx = select_target_feature_index(
                secret_key, ctx, self.sae.d_sae, wm_cfg.hash_algorithm
            )

            # Get the target feature vector
            v_target = self.sae.get_feature_vector(target_idx)  # [d_model]

            # Compute dot-product score: s_t = h_t^T @ v_target
            h_t = hidden_states[t].float()
            v_target = v_target.float()
            s_t = torch.dot(h_t, v_target).item()
            scores.append(s_t)

        # --- Step 3: Z-score hypothesis testing ---
        scores_tensor = torch.tensor(scores)
        S_N = scores_tensor.sum().item()
        mean_score = scores_tensor.mean().item()

        # Z = (S_N - mu_0) / (sigma_0 * sqrt(N))
        z_score = (S_N - wm_cfg.mu_0 * N) / (wm_cfg.sigma_0 * math.sqrt(N))

        # One-sided p-value: P(Z > z) under standard normal
        p_value = 0.5 * math.erfc(z_score / math.sqrt(2))

        is_watermarked = z_score > wm_cfg.z_threshold

        return DetectionResult(
            is_watermarked=is_watermarked,
            z_score=z_score,
            p_value=p_value,
            total_score=S_N,
            num_tokens=N,
            mean_score=mean_score,
            per_token_scores=scores if return_per_token else None,
        )

    @torch.no_grad()
    def calibrate_null_distribution(
        self,
        human_texts: List[str],
        secret_key: str,
    ) -> tuple:
        """
        Estimate mu_0 and sigma_0 from a corpus of human-written text.
        These are the null distribution parameters for Z-score testing.

        This should be run once before deployment to calibrate the detector.

        Args:
            human_texts: List of human-written texts
            secret_key: The watermark key

        Returns:
            (mu_0, sigma_0): Mean and std of per-token scores under H0
        """
        all_scores = []
        for text in human_texts:
            result = self.detect(text, secret_key, return_per_token=True)
            if result.per_token_scores:
                all_scores.extend(result.per_token_scores)

        scores_t = torch.tensor(all_scores)
        mu_0 = scores_t.mean().item()
        sigma_0 = scores_t.std().item()

        return mu_0, sigma_0
