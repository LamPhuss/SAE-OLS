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
from orthogonal import compute_orthogonal_steering_vector


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

        # --- Step 1: Pass text through model to get logits ---
        if hasattr(self.model, 'lm_head'):
            W_U = self.model.lm_head.weight.detach()
        elif hasattr(self.model, 'embed_out'):
            W_U = self.model.embed_out.weight.detach()
        else:
            raise ValueError("Không tìm thấy ma trận unembedding")

        # Chạy forward pass 1 lần duy nhất để lấy toàn bộ logits tự nhiên
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]  # [N, V]

        # --- Step 2: Reconstruct delta_h and compute scores ---
        from orthogonal import compute_orthogonal_steering_vector
        
        scores = []
        valid_tokens_count = 0  # Biến đếm số token thực sự được đo
        
        for t in range(N - 1):
            # 1. Tìm lại original_logits tại bước t y hệt như Generator
            original_logits_t = logits[t]
            
            # ==========================================================
            # GÁC CỔNG ENTROPY Ở DETECTOR (Phải khớp với Generator)
            # ==========================================================
            probs = torch.softmax(original_logits_t, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            
            # Thay '1.0' bằng đúng ngưỡng Entropy bạn đã cài ở Generator
            if entropy.item() < 2.0: 
                continue  # Bỏ qua hoàn toàn token này, không tính vào Z-Score
                
            valid_tokens_count += 1
            
            # --- Các bước tiếp theo giữ nguyên ---
            ctx = get_context_window(token_ids, t + 1, wm_cfg.context_window)
            target_idx = select_target_feature_index(
                secret_key, ctx, self.sae.d_sae, wm_cfg.hash_algorithm
            )
            v_target = self.sae.get_feature_vector(target_idx).float()
            
            top1_idx = torch.topk(original_logits_t, 1).indices
            W_top1 = W_U[top1_idx].float()

            delta_h = compute_orthogonal_steering_vector(
                v_target, W_top1, eps=wm_cfg.projection_eps
            )
            delta_h_norm = delta_h.norm()
            if delta_h_norm > 1e-8:
                delta_h = delta_h / delta_h_norm

            next_token = token_ids[t + 1]
            token_vec = W_U[next_token].float()

            tok_norm = token_vec.norm()
            d_norm = delta_h.norm()
            if tok_norm > 1e-8 and d_norm > 1e-8:
                s_t = (torch.dot(token_vec, delta_h) / (tok_norm * d_norm)).item()
            else:
                s_t = 0.0
            
            scores.append(s_t)
            
        # --- Step 3: Z-score hypothesis testing ---
        # SỬA LỖI Ở ĐÂY: Dùng số lượng token hợp lệ (valid_tokens_count) thay vì N
        if valid_tokens_count < 2:
            return DetectionResult(
                is_watermarked=False, 
                z_score=0.0, 
                p_value=1.0,
                total_score=0.0, 
                num_tokens=N, 
                mean_score=0.0
            )

        scores_tensor = torch.tensor(scores)
        mean_score = scores_tensor.mean().item()
        sample_std = scores_tensor.std().item()
        
        if sample_std > 1e-8:
            # Chia cho căn bậc hai của số token THỰC SỰ mang watermark
            z_score = mean_score / (sample_std / math.sqrt(valid_tokens_count))
        else:
            z_score = 0.0
        S_N = scores_tensor.sum().item()

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
