"""
SAE-OLS Configuration Module.

Centralizes all hyperparameters and system configuration for the
Sparse Autoencoder - Orthogonal Latent Steering Watermark system.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SAEConfig:
    """Configuration for the Sparse Autoencoder component."""
    # HuggingFace repo for pretrained SAE weights (Gemma Scope)
    repo_id: str = "google/gemma-scope-2b-pt-res"
    filename: str = "layer_20/width_16k/average_l0_71/params.npz"
    # SAE architecture dimensions (inferred from weights at load time)
    d_model: Optional[int] = None  # hidden size of the LLM (e.g. 2304 for Gemma-2-2B)
    d_sae: Optional[int] = None    # number of SAE features (e.g. 16384)
    # Which residual stream layer to hook into
    target_layer: int = 20


@dataclass
class WatermarkConfig:
    """Configuration for watermark embedding and detection."""
    # --- Key & PRF ---
    context_window: int = 2        # w: number of preceding tokens for context hash
    hash_algorithm: str = "sha256" # cryptographic hash for PRF

    # --- Orthogonal Projection ---
    top_k: int = 1                # K: number of top logit tokens defining semantic subspace S
    alpha: float = 1.0             # steering intensity coefficient (normalized delta_h)
    # Regularization epsilon for numerical stability in (W W^T)^{-1}
    projection_eps: float = 1e-6

    # --- Detection ---
    z_threshold: float = 4.0       # Z-score threshold for watermark detection
    # Null distribution parameters (estimated from human-written text)
    # These should be calibrated empirically; defaults are placeholders
    mu_0: float = 0.0              # expected mean of dot-product scores under H0
    sigma_0: float = 0.5           # expected std of dot-product scores under H0


@dataclass
class ModelConfig:
    """Configuration for the LLM and tokenizer."""
    model_name_or_path: str = "google/gemma-2-2b"
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", or "float32"
    # Generation parameters
    max_new_tokens: int = 400
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
