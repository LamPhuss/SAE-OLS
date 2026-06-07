"""
SAE Module for SAE-OLS.

Loads a pretrained JumpReLU Sparse Autoencoder (from Google Gemma Scope)
and exposes the feature dictionary D = {f_1, ..., f_F} used for watermark
steering vectors.

Key concept from the theory:
  - SAE decomposes the entangled hidden state h into sparse, monosemantic features.
  - The decoder weight matrix W_dec (shape: [d_sae, d_model]) serves as the
    "Feature Dictionary". Each ROW of W_dec is a feature vector f_i in R^d.
  - During watermark embedding, we select f_i as the steering target v_target.
"""

import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

from config import SAEConfig


class JumpReLUSAE(nn.Module):
    """
    JumpReLU Sparse Autoencoder.

    Architecture:
        Encoder: f = JumpReLU(W_enc @ h + b_enc)
            where JumpReLU(x) = ReLU(x) * (x > threshold)
        Decoder: h_hat = W_dec @ f + b_dec

    The sparsity comes from the threshold gating in JumpReLU — only features
    with pre-activation above the learned threshold are activated.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        # Initialized to zeros; weights loaded from pretrained checkpoint
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        """
        Encode hidden states into sparse feature activations.

        Args:
            input_acts: Hidden states from LLM, shape [..., d_model]

        Returns:
            Sparse feature activations, shape [..., d_sae]
        """
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Reconstruct hidden states from sparse activations."""
        return acts @ self.W_dec + self.b_dec

    def forward(self, input_acts: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(input_acts))

    def get_feature_vector(self, feature_idx: int) -> torch.Tensor:
        """
        Get a single feature vector from the decoder dictionary.

        This is the steering vector v_target = D[feature_idx].
        Each row of W_dec corresponds to one monosemantic feature direction
        in the LLM's residual stream space.

        Args:
            feature_idx: Index into the feature dictionary [0, d_sae)

        Returns:
            Feature vector of shape [d_model]
        """
        return self.W_dec[feature_idx]  # row of W_dec

    def get_feature_vectors_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get multiple feature vectors at once.

        Args:
            indices: Tensor of feature indices, shape [N]

        Returns:
            Feature vectors, shape [N, d_model]
        """
        return self.W_dec[indices]


def load_sae(config: SAEConfig, device: str = "cpu") -> JumpReLUSAE:
    """
    Load pretrained SAE weights from Gemma Scope.

    Args:
        config: SAE configuration with repo/filename info
        device: Target device

    Returns:
        Loaded JumpReLUSAE model with pretrained weights
    """
    path_to_params = hf_hub_download(
        repo_id=config.repo_id,
        filename=config.filename,
        force_download=False,
    )

    params = np.load(path_to_params)
    d_model, d_sae = params['W_enc'].shape

    # Update config with actual dimensions
    config.d_model = d_model
    config.d_sae = d_sae

    sae = JumpReLUSAE(d_model, d_sae)

    # Load numpy weights into PyTorch parameters
    pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    sae.load_state_dict(pt_params)
    sae = sae.to(device)
    sae.eval()

    return sae
