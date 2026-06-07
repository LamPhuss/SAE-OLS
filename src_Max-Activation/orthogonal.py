"""
Orthogonal Projection Module for SAE-OLS.

Implements the core mathematical operation that makes the watermark "distortion-free":
projecting the watermark steering vector onto the orthogonal complement of the
semantic subspace, ensuring top-K token logits remain unchanged.

Mathematical foundation:
  Given:
    - W_topK in R^{K x d}: unembedding vectors of the top-K tokens (semantic subspace S)
    - v_target in R^d: SAE feature vector chosen as watermark signal

  We compute:
    P_S = W_topK^T @ (W_topK @ W_topK^T)^{-1} @ W_topK   (projection onto S)
    P_S_perp = I - P_S                                      (projection onto S^perp)
    delta_h = P_S_perp @ v_target                            (distortion-free steering vector)

  Guarantee: for any w_k in W_topK, w_k^T @ delta_h = 0
    => logits of top-K tokens are EXACTLY preserved.
"""

import torch


def compute_orthogonal_steering_vector(
    v_target: torch.Tensor,
    W_topK: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Project the watermark target vector onto S^perp (orthogonal complement
    of the semantic subspace defined by top-K token embeddings).

    Args:
        v_target: SAE feature vector to steer towards, shape [d]
        W_topK: Unembedding vectors of top-K tokens, shape [K, d]
        eps: Regularization for numerical stability

    Returns:
        delta_h: The orthogonal steering vector, shape [d]
                 Guaranteed: W_topK @ delta_h ≈ 0
    """
    # W_topK: [K, d]
    # Gram matrix: G = W_topK @ W_topK^T, shape [K, K]
    G = W_topK @ W_topK.T

    # Add regularization for numerical stability (prevent singular matrix)
    G = G + eps * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)

    # G_inv = (W_topK @ W_topK^T)^{-1}, shape [K, K]
    G_inv = torch.linalg.inv(G)

    # Projection onto S: P_S @ v_target
    # P_S = W_topK^T @ G_inv @ W_topK
    # Instead of building the full [d, d] projection matrix, compute directly:
    # P_S @ v_target = W_topK^T @ (G_inv @ (W_topK @ v_target))
    proj_coeffs = W_topK @ v_target           # [K]
    proj_coeffs = G_inv @ proj_coeffs         # [K]
    proj_onto_S = W_topK.T @ proj_coeffs      # [d]

    # Projection onto S^perp: delta_h = v_target - P_S @ v_target
    delta_h = v_target - proj_onto_S

    return delta_h


def compute_orthogonal_steering_vector_batch(
    v_targets: torch.Tensor,
    W_topK_batch: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Batched version: compute orthogonal steering vectors for multiple tokens.

    Args:
        v_targets: Target feature vectors, shape [B, d]
        W_topK_batch: Top-K unembedding vectors per token, shape [B, K, d]
        eps: Regularization epsilon

    Returns:
        delta_h_batch: Orthogonal steering vectors, shape [B, d]
    """
    B, K, d = W_topK_batch.shape

    # Gram matrices: [B, K, K]
    G = torch.bmm(W_topK_batch, W_topK_batch.transpose(1, 2))
    G = G + eps * torch.eye(K, device=G.device, dtype=G.dtype).unsqueeze(0)

    # G_inv: [B, K, K]
    G_inv = torch.linalg.inv(G)

    # Project v_targets onto S for each batch element
    # proj_coeffs = W_topK @ v_target: [B, K]
    proj_coeffs = torch.bmm(W_topK_batch, v_targets.unsqueeze(-1)).squeeze(-1)
    # G_inv @ proj_coeffs: [B, K]
    proj_coeffs = torch.bmm(G_inv, proj_coeffs.unsqueeze(-1)).squeeze(-1)
    # W_topK^T @ proj_coeffs: [B, d]
    proj_onto_S = torch.bmm(W_topK_batch.transpose(1, 2), proj_coeffs.unsqueeze(-1)).squeeze(-1)

    # delta_h = v_target - projection onto S
    delta_h = v_targets - proj_onto_S

    return delta_h


def verify_orthogonality(delta_h: torch.Tensor, W_topK: torch.Tensor) -> float:
    """
    Verify that delta_h is orthogonal to the semantic subspace.
    Returns the maximum absolute dot product (should be ~0).

    Useful for debugging and testing.
    """
    dots = W_topK @ delta_h  # [K]
    return dots.abs().max().item()
