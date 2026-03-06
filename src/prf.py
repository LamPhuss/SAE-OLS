"""
PRF (Pseudo-Random Function) Module for SAE-OLS.

Implements the cryptographic context-dependent hashing mechanism that maps
(secret_key, context_tokens) -> target_feature_index.

From the theory:
  - At each token position t, extract context window c_t = (x_{t-w}, ..., x_{t-1})
  - Compute r_t = H_K(c_t) using a cryptographic hash
  - Select target feature: v_target = D[r_t mod F]

This ensures that without key K, the sequence of target features looks
indistinguishable from random noise — preventing watermark stealing attacks.
"""

import hashlib
from typing import List, Sequence

import torch


def compute_context_hash(
    secret_key: str,
    context_token_ids: Sequence[int],
    algorithm: str = "sha256"
) -> int:
    """
    Compute a deterministic pseudo-random integer from a secret key and
    a sequence of context token IDs.

    H_K(c_t) = Hash(K || token_{t-w} || ... || token_{t-1})

    Args:
        secret_key: The secret watermark key K
        context_token_ids: Token IDs forming the context window c_t
        algorithm: Hash algorithm to use

    Returns:
        A large pseudo-random integer derived from the hash
    """
    hasher = hashlib.new(algorithm)
    # Feed key
    hasher.update(secret_key.encode("utf-8"))
    # Feed each context token ID as bytes
    for tid in context_token_ids:
        hasher.update(tid.to_bytes(4, byteorder="big"))

    digest = hasher.digest()
    return int.from_bytes(digest, byteorder="big")


def select_target_feature_index(
    secret_key: str,
    context_token_ids: Sequence[int],
    num_features: int,
    algorithm: str = "sha256"
) -> int:
    """
    Select which SAE feature to use as watermark target for the current token.

    v_target = D[r_t mod F]

    Args:
        secret_key: The secret watermark key K
        context_token_ids: Preceding token IDs (context window)
        num_features: Total number of SAE features F (e.g. 16384)
        algorithm: Hash algorithm

    Returns:
        Feature index in [0, num_features)
    """
    r = compute_context_hash(secret_key, context_token_ids, algorithm)
    return r % num_features


def get_context_window(
    token_ids: List[int],
    position: int,
    window_size: int
) -> List[int]:
    """
    Extract the context window of w tokens preceding position t.

    c_t = (x_{t-w}, ..., x_{t-1})

    If position < window_size, pads with zeros on the left.

    Args:
        token_ids: Full sequence of token IDs generated so far
        position: Current token position t
        window_size: Context window size w

    Returns:
        List of w token IDs
    """
    start = max(0, position - window_size)
    context = token_ids[start:position]
    # Left-pad with 0 if not enough context
    if len(context) < window_size:
        context = [0] * (window_size - len(context)) + context
    return context


def select_target_features_for_sequence(
    secret_key: str,
    token_ids: List[int],
    num_features: int,
    window_size: int = 4,
    algorithm: str = "sha256"
) -> List[int]:
    """
    Compute target feature indices for every token position in a sequence.
    Used during detection to reconstruct the expected target features.

    Args:
        secret_key: The secret watermark key K
        token_ids: Full token ID sequence
        num_features: Total SAE features F
        window_size: Context window size w
        algorithm: Hash algorithm

    Returns:
        List of feature indices, one per token position
    """
    indices = []
    for t in range(len(token_ids)):
        ctx = get_context_window(token_ids, t, window_size)
        idx = select_target_feature_index(secret_key, ctx, num_features, algorithm)
        indices.append(idx)
    return indices
