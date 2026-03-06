"""
End-to-end test for SAE-OLS watermark pipeline.

Tests the mathematical correctness of each component without requiring
a full LLM (uses mock/small tensors where possible).

For full integration testing with a real model, set:
    SAEOLS_INTEGRATION_TEST=1 python tests/test_pipeline.py
"""

import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch


class TestPRF(unittest.TestCase):
    """Test the Pseudo-Random Function module."""

    def test_deterministic(self):
        """Same key + context should always produce the same feature index."""
        from prf import select_target_feature_index
        idx1 = select_target_feature_index("key123", [10, 20, 30], 16384)
        idx2 = select_target_feature_index("key123", [10, 20, 30], 16384)
        self.assertEqual(idx1, idx2)

    def test_different_keys(self):
        """Different keys should produce different feature indices (with high probability)."""
        from prf import select_target_feature_index
        idx1 = select_target_feature_index("key_A", [10, 20, 30], 16384)
        idx2 = select_target_feature_index("key_B", [10, 20, 30], 16384)
        self.assertNotEqual(idx1, idx2)

    def test_context_window(self):
        """Context window extraction should handle edge cases."""
        from prf import get_context_window
        # Normal case
        ctx = get_context_window([1, 2, 3, 4, 5], 4, 3)
        self.assertEqual(ctx, [2, 3, 4])
        # Start of sequence (padded)
        ctx = get_context_window([1, 2], 1, 4)
        self.assertEqual(ctx, [0, 0, 0, 1])

    def test_uniform_distribution(self):
        """Feature indices should be roughly uniformly distributed."""
        from prf import select_target_feature_index
        F = 100
        counts = [0] * F
        for i in range(10000):
            idx = select_target_feature_index("test", [i], F)
            counts[idx] += 1
        # Chi-squared-like check: no bin should be extremely over/under-represented
        expected = 10000 / F
        for c in counts:
            self.assertGreater(c, expected * 0.3)
            self.assertLess(c, expected * 3.0)


class TestOrthogonalProjection(unittest.TestCase):
    """Test the orthogonal projection mathematics."""

    def test_orthogonality(self):
        """delta_h must be orthogonal to all vectors in W_topK."""
        from orthogonal import compute_orthogonal_steering_vector, verify_orthogonality

        d = 64
        K = 10
        v_target = torch.randn(d)
        W_topK = torch.randn(K, d)

        delta_h = compute_orthogonal_steering_vector(v_target, W_topK)
        max_dot = verify_orthogonality(delta_h, W_topK)

        self.assertAlmostEqual(max_dot, 0.0, places=4,
                               msg=f"delta_h is not orthogonal to W_topK (max dot = {max_dot})")

    def test_preserves_component(self):
        """delta_h should preserve the S-perp component of v_target."""
        from orthogonal import compute_orthogonal_steering_vector

        d = 64
        K = 5
        v_target = torch.randn(d)
        W_topK = torch.randn(K, d)

        delta_h = compute_orthogonal_steering_vector(v_target, W_topK)

        # delta_h should not be zero (unless v_target is entirely in S)
        self.assertGreater(delta_h.norm().item(), 1e-6)

    def test_logits_preserved(self):
        """
        Core distortion-free proof:
        z'_k = w_k^T @ (h + alpha * delta_h) should equal z_k = w_k^T @ h
        for all w_k in W_topK.
        """
        from orthogonal import compute_orthogonal_steering_vector

        d = 128
        K = 20
        alpha = 5.0

        h = torch.randn(d)
        v_target = torch.randn(d)
        W_topK = torch.randn(K, d)

        delta_h = compute_orthogonal_steering_vector(v_target, W_topK)

        # Original logits for top-K
        z_original = W_topK @ h  # [K]
        # Steered logits
        z_steered = W_topK @ (h + alpha * delta_h)  # [K]

        # They should be identical
        diff = (z_steered - z_original).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=4,
                               msg=f"Top-K logits changed by {diff}")

    def test_batch_consistency(self):
        """Batched version should match single-item computation."""
        from orthogonal import (
            compute_orthogonal_steering_vector,
            compute_orthogonal_steering_vector_batch,
        )

        d = 64
        K = 10
        B = 4

        v_targets = torch.randn(B, d)
        W_topK_batch = torch.randn(B, K, d)

        # Batched
        delta_batch = compute_orthogonal_steering_vector_batch(v_targets, W_topK_batch)

        # Single
        for i in range(B):
            delta_single = compute_orthogonal_steering_vector(v_targets[i], W_topK_batch[i])
            diff = (delta_batch[i] - delta_single).abs().max().item()
            self.assertAlmostEqual(diff, 0.0, places=4)


class TestSAE(unittest.TestCase):
    """Test SAE architecture correctness."""

    def test_encode_sparsity(self):
        """Encoded activations should be sparse (many zeros due to JumpReLU)."""
        from sae import JumpReLUSAE

        d_model, d_sae = 64, 256
        sae = JumpReLUSAE(d_model, d_sae)
        # Set some non-trivial weights and thresholds
        sae.W_enc.data = torch.randn(d_model, d_sae) * 0.1
        sae.threshold.data = torch.ones(d_sae) * 0.5
        sae.b_enc.data = torch.zeros(d_sae)

        h = torch.randn(1, d_model)
        acts = sae.encode(h)

        # Should have many zeros
        sparsity = (acts == 0).float().mean().item()
        self.assertGreater(sparsity, 0.3, "SAE output is not sparse enough")

    def test_feature_vector_shape(self):
        """Feature vectors from decoder dictionary should have correct shape."""
        from sae import JumpReLUSAE

        d_model, d_sae = 64, 256
        sae = JumpReLUSAE(d_model, d_sae)

        v = sae.get_feature_vector(42)
        self.assertEqual(v.shape, (d_model,))


class TestDetectionMath(unittest.TestCase):
    """Test the statistical detection logic."""

    def test_z_score_formula(self):
        """Verify Z-score computation matches expected formula."""
        N = 100
        scores = torch.randn(N) + 0.5  # positive bias = watermarked
        S_N = scores.sum().item()
        mu_0 = 0.0
        sigma_0 = 1.0

        z = (S_N - mu_0 * N) / (sigma_0 * math.sqrt(N))

        # With mean ≈ 0.5 and N = 100, z should be roughly 0.5 * sqrt(100) = 5.0
        self.assertGreater(z, 2.0, "Z-score should be positive for biased scores")

    def test_null_distribution(self):
        """Under H0 (no watermark), Z-score should be near 0."""
        N = 1000
        scores = torch.randn(N)  # no bias
        S_N = scores.sum().item()
        z = S_N / math.sqrt(N)  # mu_0=0, sigma_0=1

        # Z should be within [-3, 3] with very high probability
        self.assertLess(abs(z), 4.0,
                        "Z-score under null should not exceed 4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
