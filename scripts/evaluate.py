"""
CLI script for evaluating SAE-OLS watermark performance.

Runs detection on both watermarked and unwatermarked texts,
computes metrics (Precision, Recall, F1, Accuracy) and generates ROC curves.

Usage:
    python scripts/evaluate.py \
        --watermarked wm_texts.jsonl \
        --unwatermarked uwm_texts.jsonl \
        --key "my_secret_key" \
        --model google/gemma-2-2b \
        --device cuda:0 \
        --roc_output roc_curve.png
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config, ModelConfig, SAEConfig, WatermarkConfig
from detector import WatermarkDetector
from utils import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="SAE-OLS Watermark Evaluation")

    parser.add_argument("--watermarked", type=str, required=True, help="JSONL with watermarked texts")
    parser.add_argument("--unwatermarked", type=str, required=True, help="JSONL with unwatermarked texts")
    parser.add_argument("--wm_field", type=str, default="watermarked")
    parser.add_argument("--uwm_field", type=str, default="unwatermarked")
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--sae_repo", type=str, default="google/gemma-scope-2b-pt-res")
    parser.add_argument("--sae_file", type=str, default="layer_20/width_16k/average_l0_71/params.npz")
    parser.add_argument("--target_layer", type=int, default=20)
    parser.add_argument("--z_threshold", type=float, default=4.0)
    parser.add_argument("--context_window", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--roc_output", type=str, default=None, help="Path to save ROC curve PNG")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate null distribution from unwatermarked texts first")

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        model=ModelConfig(
            model_name_or_path=args.model,
            device=args.device,
            torch_dtype=args.dtype,
        ),
        sae=SAEConfig(
            repo_id=args.sae_repo,
            filename=args.sae_file,
            target_layer=args.target_layer,
        ),
        watermark=WatermarkConfig(
            context_window=args.context_window,
            z_threshold=args.z_threshold,
        ),
    )

    print("Loading model and SAE...")
    detector = WatermarkDetector(config)
    print("Model loaded.")

    # Load data
    wm_data = read_jsonl(args.watermarked, args.start, args.end)
    uwm_data = read_jsonl(args.unwatermarked, args.start, args.end)

    # Optional calibration
    if args.calibrate:
        print("Calibrating null distribution from unwatermarked texts...")
        uwm_texts = [item[args.uwm_field] for item in uwm_data]
        mu_0, sigma_0 = detector.calibrate_null_distribution(uwm_texts, args.key)
        print(f"  mu_0 = {mu_0:.6f}, sigma_0 = {sigma_0:.6f}")
        detector.config.watermark.mu_0 = mu_0
        detector.config.watermark.sigma_0 = sigma_0

    # Detect on watermarked texts
    print("\nDetecting on watermarked texts...")
    wm_scores = []
    wm_labels = []
    tp, fp, tn, fn = 0, 0, 0, 0

    for i, item in enumerate(wm_data):
        text = item[args.wm_field]
        result = detector.detect(text, args.key)
        wm_scores.append(result.z_score)
        wm_labels.append(1)
        if result.is_watermarked:
            tp += 1
        else:
            fn += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(wm_data)}] processed")

    # Detect on unwatermarked texts
    print("Detecting on unwatermarked texts...")
    for i, item in enumerate(uwm_data):
        text = item[args.uwm_field]
        result = detector.detect(text, args.key)
        wm_scores.append(result.z_score)
        wm_labels.append(0)
        if result.is_watermarked:
            fp += 1
        else:
            tn += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(uwm_data)}] processed")

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n{'='*50}")
    print(f"Results (z_threshold={args.z_threshold}):")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"{'='*50}")

    # ROC curve
    if args.roc_output:
        try:
            import numpy as np
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt

            fpr, tpr, _ = roc_curve(wm_labels, wm_scores)
            roc_auc = auc(fpr, tpr)

            # Find TPR at FPR ≈ 0.01
            idx_001 = np.searchsorted(fpr, 0.01)
            tpr_at_001 = tpr[min(idx_001, len(tpr) - 1)]

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'SAE-OLS (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('SAE-OLS Watermark Detection ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.savefig(args.roc_output, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\nROC curve saved to {args.roc_output}")
            print(f"AUC: {roc_auc:.4f}")
            print(f"TPR@FPR=0.01: {tpr_at_001:.4f}")
        except ImportError:
            print("Warning: sklearn/matplotlib not installed, skipping ROC curve")


if __name__ == "__main__":
    main()
