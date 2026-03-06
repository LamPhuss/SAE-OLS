"""
CLI script for detecting SAE-OLS watermarks in text.

Usage:
    python scripts/detect.py \
        --input text_to_check.jsonl \
        --key "my_secret_key" \
        --model google/gemma-2-2b \
        --device cuda:0 \
        --output results.jsonl

    # Single text mode:
    python scripts/detect.py \
        --text "Some text to check for watermark..." \
        --key "my_secret_key" \
        --model google/gemma-2-2b
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config, ModelConfig, SAEConfig, WatermarkConfig
from detector import WatermarkDetector
from utils import read_jsonl, append_jsonl_line


def parse_args():
    parser = argparse.ArgumentParser(description="SAE-OLS Watermark Detection")

    # Input
    parser.add_argument("--text", type=str, default=None, help="Single text to check")
    parser.add_argument("--input", type=str, default=None, help="JSONL file with texts to check")
    parser.add_argument("--field", type=str, default="watermarked", help="Field name containing text in JSONL")
    parser.add_argument("--key", type=str, required=True, help="Watermark secret key to test")

    # Model
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32")

    # SAE
    parser.add_argument("--sae_repo", type=str, default="google/gemma-scope-2b-pt-res")
    parser.add_argument("--sae_file", type=str, default="layer_20/width_16k/average_l0_71/params.npz")
    parser.add_argument("--target_layer", type=int, default=20)

    # Detection
    parser.add_argument("--z_threshold", type=float, default=4.0, help="Z-score threshold")
    parser.add_argument("--context_window", type=int, default=4)
    parser.add_argument("--mu_0", type=float, default=0.0, help="Null mean (calibrate first)")
    parser.add_argument("--sigma_0", type=float, default=1.0, help="Null std (calibrate first)")

    # Output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.text is None and args.input is None:
        print("Error: Provide either --text or --input")
        sys.exit(1)

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
            mu_0=args.mu_0,
            sigma_0=args.sigma_0,
        ),
    )

    print("Loading model and SAE...")
    detector = WatermarkDetector(config)
    print("Model loaded.")

    # Collect texts to check
    if args.text:
        texts = [{"id": "0", "text": args.text}]
    else:
        data = read_jsonl(args.input, args.start, args.end)
        texts = [{"id": item.get("id", str(i)), "text": item[args.field]} for i, item in enumerate(data)]

    for i, item in enumerate(texts):
        result = detector.detect(item["text"], args.key)

        print(f"\n--- [{i+1}/{len(texts)}] id={item['id']} ---")
        print(f"  Watermarked: {result.is_watermarked}")
        print(f"  Z-score:     {result.z_score:.4f}")
        print(f"  p-value:     {result.p_value:.6e}")
        print(f"  Tokens:      {result.num_tokens}")
        print(f"  Mean score:  {result.mean_score:.6f}")

        if args.output:
            out = {
                "id": item["id"],
                "is_watermarked": result.is_watermarked,
                "z_score": round(result.z_score, 4),
                "p_value": result.p_value,
                "num_tokens": result.num_tokens,
            }
            append_jsonl_line(args.output, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
