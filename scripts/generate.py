"""
CLI script for generating watermarked text using SAE-OLS.

Usage:
    python scripts/generate.py \
        --prompt "Once upon a time" \
        --key "my_secret_key" \
        --model google/gemma-2-2b \
        --device cuda:0 \
        --max_tokens 200 \
        --alpha 5.0 \
        --top_k 50 \
        --output output.jsonl
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config, ModelConfig, SAEConfig, WatermarkConfig
from generator import WatermarkedGenerator
from utils import append_jsonl_line


def parse_args():
    parser = argparse.ArgumentParser(description="SAE-OLS Watermarked Text Generation")

    # Input
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--key", type=str, required=True, help="Watermark secret key")
    parser.add_argument("--dataset", type=str, default=None, help="JSONL dataset with prompts (overrides --prompt)")

    # Model
    parser.add_argument("--model", type=str, default="google/gemma-2-2b", help="HuggingFace model name or path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])

    # SAE
    parser.add_argument("--sae_repo", type=str, default="google/gemma-scope-2b-pt-res")
    parser.add_argument("--sae_file", type=str, default="layer_20/width_16k/average_l0_71/params.npz")
    parser.add_argument("--target_layer", type=int, default=20)

    # Watermark
    parser.add_argument("--alpha", type=float, default=5.0, help="Steering intensity")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K tokens for semantic subspace")
    parser.add_argument("--context_window", type=int, default=4, help="Context window size for PRF")

    # Generation
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_sample", action="store_true", help="Use greedy decoding")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path")
    parser.add_argument("--also_unwatermarked", action="store_true", help="Also generate unwatermarked text")

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        model=ModelConfig(
            model_name_or_path=args.model,
            device=args.device,
            torch_dtype=args.dtype,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample,
        ),
        sae=SAEConfig(
            repo_id=args.sae_repo,
            filename=args.sae_file,
            target_layer=args.target_layer,
        ),
        watermark=WatermarkConfig(
            top_k=args.top_k,
            alpha=args.alpha,
            context_window=args.context_window,
        ),
    )

    print("Loading model and SAE...")
    gen = WatermarkedGenerator(config)
    print("Model loaded.")

    # Handle dataset or single prompt
    if args.dataset:
        from utils import read_jsonl
        data = read_jsonl(args.dataset)
        prompts = [(item.get("id", str(i)), item["prompt"]) for i, item in enumerate(data)]
    else:
        prompts = [("0", args.prompt)]

    for idx, (item_id, prompt) in enumerate(prompts):
        print(f"\n--- Generating [{idx+1}/{len(prompts)}] id={item_id} ---")
        print(f"Prompt: {prompt[:100]}...")

        # Watermarked generation
        watermarked_text = gen.generate(prompt, args.key)
        print(f"Watermarked: {watermarked_text[:200]}...")

        result = {
            "id": item_id,
            "prompt": prompt,
            "watermarked": watermarked_text,
        }

        # Optionally generate baseline
        if args.also_unwatermarked:
            unwatermarked_text = gen.generate_unwatermarked(prompt)
            result["unwatermarked"] = unwatermarked_text
            print(f"Unwatermarked: {unwatermarked_text[:200]}...")

        if args.output:
            append_jsonl_line(args.output, result)

    print("\nDone.")


if __name__ == "__main__":
    main()
