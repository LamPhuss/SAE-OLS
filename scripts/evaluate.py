"""
Evaluation Script for SAE-OLS Watermark.
Measures Fidelity (Perplexity) and Detectability (ROC-AUC, Z-score).
"""

import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Thêm thư mục src vào path để import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from generator import WatermarkedGenerator
from detector import WatermarkDetector

def calculate_perplexity(model, tokenizer, text, device):
    """Tính Perplexity (PPL) của một đoạn văn bản bằng chính model sinh ra nó."""
    encodings = tokenizer(text, return_tensors="pt").to(device)
    max_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl


def main():
    # 1. Khởi tạo cấu hình
    config = Config()
    config.model.max_new_tokens = 400  # Sinh đoạn văn dài hơn để đánh giá chính xác
    config.watermark.alpha = 1.0 # Bạn có thể đổi alpha ở đây để test
    secret_key = "eval_secret_key_2026"

    print("Loading generator model...")
    generator = WatermarkedGenerator(config)
    
    # KHÔNG KHỞI TẠO DETECTOR BÌNH THƯỜNG ĐỂ TRÁNH LOAD MODEL LẦN 2
    # detector = WatermarkDetector(config)
    
    print("Initializing detector (sharing weights to save VRAM)...")
    # Tạo một instance rỗng (không gọi __init__ để tránh load lại checkpoint)
    detector = WatermarkDetector.__new__(WatermarkDetector)
    detector.config = config
    detector.device = config.model.device
    
    # Tái sử dụng (Share) Model, Tokenizer và SAE từ Generator
    detector.model = generator.model
    detector.tokenizer = generator.tokenizer
    detector.sae = generator.sae
    
    # 2. Tập dữ liệu Prompt test (Bạn có thể load từ thư viện datasets như 'wikitext')
    prompts = [
        "The future of artificial intelligence is",
        "In a startling turn of events, the stock market",
        "Climate change has become one of the most pressing",
        "The discovery of quantum computing could lead to",
        "During the Renaissance, art and science",
        "The recipe for a perfect chocolate chip cookie requires",
        "When planning a trip to Japan, one must consider",
        "The fundamental laws of thermodynamics state that",
        "To build a successful startup, founders need to",
        "The history of the Roman Empire is characterized by"
    ]

    unwatermarked_texts = []
    watermarked_texts = []
    
    print(f"\n--- STEP 1: GENERATION ({len(prompts)} prompts) ---")
    for prompt in tqdm(prompts, desc="Generating"):
        # Generate Unwatermarked
        uw_text = generator.generate_unwatermarked(prompt)
        unwatermarked_texts.append(uw_text)
        
        # Generate Watermarked
        w_text = generator.generate(prompt, secret_key)
        watermarked_texts.append(w_text)

    print("\n--- STEP 2: FIDELITY EVALUATION (PERPLEXITY) ---")
    uw_ppls = []
    w_ppls = []
    
    for uw_text, w_text in tqdm(zip(unwatermarked_texts, watermarked_texts), total=len(prompts), desc="Calculating PPL"):
        uw_ppls.append(calculate_perplexity(generator.model, generator.tokenizer, uw_text, generator.device))
        w_ppls.append(calculate_perplexity(generator.model, generator.tokenizer, w_text, generator.device))
        
    mean_uw_ppl = np.mean(uw_ppls)
    mean_w_ppl = np.mean(w_ppls)
    
    print("\n--- STEP 3: DETECTABILITY EVALUATION (Z-SCORE & AUC) ---")
    uw_z_scores = []
    w_z_scores = []
    
    for uw_text, w_text in tqdm(zip(unwatermarked_texts, watermarked_texts), total=len(prompts), desc="Detecting"):
        res_uw = detector.detect(uw_text, secret_key)
        uw_z_scores.append(res_uw.z_score)
        
        res_w = detector.detect(w_text, secret_key)
        w_z_scores.append(res_w.z_score)

    mean_uw_z = np.mean(uw_z_scores)
    mean_w_z = np.mean(w_z_scores)

    # Tính ROC-AUC
    # Nhãn: 0 cho Unwatermarked, 1 cho Watermarked
    y_true = [0] * len(uw_z_scores) + [1] * len(w_z_scores)
    y_scores = uw_z_scores + w_z_scores
    auc_score = roc_auc_score(y_true, y_scores)

    print("\n==================================================")
    print("             SAE-OLS EVALUATION RESULTS           ")
    print("==================================================")
    print(f"Total Prompts Tested : {len(prompts)}")
    print(f"Watermark Alpha      : {config.watermark.alpha}")
    print("\n[1] FIDELITY (LOWER Perplexity is better)")
    print(f"  - Unwatermarked PPL: {mean_uw_ppl:.2f}")
    print(f"  - Watermarked PPL  : {mean_w_ppl:.2f}")
    print(f"  -> Delta PPL       : {mean_w_ppl - mean_uw_ppl:+.2f}")
    
    print("\n[2] DETECTABILITY (HIGHER is better)")
    print(f"  - Mean Z-Score (Unwatermarked) : {mean_uw_z:.2f} (Expected ~ 0.0)")
    print(f"  - Mean Z-Score (Watermarked)   : {mean_w_z:.2f} (Expected > 4.0)")
    print(f"  - ROC-AUC Score                : {auc_score:.4f} (Max = 1.0000)")
    print("==================================================\n")


if __name__ == "__main__":
    main()