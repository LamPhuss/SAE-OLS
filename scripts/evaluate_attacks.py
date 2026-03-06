"""
Robustness Evaluation Script for SAE-OLS Watermark.
Tests the watermark's survival against Deletion, Swapping, and LLM Paraphrasing.
"""

import sys
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from generator import WatermarkedGenerator
from detector import WatermarkDetector

# ==========================================
# CÁC HÀM TẤN CÔNG (ATTACK FUNCTIONS)
# ==========================================

def attack_word_deletion(text: str, drop_ratio: float = 0.1) -> str:
    """Xóa ngẫu nhiên một tỷ lệ từ vựng trong văn bản."""
    words = text.split()
    if not words: return text
    # Giữ lại từ nếu số random sinh ra lớn hơn drop_ratio
    survived_words = [w for w in words if random.random() > drop_ratio]
    return " ".join(survived_words)

def attack_word_swap(text: str, swap_ratio: float = 0.1) -> str:
    """Đảo vị trí các cặp từ kề nhau để phá vỡ Context Hash."""
    words = text.split()
    n = len(words)
    if n < 2: return text
    
    num_swaps = int(n * swap_ratio)
    for _ in range(num_swaps):
        idx = random.randint(0, n - 2)
        # Swap
        words[idx], words[idx+1] = words[idx+1], words[idx]
    return " ".join(words)

def attack_llm_paraphrase(text: str, model, tokenizer, device) -> str:
    """Dùng chính LLM để viết lại đoạn văn (Paraphrasing Attack)."""
    prompt = f"Rewrite the following paragraph using different words but keeping the exact same meaning:\n\n{text}\n\nRewritten paragraph:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Cắt bỏ phần prompt, chỉ lấy phần được sinh ra
    input_length = inputs["input_ids"].shape[1]
    rewritten_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return rewritten_text.strip()

# ==========================================
# VÒNG LẶP ĐÁNH GIÁ (EVALUATION LOOP)
# ==========================================

def main():
    config = Config()
    config.model.max_new_tokens = 200  # Độ dài đủ để chống chịu tấn công
    secret_key = "eval_secret_key_2026"

    print("Loading models for Attack Evaluation...")
    generator = WatermarkedGenerator(config)
    
    # Share weights cho detector để tránh OOM
    detector = WatermarkDetector.__new__(WatermarkDetector)
    detector.config = config
    detector.device = config.model.device
    detector.model = generator.model
    detector.tokenizer = generator.tokenizer
    detector.sae = generator.sae

    prompts = [
        "The future of artificial intelligence is",
        "In a startling turn of events, the stock market",
        "Climate change has become one of the most pressing",
        "The discovery of quantum computing could lead to",
        "During the Renaissance, art and science"
    ]

    print("\n--- STEP 1: GENERATE BASELINE TEXTS ---")
    uw_texts = []
    w_texts = []
    
    for prompt in tqdm(prompts, desc="Generating texts"):
        uw_texts.append(generator.generate_unwatermarked(prompt))
        w_texts.append(generator.generate(prompt, secret_key))

    # Từ điển chứa kết quả Z-Score của từng loại tấn công
    results = {
        "No Attack": [],
        "Deletion (10%)": [],
        "Deletion (30%)": [],
        "Swap (10%)": [],
        "Paraphrase (LLM)": []
    }

    print("\n--- STEP 2: APPLY ATTACKS & DETECT ---")
    for w_text in tqdm(w_texts, desc="Attacking & Detecting"):
        # 1. Không tấn công (Baseline)
        results["No Attack"].append(detector.detect(w_text, secret_key).z_score)
        
        # 2. Tấn công xóa từ
        text_del_10 = attack_word_deletion(w_text, 0.1)
        results["Deletion (10%)"].append(detector.detect(text_del_10, secret_key).z_score)
        
        text_del_30 = attack_word_deletion(w_text, 0.3)
        results["Deletion (30%)"].append(detector.detect(text_del_30, secret_key).z_score)
        
        # 3. Tấn công đảo từ
        text_swap_10 = attack_word_swap(w_text, 0.1)
        results["Swap (10%)"].append(detector.detect(text_swap_10, secret_key).z_score)
        
        # 4. Tấn công Diễn đạt lại (Paraphrase)
        text_para = attack_llm_paraphrase(w_text, generator.model, generator.tokenizer, generator.device)
        results["Paraphrase (LLM)"].append(detector.detect(text_para, secret_key).z_score)

    print("\n==================================================")
    print("        ATTACK ROBUSTNESS RESULTS (MEAN Z-SCORE)  ")
    print("==================================================")
    print(f"Original Watermark Strength (No Attack) : {np.mean(results['No Attack']):.2f}")
    print("-" * 50)
    print(f"Word Deletion (10% dropped)             : {np.mean(results['Deletion (10%)']):.2f}")
    print(f"Word Deletion (30% dropped)             : {np.mean(results['Deletion (30%)']):.2f}")
    print(f"Word Swap (10% swapped)                 : {np.mean(results['Swap (10%)']):.2f}")
    print(f"LLM Paraphrasing (Complete rewrite)     : {np.mean(results['Paraphrase (LLM)']):.2f}")
    print("==================================================")
    print("Note: If Z-Score remains > 2.0 to 4.0, the watermark survived the attack!")

if __name__ == "__main__":
    main()