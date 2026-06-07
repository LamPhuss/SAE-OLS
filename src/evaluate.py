"""
Advanced Evaluation Script for SAE-OLS Watermark.
Measures Fidelity (PPL, Rep-3, Win-Rate), Detectability (ROC-AUC, Z-score, Token Efficiency),
and Full Robustness (Attacks: Deletion, Swap, Paraphrase).
"""

import sys
import os
import csv
import json
import torch
import numpy as np
import random
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from generator import WatermarkedGenerator
from detector import WatermarkDetector

# ==========================================
# 1. CÁC HÀM ĐÁNH GIÁ FIDELITY & ATTACKS
# ==========================================

def calculate_perplexity(model, tokenizer, prompt_text, generated_text, device):
    full_text = prompt_text + generated_text
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_tokens.size(1)

    encodings = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    if seq_len <= prompt_len: 
        return 0.0

    target_ids = input_ids.clone()
    target_ids[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

    return torch.exp(loss).item()

def calculate_rep3(text: str) -> float:
    words = text.split()
    if len(words) < 3: return 0.0
    ngrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def get_lcs_length(list1, list2):
    if not list1 or not list2: return 0
    m, n = len(list1), len(list2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i-1] == list2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
    
def calculate_f1(uw_scores, w_scores, threshold):
    """Tính F1 Score dựa trên Z-Score và Threshold"""
    tp = sum(1 for z in w_scores if z >= threshold) # True Positives
    fn = len(w_scores) - tp                         # False Negatives
    fp = sum(1 for z in uw_scores if z >= threshold)# False Positives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# VÁ LỖI 2: Đồng bộ tham số detector
def measure_token_efficiency(detector, prompt_text, generated_text, secret_key, tokenizer, target_threshold=4.0, thresholds=[50, 100, 150, 200, 300]):
    tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    for length in thresholds:
        if len(tokens) < length:
            continue
        trunc_text = tokenizer.decode(tokens[:length], skip_special_tokens=True)
        res = detector.detect(prompt_text, trunc_text, secret_key)
        if res.z_score >= target_threshold:
            return length
    return None 

def attack_word_deletion(text: str, drop_ratio: float) -> str:
    words = text.split()
    if not words: return text
    survived_words = [w for w in words if random.random() > drop_ratio]
    return " ".join(survived_words)

def attack_word_swap(text: str, swap_ratio: float) -> str:
    words = text.split()
    n = len(words)
    if n < 2: return text
    num_swaps = int(n * swap_ratio)
    for _ in range(num_swaps):
        idx = random.randint(0, n - 2)
        words[idx], words[idx+1] = words[idx+1], words[idx]
    return " ".join(words)

def attack_llm_paraphrase(text: str, model, tokenizer, device) -> str:
    messages = [
        {
            "role": "system", 
            "content": "You are a precise paraphrasing tool. Rewrite the user's text to change its vocabulary and sentence structure while preserving the exact original meaning. Output ONLY the paraphrased text."
        },
        {"role": "user", "content": text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    safe_max_tokens = min(500, int(len(text.split()) * 3.0))
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=safe_max_tokens, 
                temperature=0.7, 
                top_p=0.9, 
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] 
            )
        
        input_length = inputs["input_ids"].shape[1]
        rewritten = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        # VÁ LỖI 3: Dọn rác đàm thoại của Llama-3
        if "\n\n" in rewritten:
            parts = rewritten.split("\n\n", 1)
            if len(parts[0]) < 100 and any(kw in parts[0].lower() for kw in ["here", "sure", "rewrite", "version", ":"]):
                rewritten = parts[1].strip()
                
    except Exception as e:
        print(f"Paraphrase Error: {e}")
        rewritten = text
    finally:
        del inputs
        torch.cuda.empty_cache()
        
    return rewritten

# ==========================================
# 2. MAIN PIPELINE
# ==========================================

def main():
    config = Config()
    secret_key = "eval_secret_key_2026"

    print("Loading SAE-OLS generator model...")
    generator = WatermarkedGenerator(config)
    
    print("Initializing detector (sharing weights to save VRAM)...")
    detector = WatermarkDetector.__new__(WatermarkDetector)
    detector.config = config
    detector.device = config.model.device
    detector.model = generator.model
    detector.tokenizer = generator.tokenizer
    detector.sae = generator.sae
    detector.d_model = generator.d_model
    detector.anchor = generator.anchor

    prompts = []
    dataset_path = "data/alpaca_eval_10k.jsonl"
    
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): prompts.append(json.loads(line).get("prompt", ""))
    else:
        prompts = ["The future of artificial intelligence is"]

    TEST_LIMIT = int(os.environ.get('EVAL_TEST_LIMIT', 10000))
    prompts = prompts[:TEST_LIMIT]

    unwatermarked_texts = []
    watermarked_texts = []
    
    system_instruction = "You are a highly detailed academic assistant. You MUST provide exhaustive, multi-paragraph answers (minimum 250 words) to every prompt. Do not be brief."

    eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'eval_data.csv')
    avg_gen_time = 0.0 

    if os.path.exists(eval_data_path):
        print(f"Found cached generations at {eval_data_path}, loading...")
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                unwatermarked_texts.append(row["uwm_text"])
                watermarked_texts.append(row["wm_text"])
            prompts = [row["prompt"] for row in csv.DictReader(open(eval_data_path, 'r', encoding='utf-8'))]
        avg_gen_time = float('nan')
    else:
        print(f"\n--- STEP 1: GENERATION ({len(prompts)} prompts) ---")
        gen_times = [] 
        
        for prompt in tqdm(prompts, desc="Generating"):
            messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
            formatted_prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            unwatermarked_texts.append(generator.generate_unwatermarked(formatted_prompt))
            
            start_time = time.time()
            watermarked_texts.append(generator.generate(formatted_prompt, secret_key))
            end_time = time.time()
            
            gen_times.append(end_time - start_time)
            
            # Chống OOM
            torch.cuda.empty_cache()

        avg_gen_time = np.mean(gen_times) 

        with open(eval_data_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "uwm_text", "wm_text"])
            writer.writeheader()
            for p, uw, w in zip(prompts, unwatermarked_texts, watermarked_texts):
                writer.writerow({"prompt": p, "uwm_text": uw, "wm_text": w})

    print("\n--- STEP 2: FIDELITY (PPL & Rep-3) ---")
    uw_ppls, w_ppls, uw_reps, w_reps = [], [], [], []
    
    for prompt, uw_text, w_text in tqdm(zip(prompts, unwatermarked_texts, watermarked_texts), total=len(prompts), desc="Fidelity Evals"):
        messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
        formatted_prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        uw_ppls.append(calculate_perplexity(generator.model, generator.tokenizer, formatted_prompt, uw_text, generator.device))
        w_ppls.append(calculate_perplexity(generator.model, generator.tokenizer, formatted_prompt, w_text, generator.device))
        
        uw_reps.append(calculate_rep3(uw_text))
        w_reps.append(calculate_rep3(w_text))
        
    mean_uw_ppl, mean_w_ppl = np.mean(uw_ppls), np.mean(w_ppls)
    mean_uw_rep, mean_w_rep = np.mean(uw_reps), np.mean(w_reps)
    
    print("\n--- STEP 3: DETECTABILITY ---")
    uw_z_scores, w_z_scores, efficiency_tokens = [], [], []
    det_times = []
    
    for prompt, uw_text, w_text in tqdm(zip(prompts, unwatermarked_texts, watermarked_texts), total=len(prompts), desc="Detectability Evals"):
        messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
        formatted_prompt = detector.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # VÁ LỖI 2: Dùng cấu trúc detect mới
        uw_z_scores.append(detector.detect(formatted_prompt, uw_text, secret_key).z_score)
        
        start_time = time.time()
        w_z_scores.append(detector.detect(formatted_prompt, w_text, secret_key).z_score)
        end_time = time.time()
        
        det_times.append(end_time - start_time)
        
        eff = measure_token_efficiency(detector, formatted_prompt, w_text, secret_key, generator.tokenizer, target_threshold=4.0)
        if eff: efficiency_tokens.append(eff)

    mean_uw_z = np.mean(uw_z_scores)
    mean_w_z = np.mean(w_z_scores)
    auc_score = roc_auc_score([0]*len(uw_z_scores) + [1]*len(w_z_scores), uw_z_scores + w_z_scores)
    
    # BỔ SUNG TÍNH F1 SCORE
    f1_score_3 = calculate_f1(uw_z_scores, w_z_scores, 3.0)
    f1_score_4 = calculate_f1(uw_z_scores, w_z_scores, 4.0)
    
    mean_efficiency = np.mean(efficiency_tokens) if efficiency_tokens else float('inf')
    mean_det_time = np.mean(det_times)

    print("\n--- STEP 4: ROBUSTNESS (ATTACKS) ---")
    attacks = {
        "No Attack": lambda t: t,
        "Deletion (10%)": lambda t: attack_word_deletion(t, 0.1),
        "Deletion (30%)": lambda t: attack_word_deletion(t, 0.3),
        "Swap (10%)": lambda t: attack_word_swap(t, 0.1),
        "Paraphrase (LLM)": lambda t: attack_llm_paraphrase(t, generator.model, generator.tokenizer, generator.device)
    }

    attack_results = {name: [] for name in attacks}

    print("Running Attacks...")
    for prompt, w_text in tqdm(zip(prompts, watermarked_texts), desc="Robustness Evals", total=len(prompts)):
        messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
        formatted_prompt = detector.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 1. Quét bản gốc trước để lấy Tọa độ Cụm và Z-Score làm hệ quy chiếu
        orig_res = detector.detect(formatted_prompt, w_text, secret_key)
        det_orig_clusters = orig_res.used_clusters
        orig_z = orig_res.z_score

        for name, attack_fn in attacks.items():
            attacked_suffix = attack_fn(w_text) 
            res = detector.detect(formatted_prompt, attacked_suffix, secret_key)
            attack_results[name].append(res.z_score)
            
            # 2. CHỈ IN LOG KHI ĐANG THỰC HIỆN PARAPHRASE ATTACK
            if name == "Paraphrase (LLM)":
                det_para_clusters = res.used_clusters
                matched_para_clusters = get_lcs_length(det_orig_clusters, det_para_clusters)
                retention_rate = (matched_para_clusters / len(det_orig_clusters)) * 100 if det_orig_clusters else 0.0
                
                # Dùng tqdm.write để thanh load không bị trôi hay vỡ khung
                tqdm.write(f"\n[DEBUG PARAPHRASE - BÁO CÁO NHANH]")
                tqdm.write(f"- Số cụm của văn bản gốc watermark              : {len(det_orig_clusters)} cụm")
                tqdm.write(f"- Số cụm Detector tìm thấy (Sau Paraphrase)     : {len(det_para_clusters)} cụm")
                tqdm.write(f"- Số cụm TRÙNG KHỚP (Gốc vs Paraphrase)         : {matched_para_clusters} cụm")
                tqdm.write(f"- Tỷ lệ bám sát của Máy dò (Paraphrase)         : {retention_rate:.2f} %")
                tqdm.write(f"- Z-score văn bản gốc watermark                 : {orig_z:.2f}")
                tqdm.write(f"- Z-score sau khi detect (Paraphrase)           : {res.z_score:.2f}")
                tqdm.write("-" * 75)

    print("\n==================================================")
    print("      SAE-OLS ADVANCED EVALUATION RESULTS         ")
    print("==================================================")
    print(f"\n[1] FIDELITY METRICS (Utility)")
    print(f"  - Delta PPL        : {mean_w_ppl - mean_uw_ppl:+.2f} (Base: {mean_uw_ppl:.2f})")
    print(f"  - Rep-3 (%)        : WM = {mean_w_rep*100:.1f}% | UW = {mean_uw_rep*100:.1f}%")
    
    print(f"\n[2] DETECTABILITY METRICS (Security)")
    print(f"  - Mean Z-Score     : {mean_w_z:.2f} (UW: {mean_uw_z:.2f})")
    print(f"  - ROC-AUC Score    : {auc_score:.4f}")
    print(f"  - F1 Score (Z=3)   : {f1_score_3:.4f}") # <--- THÊM DÒNG NÀY
    print(f"  - F1 Score (Z=4)   : {f1_score_4:.4f}") # <--- THÊM DÒNG NÀY
    print(f"  - Token Efficiency : ~{mean_efficiency:.1f} tokens to reach Z>4")

    print("\n[3] ROBUSTNESS METRICS (Under Attack)")
    baseline_z = np.mean(attack_results["No Attack"])
    for name, scores in attack_results.items():
        mean_z = np.mean(scores)
        survived_pct = sum(1 for z in scores if z >= 4.0) / len(scores) * 100
        z_drop = (1 - mean_z / baseline_z) * 100 if baseline_z > 0 else 0.0
        print(f"  {name:<25s} | Z= {mean_z:5.2f} | Drop: {z_drop:5.1f}% | Survived(Z>4): {survived_pct:5.1f}%")
        
    print(f"\n[4] LATENCY METRICS (Speed)")
    gen_time_str = f"{avg_gen_time:.2f} s / prompt" if not np.isnan(avg_gen_time) else "N/A (Loaded from cache)"
    print(f"  - Avg Generation Time: {gen_time_str}")
    print(f"  - Avg Detection Time : {mean_det_time:.2f} s / prompt")
    print("==================================================\n")

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'eval.csv')
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "WM", "Delta_PPL", "Base_PPL", "Rep3_Score_%", "Mean_Z_Score", "UW_Z_Score", 
        "ROC_AUC", "F1@3", "F1@4", "Avg_Tokens_to_Z4", "No_Attack_Z", "Del_10_Z", "Del_30_Z", "Swap_10_Z", "Para_Z"
    ]
    
    row = {
        "WM": "SAE-OLS",
        "Delta_PPL": f"{mean_w_ppl - mean_uw_ppl:+.2f}",
        "Base_PPL": f"{mean_uw_ppl:.2f}",
        "Rep3_Score_%": f"{mean_w_rep*100:.1f}",
        "Mean_Z_Score": f"{mean_w_z:.2f}",
        "UW_Z_Score": f"{mean_uw_z:.2f}",
        "ROC_AUC": f"{auc_score:.4f}",
        "F1@3": f"{f1_score_3:.4f}",   # <--- THÊM DÒNG NÀY
        "F1@4": f"{f1_score_4:.4f}",   # <--- THÊM DÒNG NÀY
        "Avg_Tokens_to_Z4": f"{mean_efficiency:.1f}" if mean_efficiency != float('inf') else "N/A",
        "No_Attack_Z": f"{baseline_z:.2f}",
        "Del_10_Z": f"{np.mean(attack_results['Deletion (10%)']):.2f}",
        "Del_30_Z": f"{np.mean(attack_results['Deletion (30%)']):.2f}",
        "Swap_10_Z": f"{np.mean(attack_results['Swap (10%)']):.2f}",
        "Para_Z": f"{np.mean(attack_results['Paraphrase (LLM)']):.2f}"
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    main()