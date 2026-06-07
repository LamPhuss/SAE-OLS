"""
Text Quality Evaluation Script for SAE-OLS Watermark.
Reads pre-generated 'eval_data.csv' and calculates:
1. Lexical Diversity: Distinct-1, Distinct-2, Log-TTR
2. PPL & Rep-3
3. Semantic Similarity: SBERT Cosine, BERTScore
4. LLM-as-a-Judge: Win-Rate
"""

import sys
import os
import csv
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from config import Config

# ==========================================
# 1. CÁC HÀM ĐÁNH GIÁ
# ==========================================

def calculate_distinct_n(text: str, n: int) -> float:
    tokens = text.split()
    if len(tokens) < n: return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return len(set(ngrams)) / len(ngrams)

def calculate_log_ttr(text: str) -> float:
    tokens = text.split()
    if not tokens: return 0.0
    types = set(tokens)
    if len(tokens) == 1: return 0.0
    return math.log(len(types)) / math.log(len(tokens))

def calculate_rep3(text: str) -> float:
    words = text.split()
    if len(words) < 3: return 0.0
    ngrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))

def calculate_perplexity(model, tokenizer, prompt_text, generated_text, device):
    full_text = generated_text if generated_text.startswith(prompt_text) else prompt_text + " " + generated_text
    prompt_len = tokenizer(prompt_text, return_tensors="pt").input_ids.size(1)
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    if input_ids.size(1) <= prompt_len: return 0.0
    target_ids = input_ids.clone()
    target_ids[:, :prompt_len] = -100
    with torch.no_grad():
        loss = model(input_ids, labels=target_ids).loss
    return torch.exp(loss).item()

# --- LLM-as-a-Judge ---
V98_API_KEY = os.environ.get("GPT-API-KEY", "")
JUDGE_MODEL = "gpt-5.2" 
judge_client = OpenAI(base_url="https://v98store.com/v1", api_key=V98_API_KEY) if V98_API_KEY else None

def evaluate_win_rate(prompt, uw_text, w_text):
    if not judge_client: return "N/A"
    
    is_wm_A = random.random() < 0.5
    text_a, text_b = (w_text, uw_text) if is_wm_A else (uw_text, w_text)
    
    sys_prompt = (
        "You are an expert evaluator. Read the prompt and two responses (A and B). "
        "Which response is more fluent, coherent, and logically answers the prompt? "
        "Reply with EXACTLY one word: A, B, or Tie."
    )
    user_prompt = f"Prompt: {prompt}\n\n--- Response A ---\n{text_a}\n\n--- Response B ---\n{text_b}"
    
    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL, 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], 
            max_tokens=5, temperature=0.0
        )
        answer = resp.choices[0].message.content.strip().upper()
        
        if "TIE" in answer: return "Tie"
        if ("A" in answer and is_wm_A) or ("B" in answer and not is_wm_A): return "Win"
        return "Loss"
    except Exception:
        return "Error"

# ==========================================
# 2. HỆ THỐNG ĐÁNH GIÁ CHÍNH
# ==========================================

def main():
    config = Config()
    device = config.model.device

    # 1. Đọc dữ liệu từ file CSV đã sinh
    eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'eval_data_ub_llama3_1000.csv')
    if not os.path.exists(eval_data_path):
        print(f"[LỖI] Không tìm thấy file dữ liệu tại {eval_data_path}")
        print("Vui lòng chạy script sinh dữ liệu trước!")
        return

    prompts, uw_texts, w_texts = [], [], []
    print(f"Loading data from {eval_data_path}...")
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["prompt"])
            uw_texts.append(row["uwm_text"])
            w_texts.append(row["wm_text"])
    
    # --- BỔ SUNG: GIỚI HẠN SỐ LƯỢNG MẪU ĐÁNH GIÁ ---
    TEST_LIMIT = 1000 # Thay đổi con số này tùy theo nhu cầu (ví dụ: 500, 1000)
    
    if len(prompts) > TEST_LIMIT:
        print(f"Dataset contains {len(prompts)} samples. Truncating to the first {TEST_LIMIT} samples for evaluation...")
        prompts = prompts[:TEST_LIMIT]
        uw_texts = uw_texts[:TEST_LIMIT]
        w_texts = w_texts[:TEST_LIMIT]
    else:
        print(f"Loaded {len(prompts)} samples.")

    # 2. Tải các mô hình phục vụ đánh giá (Chỉ load LLM gốc để tính PPL)
    print("\nLoading models for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path, 
        device_map="auto",  # Dùng "auto" để né lỗi cấp phát nguyên khối (warmup bug)
        quantization_config=bnb_config
    ).eval()
    
    sbert_eval = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Khởi tạo mảng lưu kết quả
    metrics = {"uw_ppl": [], "w_ppl": [], "uw_rep": [], "w_rep": [], 
               "uw_d1": [], "w_d1": [], "uw_d2": [], "w_d2": [], "uw_ttr": [], "w_ttr": []}
    sbert_sims = []
    win_counts = {'Win': 0, 'Tie': 0, 'Loss': 0, 'Error': 0, 'N/A': 0}

    # ==========================================
    # BƯỚC A: ĐÁNH GIÁ LEXICAL DIVERSITY & PPL
    # ==========================================
    print("\n--- A. LEXICAL DIVERSITY & PPL ---")
    for prompt, uw, w in tqdm(zip(prompts, uw_texts, w_texts), total=len(prompts)):
        # PPL
        metrics["uw_ppl"].append(calculate_perplexity(model, tokenizer, prompt, uw, device))
        metrics["w_ppl"].append(calculate_perplexity(model, tokenizer, prompt, w, device))
        
        # Cắt suffix để đo Lexical công bằng
        s_uw = uw[len(prompt):].strip() if uw.startswith(prompt) else uw
        s_w = w[len(prompt):].strip() if w.startswith(prompt) else w
        
        # Rep-3 & Diversity
        metrics["uw_rep"].append(calculate_rep3(s_uw))
        metrics["w_rep"].append(calculate_rep3(s_w))
        metrics["uw_d1"].append(calculate_distinct_n(s_uw, 1))
        metrics["w_d1"].append(calculate_distinct_n(s_w, 1))
        metrics["uw_d2"].append(calculate_distinct_n(s_uw, 2))
        metrics["w_d2"].append(calculate_distinct_n(s_w, 2))
        metrics["uw_ttr"].append(calculate_log_ttr(s_uw))
        metrics["w_ttr"].append(calculate_log_ttr(s_w))

    # ==========================================
    # BƯỚC B: ĐÁNH GIÁ SEMANTIC SIMILARITY
    # ==========================================
    print("\n--- B. SEMANTIC SIMILARITY ---")
    for uw, w in tqdm(zip(uw_texts, w_texts), total=len(prompts), desc="SBERT Cosine"):
        e1 = sbert_eval.encode(uw, convert_to_tensor=True, show_progress_bar=False)
        e2 = sbert_eval.encode(w, convert_to_tensor=True, show_progress_bar=False)
        sbert_sims.append(util.cos_sim(e1, e2).item())

    print("Calculating BERTScore (Batch processing on GPU)...")
    _, _, F1 = bert_score(w_texts, uw_texts, lang="en", verbose=False, device=device)
    bert_scores = F1.tolist()

    # ==========================================
    # BƯỚC C: LLM-AS-A-JUDGE
    # ==========================================
    print("\n--- C. LLM-AS-A-JUDGE ---")
    if judge_client:
        for p, uw, w in tqdm(zip(prompts, uw_texts, w_texts), total=len(prompts)):
            res = evaluate_win_rate(p, uw, w)
            win_counts[res] += 1
    else:
        win_counts['N/A'] = len(prompts)
        print("Skipped LLM Judge (No API Key found).")

    # ==========================================
    # TỔNG HỢP & IN KẾT QUẢ
    # ==========================================
    print("\n==================================================")
    print("      TEXT QUALITY EVALUATION DASHBOARD           ")
    print("==================================================")
    
    print(f"\n[1] LEXICAL DIVERSITY & PPL")
    print(f"  - Base PPL (UW)    : {np.mean(metrics['uw_ppl']):.2f}")
    print(f"  - Delta PPL (WM-UW): {np.mean(metrics['w_ppl']) - np.mean(metrics['uw_ppl']):+.2f}")
    print(f"  - Rep-3 (%)        : WM = {np.mean(metrics['w_rep'])*100:.1f}% | UW = {np.mean(metrics['uw_rep'])*100:.1f}%")
    print(f"  - Distinct-1       : WM = {np.mean(metrics['w_d1']):.3f} | UW = {np.mean(metrics['uw_d1']):.3f}")
    print(f"  - Distinct-2       : WM = {np.mean(metrics['w_d2']):.3f} | UW = {np.mean(metrics['uw_d2']):.3f}")
    print(f"  - Log-TTR          : WM = {np.mean(metrics['w_ttr']):.3f} | UW = {np.mean(metrics['uw_ttr']):.3f}")
    
    print(f"\n[2] SEMANTIC FIDELITY")
    print(f"  - SBERT Cosine     : {np.mean(sbert_sims):.3f} / 1.0 (Higher is closer to original meaning)")
    print(f"  - BERTScore (F1)   : {np.mean(bert_scores):.3f} / 1.0")
    
    win_rate_pct = (win_counts['Win'] + win_counts['Tie']) / max(1, len(prompts) - win_counts['N/A'] - win_counts['Error']) * 100
    print(f"\n[3] LLM-AS-A-JUDGE (Human Preference)")
    print(f"  - Win+Tie Rate     : {win_rate_pct:.1f}%")
    print(f"  - Details          : Win: {win_counts['Win']} | Tie: {win_counts['Tie']} | Loss: {win_counts['Loss']}")
    print("==================================================\n")

if __name__ == "__main__":
    main()