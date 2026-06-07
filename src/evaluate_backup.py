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
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from openai import OpenAI
from dotenv import load_dotenv
import sys
import os
import csv
import json
import torch
import numpy as np
import random
import time # <-- Thêm thư viện này
from tqdm import tqdm

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from generator import WatermarkedGenerator
from detector import WatermarkDetector

# ==========================================
# 1. CÁC HÀM ĐÁNH GIÁ FIDELITY & ATTACKS
# ==========================================

def calculate_perplexity(model, tokenizer, prompt_text, generated_text, device):
    if generated_text.startswith(prompt_text):
        full_text = generated_text
    else:
        full_text = prompt_text + " " + generated_text

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

def measure_token_efficiency(detector, full_text, secret_key, tokenizer, target_threshold=4.0, prompt_len=0, thresholds=[50, 100, 150, 200, 300]):
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    for length in thresholds:
        if len(tokens) < length:
            continue
        truncated_text = tokenizer.decode(tokens[:length])
        res = detector.detect(truncated_text, secret_key, prompt_len=prompt_len)
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
    prompt = f"Rewrite this clearly:\n{text}\n\nRewritten:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    safe_max_tokens = min(150, int(len(text.split()) * 1.5)) 
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=safe_max_tokens, temperature=0.7, top_p=0.9, 
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        input_length = inputs["input_ids"].shape[1]
        rewritten = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    except Exception:
        rewritten = text
    finally:
        del inputs
        torch.cuda.empty_cache()
    return rewritten

# --- LLM-as-a-Judge ---
V98_API_KEY = os.environ.get("GPT-API-KEY", "")
JUDGE_MODEL = "gpt-5.2"
judge_client = OpenAI(base_url="https://v98store.com/v1", api_key=V98_API_KEY) if V98_API_KEY else None

def evaluate_win_rate(uw_text, w_text):
    if not judge_client: return "Tie"
    if random.random() < 0.5:
        text_a, text_b, watermarked_is = uw_text, w_text, "B"
    else:
        text_a, text_b, watermarked_is = w_text, uw_text, "A"

    prompt = (
        "You are an impartial text quality judge. Compare the two texts below "
        "on fluency, coherence, and informativeness.\n\n"
        f"--- Text A ---\n{text_a}\n\n"
        f"--- Text B ---\n{text_b}\n\n"
        "Which text is better overall? Reply with EXACTLY one word: A, B, or Tie."
    )
    try:
        resp = judge_client.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=5, temperature=0.0)
        answer = resp.choices[0].message.content.strip().upper()
        if "TIE" in answer: return "Tie"
        elif watermarked_is in answer: return "Win"
        else: return "Loss"
    except Exception:
        return "Tie"

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
    dataset_path = "data/large_dataset.jsonl"
    # ==========================================
    # TẢI ORACLE MODEL (ĐÁNH GIÁ PPL KHÁCH QUAN)
    # ==========================================
    oracle_model_name = "meta-llama/Llama-2-7b-hf" # Hoặc "google/gemma-7b" tùy bạn có sẵn model nào
    print(f"\nLoading Oracle Model ({oracle_model_name}) for independent PPL Evaluation...")
    oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
    if oracle_tokenizer.pad_token is None:
        oracle_tokenizer.pad_token = oracle_tokenizer.eos_token
        
    oracle_model = AutoModelForCausalLM.from_pretrained(
        oracle_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 # Ép kiểu 16-bit để tiết kiệm VRAM
    )
    oracle_model.eval()

    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): prompts.append(json.loads(line).get("prompt", ""))
    else:
        prompts = ["The future of artificial intelligence is"]
    
    TEST_LIMIT = 1000
    prompts = prompts[:TEST_LIMIT]

    unwatermarked_texts = []
    watermarked_texts = []
    
    # --- STEP 1: GENERATION ---
    eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'eval_data.csv')
    avg_gen_time = 0.0 # Thêm biến lưu thời gian

    if os.path.exists(eval_data_path):
        print(f"Found cached generations at {eval_data_path}, loading...")
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                unwatermarked_texts.append(row["uwm_text"])
                watermarked_texts.append(row["wm_text"])
            prompts = [row["prompt"] for row in csv.DictReader(open(eval_data_path, 'r', encoding='utf-8'))]
        avg_gen_time = float('nan') # Đánh dấu là N/A vì load từ cache
    else:
        print(f"\n--- STEP 1: GENERATION ({len(prompts)} prompts) ---")
        gen_times = [] # List lưu thời gian sinh
        
        for prompt in tqdm(prompts, desc="Generating"):
            unwatermarked_texts.append(generator.generate_unwatermarked(prompt))
            
            # Bắt đầu bấm giờ sinh Watermark
            start_time = time.time()
            watermarked_texts.append(generator.generate(prompt, secret_key))
            end_time = time.time()
            
            gen_times.append(end_time - start_time)

        avg_gen_time = np.mean(gen_times) # Tính trung bình

        with open(eval_data_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "uwm_text", "wm_text"])
            writer.writeheader()
            for p, uw, w in zip(prompts, unwatermarked_texts, watermarked_texts):
                writer.writerow({"prompt": p, "uwm_text": uw, "wm_text": w})
    # --- STEP 2: FIDELITY (PPL, Rep-3 & WIN-RATE) ---
    # --- STEP 2: FIDELITY (PPL & Rep-3) ---
    print("\n--- STEP 2: FIDELITY (PPL & Rep-3) ---")
    uw_ppls, w_ppls, uw_reps, w_reps = [], [], [], []
    
    for prompt, uw_text, w_text in tqdm(zip(prompts, unwatermarked_texts, watermarked_texts), total=len(prompts), desc="Fidelity Evals"):
        
        # DÙNG ORACLE ĐỂ CHẤM PPL THAY VÌ GENERATOR
        uw_ppls.append(calculate_perplexity(oracle_model, oracle_tokenizer, prompt, uw_text, generator.device))
        w_ppls.append(calculate_perplexity(oracle_model, oracle_tokenizer, prompt, w_text, generator.device))
        
        # Tách suffix để đo Rep-3 công bằng
        suffix_uw = uw_text[len(prompt):].strip() if uw_text.startswith(prompt) else uw_text
        suffix_w = w_text[len(prompt):].strip() if w_text.startswith(prompt) else w_text
        
        uw_reps.append(calculate_rep3(suffix_uw))
        w_reps.append(calculate_rep3(suffix_w))
        
    mean_uw_ppl, mean_w_ppl = np.mean(uw_ppls), np.mean(w_ppls)
    mean_uw_rep, mean_w_rep = np.mean(uw_reps), np.mean(w_reps)
    
    # --- STEP 3: DETECTABILITY (Z-SCORE, AUC & EFFICIENCY) ---
    print("\n--- STEP 3: DETECTABILITY ---")
    uw_z_scores, w_z_scores, efficiency_tokens = [], [], []
    det_times = [] # Thêm list lưu thời gian detect
    
    for prompt, uw_text, w_text in tqdm(zip(prompts, unwatermarked_texts, watermarked_texts), total=len(prompts), desc="Detectability Evals"):
        prompt_ids = generator.tokenizer.encode(prompt, add_special_tokens=True)
        p_len = len(prompt_ids)
        
        full_uw = uw_text if uw_text.startswith(prompt) else prompt + " " + uw_text
        full_w = w_text if w_text.startswith(prompt) else prompt + " " + w_text
        
        uw_z_scores.append(detector.detect(full_uw, secret_key, prompt_len=p_len).z_score)
        
        # Bắt đầu bấm giờ dò tìm Watermark
        start_time = time.time()
        w_z_scores.append(detector.detect(full_w, secret_key, prompt_len=p_len).z_score)
        end_time = time.time()
        
        det_times.append(end_time - start_time)
        
        eff = measure_token_efficiency(detector, full_w, secret_key, generator.tokenizer, target_threshold=4.0, prompt_len=p_len)
        if eff: efficiency_tokens.append(eff)

    mean_uw_z = np.mean(uw_z_scores)
    mean_w_z = np.mean(w_z_scores)
    auc_score = roc_auc_score([0]*len(uw_z_scores) + [1]*len(w_z_scores), uw_z_scores + w_z_scores)
    mean_efficiency = np.mean(efficiency_tokens) if efficiency_tokens else float('inf')
    mean_det_time = np.mean(det_times) # Tính trung bình thời gian detect
    # --- STEP 4: ROBUSTNESS (ATTACKS) ---
    print("\n--- STEP 4: ROBUSTNESS (ATTACKS) ---")
    attacks = {
        "No Attack": lambda t: t,
        "Deletion (10%)": lambda t: attack_word_deletion(t, 0.1),
        "Deletion (30%)": lambda t: attack_word_deletion(t, 0.3),
        "Swap (10%)": lambda t: attack_word_swap(t, 0.1),
        "Paraphrase (LLM)": lambda t: attack_llm_paraphrase(t, generator.model, generator.tokenizer, generator.device)
    }

    attack_results = {name: [] for name in attacks}

    print("Running Attacks (Strictly attacking generated suffix only)...")
    for prompt, w_text in tqdm(zip(prompts, watermarked_texts), desc="Robustness Evals", total=len(prompts)):
        prompt_ids = generator.tokenizer.encode(prompt, add_special_tokens=True)
        p_len = len(prompt_ids)
        
        # Chỉ tấn công phần sinh ra, giữ nguyên Prompt để truyền vào Detector
        suffix_w = w_text[len(prompt):].strip() if w_text.startswith(prompt) else w_text
        
        for name, attack_fn in attacks.items():
            attacked_suffix = attack_fn(suffix_w)
            full_attacked = prompt + " " + attacked_suffix
            z_score = detector.detect(full_attacked, secret_key, prompt_len=p_len).z_score
            attack_results[name].append(z_score)

    print("\n==================================================")
    print("      SAE-OLS ADVANCED EVALUATION RESULTS         ")
    print("==================================================")
    print(f"\n[1] FIDELITY METRICS (Utility)")
    print(f"  - Delta PPL        : {mean_w_ppl - mean_uw_ppl:+.2f} (Base: {mean_uw_ppl:.2f})")
    print(f"  - Rep-3 (%)        : WM = {mean_w_rep*100:.1f}% | UW = {mean_uw_rep*100:.1f}%")
    
    print(f"\n[2] DETECTABILITY METRICS (Security)")
    print(f"  - Mean Z-Score     : {mean_w_z:.2f} (UW: {mean_uw_z:.2f})")
    print(f"  - ROC-AUC Score    : {auc_score:.4f}")
    print(f"  - Token Efficiency : ~{mean_efficiency:.1f} tokens to reach Z>4")
    
    print("\n[3] ROBUSTNESS METRICS (Under Attack)")
    baseline_z = np.mean(attack_results["No Attack"])
    for name, scores in attack_results.items():
        mean_z = np.mean(scores)
        survived_pct = sum(1 for z in scores if z >= 4.0) / len(scores) * 100
        z_drop = (1 - mean_z / baseline_z) * 100 if baseline_z > 0 else 0.0
        print(f"  {name:<25s} | Z= {mean_z:5.2f} | Drop: {z_drop:5.1f}% | Survived(Z>4): {survived_pct:5.1f}%")
        
    # Thêm mục in thời gian tại đây
    print(f"\n[4] LATENCY METRICS (Speed)")
    gen_time_str = f"{avg_gen_time:.2f} s / prompt" if not np.isnan(avg_gen_time) else "N/A (Loaded from cache)"
    print(f"  - Avg Generation Time: {gen_time_str}")
    print(f"  - Avg Detection Time : {mean_det_time:.2f} s / prompt")
    print("==================================================\n")
    # --- SAVE RESULTS TO CSV ---
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'eval.csv')
    file_exists = os.path.exists(csv_path)

    # Đảm bảo columns khớp chính xác với Baseline script
    fieldnames = [
        "WM", "Delta_PPL", "Base_PPL", "Rep3_Score_%", "Mean_Z_Score", "UW_Z_Score", 
        "ROC_AUC", "Avg_Tokens_to_Z4", "No_Attack_Z", "Del_10_Z", "Del_30_Z", "Swap_10_Z", "Para_Z"
    ]
    
    row = {
        "WM": "SAE-OLS",
        "Delta_PPL": f"{mean_w_ppl - mean_uw_ppl:+.2f}",
        "Base_PPL": f"{mean_uw_ppl:.2f}",
        "Rep3_Score_%": f"{mean_w_rep*100:.1f}",
        "Mean_Z_Score": f"{mean_w_z:.2f}",
        "UW_Z_Score": f"{mean_uw_z:.2f}",
        "ROC_AUC": f"{auc_score:.4f}",
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

    print(f"Results successfully appended to {csv_path}")

if __name__ == "__main__":
    main()