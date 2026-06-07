"""
Advanced Evaluation Script for Baseline: KGW (Kirchenbauer et al.) Watermark.
Measures Fidelity (PPL, Rep-3), Detectability, and Full Robustness (Attacks).
"""

import sys
import os
import csv
import json
import torch
import numpy as np
import random
import math
import hashlib
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, BitsAndBytesConfig

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- THUẬT TOÁN KGW ---
class KGWLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.25, delta=4.0, context_width=5): # <-- Đã tăng delta lên 5.0
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.context_width = context_width
        self.rng = torch.Generator(device='cpu')

    def _get_green_list(self, context):
        if torch.is_tensor(context):
            context_list = context.tolist()
        else:
            context_list = list(context)
            
        context_str = "_".join(map(str, context_list))
        context_bytes = context_str.encode('utf-8')
        hash_hex = hashlib.sha256(context_bytes).hexdigest()
        seed = int(hash_hex[:8], 16)
        
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist_size = int(self.vocab_size * self.gamma)
        return vocab_permutation[:greenlist_size]

    def __call__(self, input_ids, scores):
        for batch_idx in range(input_ids.shape[0]):
            context = input_ids[batch_idx][-self.context_width:]
            greenlist_ids = self._get_green_list(context)
            scores[batch_idx, greenlist_ids] += self.delta
        return scores

class KGWDetector:
    def __init__(self, vocab_size, gamma=0.25, context_width=5):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.context_width = context_width
        self.rng = torch.Generator(device='cpu')

    def _get_green_list(self, context):
        if torch.is_tensor(context):
            context_list = context.tolist()
        else:
            context_list = list(context)
            
        context_str = "_".join(map(str, context_list))
        context_bytes = context_str.encode('utf-8')
        hash_hex = hashlib.sha256(context_bytes).hexdigest()
        seed = int(hash_hex[:8], 16)
        
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist_size = int(self.vocab_size * self.gamma)
        return vocab_permutation[:greenlist_size]

    class Result:
        def __init__(self, z_score):
            self.z_score = z_score

    # ĐỘT PHÁ VÁ LỖI BPE: Nhận Prompt và Generated Text rành mạch
    def detect(self, prompt_text, generated_text, tokenizer):
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        gen_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
        
        # Nối danh sách Token ID thay vì nối chuỗi chữ (Tránh bị gộp từ)
        tokens_tensor = torch.tensor(prompt_tokens + gen_tokens)
        N = len(tokens_tensor)
        start_idx = len(prompt_tokens)
        
        if N <= start_idx + self.context_width: return self.Result(0.0)

        green_tokens_count = 0
        valid_tokens = N - start_idx

        for t in range(start_idx, N):
            context = tokens_tensor[t - self.context_width : t]
            current_token = tokens_tensor[t].item()
            greenlist_ids = self._get_green_list(context)
            if current_token in greenlist_ids:
                green_tokens_count += 1

        expected_green = self.gamma * valid_tokens
        variance = self.gamma * (1 - self.gamma) * valid_tokens
        z_score = (green_tokens_count - expected_green) / math.sqrt(variance) if variance > 0 else 0.0
        return self.Result(z_score)

# --- CÁC HÀM ĐÁNH GIÁ (FIDELITY & METRICS) ---
def calculate_perplexity(model, tokenizer, prompt_text, generated_text, device):
    full_text = prompt_text + " " + generated_text
    prompt_len = tokenizer(prompt_text, return_tensors="pt").input_ids.size(1)
    encodings = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    if input_ids.size(1) <= prompt_len: return 0.0

    target_ids = input_ids.clone()
    target_ids[:, :prompt_len] = -100
    with torch.no_grad():
        loss = model(input_ids, labels=target_ids).loss
    return torch.exp(loss).item()

def calculate_rep3(text: str) -> float:
    words = text.split()
    if len(words) < 3: return 0.0
    ngrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def measure_token_efficiency(detector, w_text, prompt, tokenizer, target_threshold, thresholds=[50, 100, 150, 200, 300]):
    w_tokens = tokenizer.encode(w_text, add_special_tokens=False)
    for length in thresholds:
        if len(w_tokens) < length: continue
        trunc_w_text = tokenizer.decode(w_tokens[:length], skip_special_tokens=True)
        res = detector.detect(prompt, trunc_w_text, tokenizer)
        if res.z_score >= target_threshold: return length
    return None

# --- CÁC HÀM TẤN CÔNG (ROBUSTNESS) ---
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
        {"role": "system", "content": "You are a precise paraphrasing tool. Rewrite the user's text to change its vocabulary and sentence structure while preserving the exact original meaning. Output ONLY the paraphrased text."},
        {"role": "user", "content": text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    safe_max_tokens = min(500, int(len(text.split()) * 3.0)) 
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

def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 400

    print("Loading models for KGW Baseline...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=bnb_config)
    
    vocab_size = model.config.vocab_size
    kgw_processor = KGWLogitsProcessor(vocab_size=vocab_size, gamma=0.25, delta=5.0) # Đã nâng Delta
    detector = KGWDetector(vocab_size=vocab_size, gamma=0.25)
    
    raw_prompts = []
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'alpaca_eval_10k.jsonl')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'large_dataset.jsonl')
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): raw_prompts.append(json.loads(line).get("prompt", ""))
    else:
        raw_prompts = ["The future of artificial intelligence is"]
        
    system_inst = "You are a highly detailed academic assistant. You MUST provide exhaustive, multi-paragraph answers (minimum 250 words)."
    prompts = [
        tokenizer.apply_chat_template([{"role": "system", "content": system_inst}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in raw_prompts
    ]

    TEST_LIMIT = int(os.environ.get('EVAL_TEST_LIMIT', 1000))
    prompts = prompts[:TEST_LIMIT]
    unwatermarked_texts, watermarked_texts = [], []

    # --- STEP 1: GENERATION ---
    eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'eval_data_kgw.csv')
    if os.path.exists(eval_data_path):
        print(f"Found cached generations at {eval_data_path}, loading...")
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                unwatermarked_texts.append(row["uwm_text"])
                watermarked_texts.append(row["wm_text"])
            prompts = [row["prompt"] for row in csv.DictReader(open(eval_data_path, 'r', encoding='utf-8'))]
    else:
        print(f"\n--- STEP 1: GENERATION ({len(prompts)} prompts) ---")
        for prompt in tqdm(prompts, desc="Generating (KGW)"):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                uw_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
                w_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, logits_processor=LogitsProcessorList([kgw_processor]))
            
            p_len = inputs.input_ids.shape[1]
            unwatermarked_texts.append(tokenizer.decode(uw_out[0][p_len:], skip_special_tokens=True))
            watermarked_texts.append(tokenizer.decode(w_out[0][p_len:], skip_special_tokens=True))
            
            del inputs, uw_out, w_out
            torch.cuda.empty_cache()

        with open(eval_data_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "uwm_text", "wm_text"])
            writer.writeheader()
            for p, uw, w in zip(prompts, unwatermarked_texts, watermarked_texts):
                writer.writerow({"prompt": p, "uwm_text": uw, "wm_text": w})

    # --- STEP 2: FIDELITY (PPL & REP-3) ---
    print("\n--- STEP 2: FIDELITY (PPL & Rep-3) ---")
    uw_ppls, w_ppls, uw_reps, w_reps = [], [], [], []
    for prompt, uw_text, w_text in tqdm(zip(prompts, unwatermarked_texts, watermarked_texts), total=len(prompts)):
        uw_ppls.append(calculate_perplexity(model, tokenizer, prompt, uw_text, device))
        w_ppls.append(calculate_perplexity(model, tokenizer, prompt, w_text, device))
        uw_reps.append(calculate_rep3(uw_text))
        w_reps.append(calculate_rep3(w_text))
        
    mean_uw_ppl, mean_w_ppl = np.mean(uw_ppls), np.mean(w_ppls)
    mean_uw_rep, mean_w_rep = np.mean(uw_reps), np.mean(w_reps)
    
    # --- STEP 3: DETECTABILITY ---
    print("\n--- STEP 3: DETECTABILITY ---")
    uw_z_scores, w_z_scores = [], []
    for prompt, uw_text, w_text in tqdm(zip(prompts, unwatermarked_texts, watermarked_texts), total=len(prompts)):
        uw_z_scores.append(detector.detect(prompt, uw_text, tokenizer).z_score)
        w_z_scores.append(detector.detect(prompt, w_text, tokenizer).z_score)

    mean_uw_z, mean_w_z = np.mean(uw_z_scores), np.mean(w_z_scores)
    empirical_threshold = 4.0
    auc_score = roc_auc_score([0]*len(uw_z_scores) + [1]*len(w_z_scores), uw_z_scores + w_z_scores)

    efficiency_tokens = []
    for prompt, w_text in tqdm(zip(prompts, watermarked_texts), desc="Testing Efficiency", total=len(prompts)):
        eff = measure_token_efficiency(detector, w_text, prompt, tokenizer, target_threshold=empirical_threshold)
        if eff: efficiency_tokens.append(eff)

    mean_eff = np.mean(efficiency_tokens) if efficiency_tokens else float('inf')
    
    print(f"\n[KGW DETECTABILITY METRICS]")
    print(f"  - UW Z-Score (Null)  : {mean_uw_z:.2f}")
    print(f"  - WM Z-Score         : {mean_w_z:.2f}")
    print(f"  - Token Efficiency   : ~{mean_eff:.1f} tokens")
    print(f"  - ROC-AUC            : {auc_score:.4f}")

    print(f"\n[KGW FIDELITY OVERVIEW]")
    print(f"  - Delta PPL: {mean_w_ppl - mean_uw_ppl:+.2f} | Base PPL: {mean_uw_ppl:.2f}")
    print(f"  - Rep-3 (%) : WM = {mean_w_rep*100:.1f}% | UW = {mean_uw_rep*100:.1f}%")

    # --- STEP 4: ROBUSTNESS (ATTACKS) ---
    print("\n--- STEP 4: ROBUSTNESS (ATTACKS) ---")
    attacks = {
        "No Attack": lambda t: t,
        "Deletion (10%)": lambda t: attack_word_deletion(t, 0.1),
        "Deletion (30%)": lambda t: attack_word_deletion(t, 0.3),
        "Swap (10%)": lambda t: attack_word_swap(t, 0.1),
        "Paraphrase (LLM)": lambda t: attack_llm_paraphrase(t, model, tokenizer, device)
    }

    attack_results = {name: [] for name in attacks}

    for prompt, w_text in tqdm(zip(prompts, watermarked_texts), desc="Robustness Evals", total=len(prompts)):
        for name, attack_fn in attacks.items():
            attacked_text = attack_fn(w_text)
            attack_results[name].append(detector.detect(prompt, attacked_text, tokenizer).z_score)

    print("\n[4] ROBUSTNESS METRICS (Under Attack)")
    baseline_z = np.mean(attack_results["No Attack"])
    
    for name, scores in attack_results.items():
        mean_z = np.mean(scores)
        survived_pct = sum(1 for z in scores if z >= 4.0) / len(scores) * 100
        z_drop = (1 - mean_z / baseline_z) * 100 if baseline_z > 0 else 0.0
        print(f"  {name:<25s} | Z= {mean_z:5.2f} | Drop: {z_drop:5.1f}% | Survived(Z>4): {survived_pct:5.1f}%")
    print("==================================================\n")

    # --- LƯU VÀO eval.csv ---
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'eval.csv')
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "WM", "Delta_PPL", "Base_PPL", "Rep3_Score_%", "Mean_Z_Score", "UW_Z_Score", 
        "ROC_AUC", "Avg_Tokens_to_Z4", "No_Attack_Z", "Del_10_Z", "Del_30_Z", "Swap_10_Z", "Para_Z"
    ]
    row = {
        "WM": "KGW",
        "Delta_PPL": f"{mean_w_ppl - mean_uw_ppl:+.2f}",
        "Base_PPL": f"{mean_uw_ppl:.2f}",
        "Rep3_Score_%": f"{mean_w_rep*100:.1f}",
        "Mean_Z_Score": f"{mean_w_z:.2f}",
        "UW_Z_Score": f"{mean_uw_z:.2f}",
        "ROC_AUC": f"{auc_score:.4f}",
        "Avg_Tokens_to_Z4": f"{mean_eff:.1f}" if efficiency_tokens else "N/A",
        "No_Attack_Z": f"{baseline_z:.2f}",
        "Del_10_Z": f"{np.mean(attack_results['Deletion (10%)']):.2f}",
        "Del_30_Z": f"{np.mean(attack_results['Deletion (30%)']):.2f}",
        "Swap_10_Z": f"{np.mean(attack_results['Swap (10%)']):.2f}",
        "Para_Z": f"{np.mean(attack_results['Paraphrase (LLM)']):.2f}",
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    main()