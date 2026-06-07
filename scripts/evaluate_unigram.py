"""
Advanced Evaluation Script for Baseline: Unigram Watermark (Zhao et al., 2024).
Measures Fidelity (PPL, Rep-3), Detectability, and Full Robustness (Attacks).
"""

import sys
import os
import csv
import json
import torch
import numpy as np
import random
import hashlib
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- THUẬT TOÁN UNIGRAM (BÁM SÁT REPO GỐC) ---
class UnigramLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, watermark_key: int, gamma=0.5, delta=2.0):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.watermark_key = watermark_key
        
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(gamma * vocab_size) + [False] * (vocab_size - int(gamma * vocab_size)))
        rng.shuffle(mask)
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)

    @staticmethod
    def _hash_fn(x: int) -> int:
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        watermark = self.delta * self.green_list_mask
        new_logits = scores + watermark.to(scores.device)
        return new_logits

class UnigramDetector:
    def __init__(self, vocab_size, watermark_key: int, gamma=0.5):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.watermark_key = watermark_key
        
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(gamma * vocab_size) + [False] * (vocab_size - int(gamma * vocab_size)))
        rng.shuffle(mask)
        self.green_list_mask = mask 

    @staticmethod
    def _hash_fn(x: int) -> int:
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        if total == 0: return 0.0
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)

    class Result:
        def __init__(self, z_score):
            self.z_score = z_score

    def detect(self, text, secret_key_ignored, tokenizer):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 0: return self.Result(0.0)

        unique_sequence = list(set(tokens))
        green_tokens = int(sum(self.green_list_mask[i] for i in unique_sequence))
        z_score = self._z_score(green_tokens, len(unique_sequence), self.gamma)
        return self.Result(z_score)

# --- CÁC HÀM ĐÁNH GIÁ (FIDELITY & METRICS) ---
def calculate_perplexity(model, tokenizer, prompt_text, generated_text, device):
    full_text = generated_text if generated_text.startswith(prompt_text) else prompt_text + generated_text
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
    """Tính tỷ lệ lặp 3-gram (0.0 = không lặp, 1.0 = lặp hoàn toàn)"""
    words = text.split()
    if len(words) < 3: return 0.0
    ngrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def measure_token_efficiency(detector, text, tokenizer, target_threshold, thresholds=[50, 100, 150, 200, 300]):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    for length in thresholds:
        if len(tokens) < length: continue
        res = detector.detect(tokenizer.decode(tokens[:length]), None, tokenizer)
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
    prompt = f"Rewrite the following paragraph using different words but keeping the exact same meaning:\n\n{text}\n\nRewritten paragraph:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    word_count = len(text.split())
    safe_max_tokens = int(word_count * 2) + 50 

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=safe_max_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    input_length = inputs["input_ids"].shape[1]
    rewritten_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    del inputs, outputs
    torch.cuda.empty_cache()
    return rewritten_text.strip()


def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 400
    secret_key = 42

    print("Loading models for Unigram Baseline...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map=device, 
        quantization_config=bnb_config
    )
    
    vocab_size = model.config.vocab_size
    unigram_processor = UnigramLogitsProcessor(vocab_size=vocab_size, watermark_key=secret_key, gamma=0.5, delta=2.0)
    detector = UnigramDetector(vocab_size=vocab_size, watermark_key=secret_key, gamma=0.5)
    
    raw_prompts = []
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'alpaca_eval_10k.jsonl')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'large_dataset.jsonl')
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): raw_prompts.append(json.loads(line).get("prompt", ""))
    else:
        raw_prompts = ["The future of artificial intelligence is"]
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in raw_prompts
    ]

    TEST_LIMIT = int(os.environ.get('EVAL_TEST_LIMIT', 1000))
    prompts = prompts[:TEST_LIMIT]
    unwatermarked_texts, watermarked_texts = [], []

    # --- STEP 1: GENERATION ---
    eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'eval_data_unigram.csv')
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
        with open(eval_data_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "uwm_text", "wm_text"])
            writer.writeheader()
            for prompt in tqdm(prompts, desc="Generating (Unigram)"):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    uw_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
                    w_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, logits_processor=LogitsProcessorList([unigram_processor]))

                p_len = inputs.input_ids.shape[1]
                uw_text = tokenizer.decode(uw_out[0][p_len:], skip_special_tokens=True)
                w_text = tokenizer.decode(w_out[0][p_len:], skip_special_tokens=True)
                unwatermarked_texts.append(uw_text)
                watermarked_texts.append(w_text)
                writer.writerow({"prompt": prompt, "uwm_text": uw_text, "wm_text": w_text})
                f.flush()
       
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
    for uw_text, w_text in tqdm(zip(unwatermarked_texts, watermarked_texts), total=len(prompts)):
        uw_z_scores.append(detector.detect(uw_text, None, tokenizer).z_score)
        w_z_scores.append(detector.detect(w_text, None, tokenizer).z_score)

    mean_uw_z, mean_w_z = np.mean(uw_z_scores), np.mean(w_z_scores)
    empirical_threshold = 4.0
    auc_score = roc_auc_score([0]*len(uw_z_scores) + [1]*len(w_z_scores), uw_z_scores + w_z_scores)

    efficiency_tokens = []
    for w_text in tqdm(watermarked_texts, desc="Testing Efficiency"):
        eff = measure_token_efficiency(detector, w_text, tokenizer, target_threshold=empirical_threshold)
        if eff: efficiency_tokens.append(eff)

    mean_eff = np.mean(efficiency_tokens) if efficiency_tokens else float('inf')
    
    tpr = sum(1 for z in w_z_scores if z > empirical_threshold) / len(w_z_scores) if w_z_scores else 0.0
    tnr = sum(1 for z in uw_z_scores if z <= empirical_threshold) / len(uw_z_scores) if uw_z_scores else 0.0
    f1_score = (2 * tpr * tnr) / (tpr + tnr) if (tpr + tnr) > 0 else 0.0

    print(f"\n[UNIGRAM DETECTABILITY METRICS]")
    print(f"  - UW Z-Score (Null)  : {mean_uw_z:.2f}")
    print(f"  - WM Z-Score         : {mean_w_z:.2f}")
    print(f"  - Token Efficiency   : ~{mean_eff:.1f} tokens")
    print(f"  - ROC-AUC / F1 Score : {auc_score:.4f} / {f1_score:.4f}")

    print(f"\n[UNIGRAM FIDELITY OVERVIEW]")
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

    for w_text in tqdm(watermarked_texts, desc="Robustness Evals"):
        for name, attack_fn in attacks.items():
            attacked_text = attack_fn(w_text)
            attack_results[name].append(detector.detect(attacked_text, None, tokenizer).z_score)

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
        "WM": "Unigram",
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
    print(f"Appended Unigram results to {csv_path}")

if __name__ == "__main__":
    main()