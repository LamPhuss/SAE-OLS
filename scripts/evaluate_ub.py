"""
Advanced Evaluation Script for Baseline: UB (Unbiased Watermark).
Implements Delta-Reweight Exact Match Z-Score based on the Hu et al., 2024 architecture.
"""

import sys
import os
import csv
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
import math
import hashlib
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import re

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ==========================================
# 1. THUẬT TOÁN UNBIASED WATERMARK (DELTA REWEIGHT)
# ==========================================

class Delta_WatermarkCode:
    def __init__(self, u: torch.FloatTensor):
        self.u = u

    @classmethod
    def from_random(cls, rng: list, device: torch.device):
        batch_size = len(rng)
        u = torch.stack([
            torch.rand((), generator=rng[i], device='cpu')
            for i in range(batch_size)
        ]).to(device)
        return cls(u)

class Delta_Reweight:
    def reweight_logits(self, code: Delta_WatermarkCode, p_logits: torch.FloatTensor) -> torch.FloatTensor:
        probs = F.softmax(p_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        u = code.u.unsqueeze(-1)
        
        # Tìm token mục tiêu thông qua Inverse Transform Sampling
        index = torch.searchsorted(cumsum, u, right=True)
        index = torch.clamp(index, 0, p_logits.shape[-1] - 1)
        
        # Ép xác suất của token mục tiêu lên 100%, những token khác về 0%
        modified_logits = torch.full_like(p_logits, float("-inf"))
        modified_logits.scatter_(-1, index, 0.0)
        return modified_logits

class UBLogitsProcessor(LogitsProcessor):
    def __init__(self, private_key: str, context_width: int=1, temp: float=0.7):
        self.private_key = private_key.encode('utf-8')
        self.reweight = Delta_Reweight()
        self.context_width = context_width
        self.temp = temp
        
    def get_rng_seed(self, context_list: list) -> int:
        context_str = "_".join(map(str, context_list))
        m = hashlib.sha256()
        m.update(context_str.encode('utf-8'))
        m.update(self.private_key)
        return int.from_bytes(m.digest()[:8], "big") % (2**32 - 1)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        context_lists = [input_ids[i][-self.context_width:].tolist() for i in range(batch_size)]
        seeds = [self.get_rng_seed(cl) for cl in context_lists]
        
        rng = [torch.Generator(device='cpu').manual_seed(seed) for seed in seeds]
        watermark_code = Delta_WatermarkCode.from_random(rng, scores.device)
        
        # Đảm bảo phân phối được tính toán chuẩn theo Temperature
        scaled_scores = scores.float() / self.temp
        reweighted_scores = self.reweight.reweight_logits(watermark_code, scaled_scores)
        
        return reweighted_scores * self.temp

class UBDetector:
    def __init__(self, model, vocab_size: int, private_key: str, context_width: int=1):
        self.model = model
        self.vocab_size = vocab_size
        self.private_key = private_key.encode('utf-8')
        self.context_width = context_width

    def get_rng_seed(self, context_list: list) -> int:
        context_str = "_".join(map(str, context_list))
        m = hashlib.sha256()
        m.update(context_str.encode('utf-8'))
        m.update(self.private_key)
        return int.from_bytes(m.digest()[:8], "big") % (2**32 - 1)

    class Result:
        def __init__(self, z_score):
            self.z_score = z_score

    def detect(self, full_text: str, tokenizer, prompt_len: int, temp: float=0.7):
        """
        Exact Match Z-Score: Đo lường độ lệch giữa số lần token thực tế khớp...
        """
        # [SỬA LỖI 1]: Dùng chung tokenizer(...) như lúc generation thay vì .encode
        # để bảo toàn được cấu trúc special tokens của Llama-3
        inputs = tokenizer(full_text, return_tensors="pt").to(self.model.device)
        tokens = inputs.input_ids[0].tolist()
        N = len(tokens)
        
        if N <= prompt_len + self.context_width: 
            return self.Result(0.0)

        with torch.no_grad():
            outputs = self.model(inputs.input_ids)

        matches = 0
        expected_matches = 0.0
        variance_matches = 0.0

        start_idx = max(prompt_len, self.context_width)
        
        for i in range(start_idx, N):
            context_list = tokens[i - self.context_width : i]
            current_token = tokens[i]

            original_logits = outputs.logits[0, i-1].float()
            scaled_logits = original_logits / temp
            probs = F.softmax(scaled_logits, dim=-1)
            
            p_val = probs[current_token].item()
            expected_matches += p_val
            variance_matches += p_val * (1.0 - p_val)

            seed = self.get_rng_seed(context_list)
            rng = torch.Generator(device='cpu').manual_seed(seed)
            u = torch.rand((1,), generator=rng).to(self.model.device)
            
            cumsum = torch.cumsum(probs, dim=-1)
            
            # [SỬA LỖI 3]: Bỏ u.unsqueeze(-1) vì u và cumsum đều đang ở hệ 1D
            target_index = torch.searchsorted(cumsum, u, right=True)
            target_index = torch.clamp(target_index, 0, self.vocab_size - 1).item()
            
            if target_index == current_token:
                matches += 1

        if variance_matches == 0:
            return self.Result(0.0)

        z_score = (matches - expected_matches) / math.sqrt(variance_matches)
        return self.Result(z_score)

# ==========================================
# 2. CÁC HÀM ĐÁNH GIÁ
# ==========================================
def clean_filler_text(text: str) -> str:
    """Loại bỏ các câu rườm rà (conversational filler) thường gặp của LLM khi paraphrase."""
    # Các pattern phổ biến mà Llama/ChatGPT hay dùng để mở đầu
    patterns = [
        r"^(Here is|Here's) .*?(rewritten|paraphrased|modified|version|text|response)[\s\S]*?:\s*\n*",
        r"^(Sure|Certainly),? .*?(here is|I can).*?\n+",
        r"^I have .*?:\s*\n*",
        r"^\*\*Rewritten.*?\*\*\s*\n*"
    ]
    cleaned_text = text.strip()
    for pattern in patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE).strip()
    return cleaned_text

def calculate_perplexity(model, tokenizer, prompt_text, generated_text, device):
    full_text = generated_text if generated_text.startswith(prompt_text) else prompt_text + generated_text
    prompt_len = tokenizer(prompt_text, return_tensors="pt").input_ids.size(1)
    encodings = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    if input_ids.size(1) <= prompt_len: return 0.0

    target_ids = input_ids.clone()
    target_ids[:, :prompt_len] = -100
    with torch.no_grad(): loss = model(input_ids, labels=target_ids).loss
    return torch.exp(loss).item()

def calculate_rep3(text: str) -> float:
    words = text.split()
    if len(words) < 3: return 0.0
    ngrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def measure_token_efficiency(detector, w_text, prompt, tokenizer, target_threshold, thresholds=[50, 100, 150, 200, 300]):
    # [SỬA LỖI 1]: Đồng bộ cách đếm p_len
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    p_len = prompt_inputs.input_ids.shape[1]
    
    w_tokens = tokenizer(w_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    
    for length in thresholds:
        if len(w_tokens) < length: continue
        # [SỬA LỖI 2]: Bỏ khoảng trắng dư thừa lúc ghép text
        full_trunc_text = prompt + tokenizer.decode(w_tokens[:length])
        res = detector.detect(full_trunc_text, tokenizer, p_len, temp=0.7)
        if res.z_score >= target_threshold: return length
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
    # Dùng format chat chuẩn của Llama 3 và ép nó bằng System Prompt
    messages = [
        {
            "role": "system", 
            "content": "You are a text paraphraser. Rewrite the user's text clearly. CRITICAL INSTRUCTION: Output ONLY the exact paraphrased text. Do NOT include any conversational filler, introductions, greetings, or conclusions like 'Here is the rewritten version'."
        },
        {
            "role": "user", 
            "content": text
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    safe_max_tokens = min(250, int(len(text.split()) * 1.5)) 
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=safe_max_tokens, 
                temperature=0.3,  # Giảm temp để model tuân thủ system prompt tốt hơn
                top_p=0.9, 
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        input_length = inputs["input_ids"].shape[1]
        rewritten = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        # Bước lọc rác: Xóa bỏ các câu dẫn (nếu model vẫn lỡ sinh ra)
        rewritten = clean_filler_text(rewritten)
        
    except Exception as e:
        print(f"Paraphrase failed: {e}")
        rewritten = text
    finally:
        del inputs
        torch.cuda.empty_cache()
        
    return rewritten

def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 400
    secret_key = "ub_secure_key_2026"

    print("Loading models for Unbiased Watermark Baseline...")
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
    # context_width = 1 giúp hệ thống có khả năng tự phục hồi (resilient) khi bị xóa từ
    ub_processor = UBLogitsProcessor(private_key=secret_key, context_width=1, temp=0.7)
    detector = UBDetector(model=model, vocab_size=vocab_size, private_key=secret_key, context_width=1)
    
    raw_prompts = []
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data/alpaca_eval_10k.jsonl')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data/large_dataset.jsonl')
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

    eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'eval_data_ub_llama3_1000.csv')
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
        for prompt in tqdm(prompts, desc="Generating (UB)"):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                uw_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
                w_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, logits_processor=LogitsProcessorList([ub_processor]))
            
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

    print("\n--- STEP 2: FIDELITY (PPL & Rep-3) ---")
    uw_ppls, w_ppls, uw_reps, w_reps = [], [], [], []
    for prompt, uw_text, w_text in tqdm(zip(prompts, unwatermarked_texts, watermarked_texts), total=len(prompts)):
        uw_ppls.append(calculate_perplexity(model, tokenizer, prompt, uw_text, device))
        w_ppls.append(calculate_perplexity(model, tokenizer, prompt, w_text, device))
        uw_reps.append(calculate_rep3(uw_text))
        w_reps.append(calculate_rep3(w_text))
        
    mean_uw_ppl, mean_w_ppl = np.mean(uw_ppls), np.mean(w_ppls)
    mean_uw_rep, mean_w_rep = np.mean(uw_reps), np.mean(w_reps)
    
    print("\n--- STEP 3: DETECTABILITY ---")
    uw_z_scores, w_z_scores = [], []
    for prompt, uw_text, w_text in tqdm(zip(prompts, unwatermarked_texts, watermarked_texts), total=len(prompts)):
        # Tính p_len chuẩn như lúc model generation
        p_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        
        # [SỬA LỖI 2]: Khắc phục lỗi ghép chuỗi dư dấu cách " "
        full_uw = uw_text if uw_text.startswith(prompt) else prompt + uw_text
        full_w = w_text if w_text.startswith(prompt) else prompt + w_text
        
        uw_z_scores.append(detector.detect(full_uw, tokenizer, p_len, temp=0.7).z_score)
        w_z_scores.append(detector.detect(full_w, tokenizer, p_len, temp=0.7).z_score)

    mean_uw_z, mean_w_z = np.mean(uw_z_scores), np.mean(w_z_scores)
    
    empirical_threshold = 4.0
    auc_score = roc_auc_score([0]*len(uw_z_scores) + [1]*len(w_z_scores), uw_z_scores + w_z_scores)
    
    efficiency_tokens = []
    for prompt, w_text in tqdm(zip(prompts, watermarked_texts), desc="Testing Efficiency", total=len(prompts)):
        eff = measure_token_efficiency(detector, w_text, prompt, tokenizer, target_threshold=empirical_threshold)
        if eff: efficiency_tokens.append(eff)
    mean_eff = np.mean(efficiency_tokens) if efficiency_tokens else float('inf')

    print(f"\n[UB DETECTABILITY METRICS]")
    print(f"  - UW Z-Score (Null)   : {mean_uw_z:.2f} (Std: {np.std(uw_z_scores):.2f})")
    print(f"  - WM Z-Score          : {mean_w_z:.2f}")
    print(f"  - Token Efficiency    : ~{mean_eff:.1f} tokens")
    print(f"  - ROC-AUC Score       : {auc_score:.4f}")

    print(f"\n[UB FIDELITY OVERVIEW]")
    print(f"  - Delta PPL: {mean_w_ppl - mean_uw_ppl:+.2f} | Base PPL: {mean_uw_ppl:.2f}")
    print(f"  - Rep-3 (%) : WM = {mean_w_rep*100:.1f}% | UW = {mean_uw_rep*100:.1f}%")

    print("\n--- STEP 4: ROBUSTNESS (ATTACKS) ---")
    attacks = {
        "No Attack": lambda t: t,
        "Deletion (10%)": lambda t: attack_word_deletion(t, 0.1),
        "Deletion (30%)": lambda t: attack_word_deletion(t, 0.3),
        "Swap (10%)": lambda t: attack_word_swap(t, 0.1),
        "Paraphrase (LLM)": lambda t: attack_llm_paraphrase(t, model, tokenizer, device)
    }

    attack_results = {name: [] for name in attacks}

    print("Running Attacks (Paraphrase is optimized and protected)...")
    for prompt, w_text in tqdm(zip(prompts, watermarked_texts), desc="Robustness Evals", total=len(prompts)):
        # Tính lại p_len đúng chuẩn
        p_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        
        for name, attack_fn in attacks.items():
            attacked_suffix = attack_fn(w_text)
            # [SỬA LỖI 2]: Bỏ " " khi nối chuỗi attack
            full_attacked = prompt + attacked_suffix
            attack_results[name].append(detector.detect(full_attacked, tokenizer, p_len, temp=0.7).z_score)

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
        "WM": "Unbiased (UB)",
        "Delta_PPL": f"{mean_w_ppl - mean_uw_ppl:+.2f}",
        "Base_PPL": f"{mean_uw_ppl:.2f}",
        "Rep3_Score_%": f"{mean_w_rep*100:.1f}",
        "Mean_Z_Score": f"{mean_w_z:.2f}",
        "UW_Z_Score": f"{mean_uw_z:.2f}",
        "ROC_AUC": f"{auc_score:.4f}",
        "Avg_Tokens_to_Z4": f"{mean_eff:.1f}" if mean_eff != float('inf') else "N/A",
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
    print(f"Appended UB results to {csv_path}")

if __name__ == "__main__":
    main()