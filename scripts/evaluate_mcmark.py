"""
Advanced Evaluation Script for Baseline: MCMark (Multi-Channel-based Unbiased Watermark).
Based on Chen et al., 2025. Uses extremely fast Likelihood-Agnostic Detection.
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
from sklearn.metrics import roc_auc_score, f1_score

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ==========================================
# 1. THUẬT TOÁN MCMARK
# ==========================================

class MCMark_WatermarkCode:
    def __init__(self, shuffle: torch.LongTensor, split_k: torch.BoolTensor):
        self.shuffle = shuffle
        self.split_k = split_k
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(cls, rng_list: list, vocab_size: int, split_num: int, device: torch.device):
        batch_size = len(rng_list)
        # Đảm bảo thứ tự: 1. shuffle, 2. split_k
        shuffles = []
        split_ks = []
        for gen in rng_list:
            shuffles.append(torch.randperm(vocab_size, generator=gen)) # Gọi lần 1
            split_ks.append(torch.randint(low=0, high=split_num, size=(1,), generator=gen)) # Gọi lần 2
    
        shuffle = torch.stack(shuffles).to(device)
        split_k = torch.cat(split_ks, dim=0).to(device)
        return cls(shuffle, split_k)

class MC_Reweight:
    def __init__(self, n: int):
        self.n = n # Number of channels

    def reweight_logits(self, code: MCMark_WatermarkCode, p_logits: torch.FloatTensor) -> torch.FloatTensor:
        def set_nan_to_zero(x):
            x[torch.isnan(x)] = 0
            return x

        s_logits = torch.gather(p_logits, -1, code.shuffle)
        s_probs = F.softmax(s_logits, dim=-1)
        bsz, vocab_size = s_logits.shape

        splits = []
        if vocab_size % self.n == 0:
            splits = torch.arange(start=0, end=vocab_size).reshape(self.n, vocab_size // self.n).to(p_logits.device)
            split_sums = s_probs.view(bsz, self.n, vocab_size // self.n).sum(dim=-1)
        else:
            for n_idx in range(self.n):
                splits.append(list(range(round(vocab_size * n_idx / self.n), round(vocab_size * (n_idx + 1) / self.n))))
            split_sums = []
            for n_idx in range(self.n):
                cur_split = splits[n_idx]
                split_sums.append(s_probs[:, cur_split].sum(dim=-1, keepdim=True))
            split_sums = torch.cat(split_sums, dim=-1)

        split_k = code.split_k
        scales = torch.minimum(self.n * torch.ones_like(split_sums).to(s_probs.device), 1 / split_sums)

        overflow_scales = (self.n * split_sums - 1) / split_sums
        overflow_scales = set_nan_to_zero(overflow_scales)
        overflow_scales[overflow_scales < 0] = 0

        target_scales = scales[range(bsz), split_k.squeeze(-1) if split_k.dim()>1 else split_k]
        target_sums = split_sums[range(bsz), split_k.squeeze(-1) if split_k.dim()>1 else split_k]

        remain_sums = 1 - target_scales * target_sums
        overflow_sums = (overflow_scales * split_sums).sum(dim=-1)
        fill_scale = remain_sums / overflow_sums
        fill_scale = set_nan_to_zero(fill_scale)

        split_mask = torch.arange(0, self.n).to(s_logits.device).view(1, -1).repeat(bsz, 1) == split_k.view(-1, 1).repeat(1, self.n)
        final_scale = torch.where(
            split_mask,
            target_scales.view(-1, 1).repeat(1, self.n),
            fill_scale.view(-1, 1) * overflow_scales,
        )

        reweighted_s_probs = torch.zeros_like(s_probs).to(s_logits.device)
        if vocab_size % self.n == 0:
            reweighted_s_probs = (final_scale.view(bsz, self.n, 1).expand((-1, -1, vocab_size // self.n)).reshape(bsz, vocab_size) * s_probs)
        else:
            for n_idx in range(self.n):
                cur_split = splits[n_idx]
                reweighted_s_probs[:, cur_split] = (final_scale[:, n_idx].view(-1, 1) * s_probs[:, cur_split])

        reweighted_s_probs[reweighted_s_probs < 0] = 0
        reweighted_s_probs = torch.clamp(reweighted_s_probs, min=1e-12)
        reweighted_s_logits = torch.log(reweighted_s_probs)
        reweighted_logits = torch.gather(reweighted_s_logits, -1, code.unshuffle)

        return reweighted_logits

class MCMarkLogitsProcessor(LogitsProcessor):
    def __init__(self, private_key: str, num_channels: int=256, context_width: int=5, temp: float=0.7):
        self.private_key = private_key.encode('utf-8')
        self.reweight = MC_Reweight(num_channels)
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
        watermark_code = MCMark_WatermarkCode.from_random(rng, scores.size(1), self.reweight.n, scores.device)
        
        scaled_scores = scores.float() / self.temp
        reweighted_scores = self.reweight.reweight_logits(watermark_code, scaled_scores)
        return reweighted_scores * self.temp

class MCMarkDetector:
    def __init__(self, vocab_size: int, private_key: str, context_width: int=5, num_channels: int=256):
        self.vocab_size = vocab_size
        self.private_key = private_key.encode('utf-8')
        self.context_width = context_width
        self.n = num_channels

    def get_rng_seed(self, context_list: list) -> int:
        context_str = "_".join(map(str, context_list))
        m = hashlib.sha256()
        m.update(context_str.encode('utf-8'))
        m.update(self.private_key)
        return int.from_bytes(m.digest()[:8], "big") % (2**32 - 1)

    class Result:
        def __init__(self, z_score):
            self.z_score = z_score

    def detect(self, prompt_text: str, generated_text: str, tokenizer):
        # 1. Mã hóa riêng prompt để lấy độ dài
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        # 2. Gộp chuỗi để Tokenizer mã hóa chuẩn xác như lúc generate
        # Xử lý khoảng trắng nối nếu cần thiết
        full_text = prompt_text + generated_text
        if not generated_text.startswith(" ") and not generated_text.startswith("\n"):
            full_text = prompt_text + " " + generated_text
            
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        N = len(full_tokens)
        
        # Đảm bảo có đủ token để check
        if N <= prompt_len: return self.Result(0.0)

        matches = 0
        valid_tokens = N - prompt_len

        # 3. Chỉ kiểm tra các token thuộc phần 'generated_text' (Bắt đầu từ prompt_len)
        for i in range(prompt_len, N):
            # Lấy 5 token ngay trước token hiện tại (có thể chứa token của prompt)
            context_list = full_tokens[i - self.context_width : i]
            current_token = full_tokens[i]

            seed = self.get_rng_seed(context_list)
            rng = torch.Generator(device='cpu').manual_seed(seed)
            
            # Khởi tạo lại mã (THỨ TỰ BẮT BUỘC KHỚP VỚI LÚC GEN)
            shuffle = torch.randperm(self.vocab_size, generator=rng, device='cpu')
            split_k = torch.randint(low=0, high=self.n, size=(1,), dtype=torch.long, generator=rng, device='cpu').item()

            token_idx_in_shuffled = (shuffle == current_token).nonzero(as_tuple=True)[0].item()

            # Phân kênh
            current_token_channel = -1
            if self.vocab_size % self.n == 0:
                segment_size = self.vocab_size // self.n
                current_token_channel = token_idx_in_shuffled // segment_size
            else:
                for n_idx in range(self.n):
                    s_idx = round(self.vocab_size * n_idx / self.n)
                    e_idx = round(self.vocab_size * (n_idx + 1) / self.n)
                    if s_idx <= token_idx_in_shuffled < e_idx:
                        current_token_channel = n_idx
                        break

            if current_token_channel == split_k:
                matches += 1

        p = 1.0 / self.n
        expected_matches = valid_tokens * p
        variance = valid_tokens * p * (1 - p)
        
        z_score = (matches - expected_matches) / math.sqrt(variance) if variance > 0 else 0.0
        return self.Result(z_score)

# ==========================================
# 2. CÁC HÀM ĐÁNH GIÁ & TẤN CÔNG
# ==========================================

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

def measure_token_efficiency(detector, prompt_text, generated_text, tokenizer, target_threshold, thresholds=[50, 100, 150, 200, 300]):
    # Chỉ lấy token của phần text được sinh ra
    tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    
    for length in thresholds:
        if len(tokens) < length: continue
        
        # Cắt ngắn phần text được sinh ra theo ngưỡng length
        partial_gen_text = tokenizer.decode(tokens[:length])
        
        # Truyền cả prompt_text và partial_gen_text vào hàm detect mới
        res = detector.detect(prompt_text, partial_gen_text, tokenizer)
        
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

def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 400
    secret_key = "mcmark_secure_key_2026"
    num_channels = 64 # Cấu hình tối ưu từ bài báo

    print("Loading models for MCMark Baseline...")
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
    mc_processor = MCMarkLogitsProcessor(private_key=secret_key, num_channels=num_channels, context_width=5, temp=0.7)
    detector = MCMarkDetector(vocab_size=vocab_size, private_key=secret_key, context_width=5, num_channels=num_channels)
    
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

    eval_data_path = os.path.join(os.path.dirname(__file__), '..', 'eval_data_mcmark_llama3_1000.csv')
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
        for prompt in tqdm(prompts, desc="Generating (MCMark)"):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                uw_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
                w_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, logits_processor=LogitsProcessorList([mc_processor]))
            
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
        uw_z_scores.append(detector.detect(prompt, uw_text, tokenizer).z_score)
        w_z_scores.append(detector.detect(prompt, w_text, tokenizer).z_score)

    mean_uw_z, mean_w_z = np.mean(uw_z_scores), np.mean(w_z_scores)
    
    y_true = [0] * len(uw_z_scores) + [1] * len(w_z_scores)
    y_scores = uw_z_scores + w_z_scores
    auc_score = roc_auc_score(y_true, y_scores)

    # ĐỊNH NGHĨA CÁC NGƯỠNG ĐỘNG
    eval_thresholds = [1.0, 2.0, 3.0, 4.0]
    
    print(f"\n[MCMARK DETECTABILITY METRICS]")
    print(f"  - UW Z-Score (Null)   : {mean_uw_z:.2f} (Std: {np.std(uw_z_scores):.2f})")
    print(f"  - WM Z-Score          : {mean_w_z:.2f} (Std: {np.std(w_z_scores):.2f})")
    print(f"  - ROC-AUC Score       : {auc_score:.4f}\n")

    # Đánh giá F1-Score và Token Efficiency theo từng ngưỡng
    eff_results = {}
    for t in eval_thresholds:
        # Tính F1-Score
        y_pred = [1 if z >= t else 0 for z in y_scores]
        f1 = f1_score(y_true, y_pred)
        
        # Tính Token Efficiency
        efficiency_tokens = []
        for prompt, w_text in zip(prompts, watermarked_texts):
            eff = measure_token_efficiency(detector, prompt, w_text, tokenizer, target_threshold=t)
            if eff: efficiency_tokens.append(eff)
            
        mean_eff = np.mean(efficiency_tokens) if efficiency_tokens else float('inf')
        eff_results[t] = mean_eff
        
        print(f"  [Ngưỡng Z={t:.1f}] F1-Score: {f1:.4f} | Avg Tokens to hit: {'~' + str(round(mean_eff, 1)) if mean_eff != float('inf') else 'N/A'}")

    print(f"\n[MCMARK FIDELITY OVERVIEW]")
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
        for name, attack_fn in attacks.items():
            attacked_text = attack_fn(w_text)
            z_score = detector.detect(prompt, attacked_text, tokenizer).z_score
            attack_results[name].append(z_score)

    print("\n[4] ROBUSTNESS METRICS (Under Attack)")
    baseline_z = np.mean(attack_results["No Attack"])
    
    for name, scores in attack_results.items():
        mean_z = np.mean(scores)
        z_drop = (1 - mean_z / baseline_z) * 100 if baseline_z > 0 else 0.0
        print(f"\n  > {name:<20s} | Mean Z: {mean_z:5.2f} | Mất mát Z-Score: {z_drop:5.1f}%")
        
        # In tỷ lệ sống sót cho từng ngưỡng
        surv_str = []
        for t in eval_thresholds:
            survived_pct = sum(1 for z in scores if z >= t) / len(scores) * 100
            surv_str.append(f"Z>={t}: {survived_pct:5.1f}%")
        print(f"    Sống sót: [ " + " | ".join(surv_str) + " ]")
        
    print("\n==================================================\n")

    # --- LƯU VÀO eval.csv (ĐÃ CẬP NHẬT TRƯỜNG ĐỘNG) ---
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'eval_dynamic.csv')
    file_exists = os.path.exists(csv_path)
    
    # Tạo header cho file CSV mới
    fieldnames = ["WM", "Delta_PPL", "Base_PPL", "Mean_Z", "ROC_AUC"]
    for t in eval_thresholds: fieldnames.extend([f"F1_Z{t}", f"Tokens_Z{t}"])
    for name in attacks.keys():
        atk_key = name.split()[0].replace("(10%)", "10").replace("(30%)", "30")
        fieldnames.append(f"{atk_key}_Z")
        for t in eval_thresholds: fieldnames.append(f"{atk_key}_Surv_Z{t}")

    # Chuẩn bị dữ liệu để lưu
    row = {
        "WM": "MCMark",
        "Delta_PPL": f"{mean_w_ppl - mean_uw_ppl:+.2f}",
        "Base_PPL": f"{mean_uw_ppl:.2f}",
        "Mean_Z": f"{mean_w_z:.2f}",
        "ROC_AUC": f"{auc_score:.4f}",
    }
    
    for t in eval_thresholds:
        row[f"F1_Z{t}"] = f"{f1_score(y_true, [1 if z >= t else 0 for z in y_scores]):.4f}"
        row[f"Tokens_Z{t}"] = f"{eff_results[t]:.1f}" if eff_results[t] != float('inf') else "N/A"
        
    for name, scores in attack_results.items():
        atk_key = name.split()[0].replace("(10%)", "10").replace("(30%)", "30")
        row[f"{atk_key}_Z"] = f"{np.mean(scores):.2f}"
        for t in eval_thresholds:
            row[f"{atk_key}_Surv_Z{t}"] = f"{sum(1 for z in scores if z >= t) / len(scores) * 100:.1f}"

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        writer.writerow(row)
    print(f"Đã lưu kết quả động vào {csv_path}")

if __name__ == "__main__":
    main()