"""
Fixed Evaluation Script for Baseline: DiPmark
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. THUẬT TOÁN DIPMARK
# ==========================================

class Dipmark_WatermarkCode:
    def __init__(self, shuffle: torch.LongTensor):
        self.shuffle = shuffle
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(cls, rng: list, vocab_size: int, device: torch.device):
        batch_size = len(rng)
        shuffle = torch.stack([
            torch.randperm(vocab_size, generator=rng[i], device='cpu')
            for i in range(batch_size)
        ]).to(device)
        return cls(shuffle)

class Dip_Reweight:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def reweight_logits(self, code: Dipmark_WatermarkCode, p_logits: torch.FloatTensor) -> torch.FloatTensor:
        s_p_logits = torch.gather(p_logits, -1, code.shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        boundary_1 = torch.argmax((s_cumsum > self.alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.clamp(torch.gather(s_p, -1, boundary_1), min=1e-10)
        portion_in_right_1 = torch.clamp((torch.gather(s_cumsum, -1, boundary_1) - self.alpha) / p_boundary_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > self.alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        boundary_2 = torch.argmax((s_cumsum > (1-self.alpha)).to(torch.int), dim=-1, keepdim=True)
        p_boundary_2 = torch.clamp(torch.gather(s_p, -1, boundary_2), min=1e-10)
        portion_in_right_2 = torch.clamp((torch.gather(s_cumsum, -1, boundary_2) - (1-self.alpha)) / p_boundary_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1-self.alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = torch.clamp(s_all_portion_in_right_2/2 + s_all_portion_in_right_1/2, min=1e-12)
        s_shift_logits = torch.log(s_all_portion_in_right)
        shift_logits = torch.gather(s_shift_logits, -1, code.unshuffle)
        return p_logits + shift_logits

class DiPmarkLogitsProcessor(LogitsProcessor):
    # SỬA LỖI: Chuyển context_width mặc định lên 5 theo code gốc của tác giả
    def __init__(self, private_key: str, alpha: float=0.45, context_width: int=5, temperature: float=0.7, top_p: float=0.9):
        self.private_key = private_key.encode('utf-8')
        self.reweight = Dip_Reweight(alpha)
        self.context_width = context_width
        self.temperature = temperature
        self.top_p = top_p
        
    def get_rng_seed(self, context_code: bytes) -> int:
        m = hashlib.sha256()
        m.update(str(context_code).encode('utf-8'))
        m.update(self.private_key)
        return int.from_bytes(m.digest(), "big") % (2**32 - 1)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.temperature != 1.0:
            scores = scores / self.temperature

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores[indices_to_remove] = -1e9 

        batch_size = input_ids.size(0)
        context_codes = [
            input_ids[i][-self.context_width:].tolist() 
            for i in range(batch_size)
        ]
        seeds = [self.get_rng_seed(cc) for cc in context_codes]
        rng = [torch.Generator(device='cpu').manual_seed(seed) for seed in seeds]
        
        watermark_code = Dipmark_WatermarkCode.from_random(rng, scores.size(1), scores.device)
        reweighted_scores = self.reweight.reweight_logits(watermark_code, scores.float())
        return reweighted_scores

class DiPmarkDetector:
    # SỬA LỖI: context_width đồng bộ bằng 5
    def __init__(self, vocab_size: int, private_key: str, context_width: int=5, gamma: float=0.5):
        self.vocab_size = vocab_size
        self.private_key = private_key.encode('utf-8')
        self.context_width = context_width
        self.gamma = gamma

    def get_rng_seed(self, context_list: list) -> int:
        m = hashlib.sha256()
        m.update(str(context_list).encode('utf-8')) 
        m.update(self.private_key)
        return int.from_bytes(m.digest(), "big") % (2**32 - 1)

    class Result:
        def __init__(self, z_score):
            self.z_score = z_score

    # SỬA LỖI MẠNH: Nhận trực tiếp mảng array chứa token ID thay vì string
    def detect_from_tokens(self, prompt_tokens: list, gen_tokens: list):
        token_ids = prompt_tokens + gen_tokens
        prompt_len = len(prompt_tokens)
        N = len(token_ids)

        if N <= prompt_len: 
            return self.Result(0.0)

        green_tokens_count = 0
        valid_tokens = N - prompt_len

        for i in range(prompt_len, N):
            context = token_ids[max(0, i - self.context_width) : i]
            current_token = token_ids[i]

            seed = self.get_rng_seed(context)
            rng = torch.Generator(device='cpu').manual_seed(seed)
            
            shuffle = torch.randperm(self.vocab_size, generator=rng)
            
            green_start = int(self.gamma * self.vocab_size)
            unshuffle_idx = (shuffle == current_token).nonzero(as_tuple=True)[0].item()
            
            if unshuffle_idx >= green_start:
                green_tokens_count += 1

        expected_green = (1 - self.gamma) * valid_tokens
        variance = self.gamma * (1 - self.gamma) * valid_tokens
        z_score = (green_tokens_count - expected_green) / math.sqrt(variance) if variance > 0 else 0.0
        
        return self.Result(z_score)

    def detect_from_text(self, prompt_text: str, gen_text: str, tokenizer):
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
        return self.detect_from_tokens(prompt_tokens, gen_tokens)

# ==========================================
# 2. CÁC HÀM ĐÁNH GIÁ & TẤN CÔNG (Giữ Nguyên)
# ==========================================

def calculate_perplexity(model, tokenizer, prompt_text, generated_text, device):
    full_text = prompt_text + generated_text
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
def attack_paraphrase(text: str, model, tokenizer, device) -> str:
    # Prompt yêu cầu mô hình viết lại đoạn văn bản
    messages = [
        {"role": "system", "content": "You are a precise paraphrasing tool. Rewrite the user's text to change its vocabulary and sentence structure while preserving the exact original meaning. Output ONLY the paraphrased text."},
        {"role": "user", "content": text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Giới hạn độ dài sinh ra dựa trên văn bản gốc để tránh tràn RAM
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
    secret_key = "dipmark_secure_key_2026"
    Z_THRESHOLD = 1.0  # Ngưỡng Z-score chung cho tất cả các loại evaluate

    print("Loading models for DiPmark Baseline...")
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
    dip_processor = DiPmarkLogitsProcessor(private_key=secret_key, alpha=0.45, context_width=5)
    detector = DiPmarkDetector(vocab_size=vocab_size, private_key=secret_key, context_width=5, gamma=0.5)
    
    # KHUYẾN CÁO: Test_limit > 50 để thấy Z-score Null về 0
    raw_prompts = []
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'alpaca_eval_10k.jsonl')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'large_dataset.jsonl')
        
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): 
                    raw_prompts.append(json.loads(line).get("prompt", ""))
    else:
        raw_prompts = ["The future of artificial intelligence is"]
        
    # 2. CHÈN SYSTEM PROMPT ĐỂ ÉP ĐỘ DÀI (Tuyệt chiêu buff Z-score cho DiPmark)
    system_inst = "You are a highly detailed academic assistant. You MUST provide exhaustive, multi-paragraph answers (minimum 250 words)."
    prompts = [
        tokenizer.apply_chat_template([
            {"role": "system", "content": system_inst}, 
            {"role": "user", "content": p}
        ], tokenize=False, add_generation_prompt=True)
        for p in raw_prompts
    ]

    # 3. GIỚI HẠN SỐ LƯỢNG TEST
    TEST_LIMIT = int(os.environ.get('EVAL_TEST_LIMIT', 1000))
    prompts = prompts[:TEST_LIMIT]

    unwatermarked_texts, watermarked_texts = [], []
    prompt_tokens_list = []
    uw_gen_tokens_list, w_gen_tokens_list = [], []

    print(f"\n--- STEP 1: GENERATION ({len(prompts)} prompts) ---")
    for prompt in tqdm(prompts, desc="Generating (DiPmark)"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            uw_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
            w_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, logits_processor=LogitsProcessorList([dip_processor]))
        
        p_len = inputs.input_ids.shape[1]
        
        # SỬA LỖI: Lưu lại trực tiếp mảng token_id để tránh sai lệch do tokenizer
        prompt_tokens_list.append(inputs.input_ids[0].tolist())
        uw_gen_tokens_list.append(uw_out[0][p_len:].tolist())
        w_gen_tokens_list.append(w_out[0][p_len:].tolist())

        unwatermarked_texts.append(tokenizer.decode(uw_out[0][p_len:], skip_special_tokens=True))
        watermarked_texts.append(tokenizer.decode(w_out[0][p_len:], skip_special_tokens=True))
        del inputs, uw_out, w_out
        torch.cuda.empty_cache()

    # Lưu kết quả generate vào CSV
    csv_path = "eval_data_dip.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "prompt", "unwatermarked_text", "watermarked_text"])
        for i, (p, uw, w) in enumerate(zip(prompts, unwatermarked_texts, watermarked_texts)):
            writer.writerow([i, p, uw, w])
    print(f"\n[Saved generation results to {csv_path}]")

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
    for pt, uw_toks, w_toks in tqdm(zip(prompt_tokens_list, uw_gen_tokens_list, w_gen_tokens_list), total=len(prompts), desc="Detectability"):
        # Detect thẳng từ arrays
        uw_z_scores.append(detector.detect_from_tokens(pt, uw_toks).z_score)
        w_z_scores.append(detector.detect_from_tokens(pt, w_toks).z_score)

    mean_uw_z, mean_w_z = np.mean(uw_z_scores), np.mean(w_z_scores)
    
    auc_score = 1.0
    if len(uw_z_scores) > 1 and len(w_z_scores) > 1:
        auc_score = roc_auc_score([0]*len(uw_z_scores) + [1]*len(w_z_scores), uw_z_scores + w_z_scores)
    
    # TPR / FPR tại ngưỡng Z_THRESHOLD
    tpr = sum(1 for z in w_z_scores if z >= Z_THRESHOLD) / len(w_z_scores) * 100 if w_z_scores else 0.0
    fpr = sum(1 for z in uw_z_scores if z >= Z_THRESHOLD) / len(uw_z_scores) * 100 if uw_z_scores else 0.0

    print(f"\n[DIPMARK DETECTABILITY METRICS] (Z_THRESHOLD = {Z_THRESHOLD})")
    print(f"  - UW Z-Score (Null)   : {mean_uw_z:.2f} (Std: {np.std(uw_z_scores):.2f})")
    print(f"  - WM Z-Score          : {mean_w_z:.2f}")
    print(f"  - ROC-AUC Score       : {auc_score:.4f}")
    print(f"  - TPR (WM >= {Z_THRESHOLD})     : {tpr:.1f}%")
    print(f"  - FPR (UW >= {Z_THRESHOLD})     : {fpr:.1f}%")

    print("\n--- STEP 4: ROBUSTNESS (ATTACKS) ---")
    attacks = {
        "No Attack": lambda t: t,
        "Deletion (10%)": lambda t: attack_word_deletion(t, 0.1),
        "Deletion (30%)": lambda t: attack_word_deletion(t, 0.3),
        "Swap (10%)": lambda t: attack_word_swap(t, 0.1),
        "Paraphrase (LLM)": lambda t: attack_paraphrase(t, model, tokenizer, device)
    }

    attack_results = {name: [] for name in attacks}

    # Quá trình chạy (Có in debug paraphrase nếu muốn)
    for prompt_text, w_text in tqdm(zip(prompts, watermarked_texts), desc="Robustness Evals", total=len(prompts)):
        for name, attack_fn in attacks.items():
            attacked_text = attack_fn(w_text)
            
            # Detect lại Z-score sau khi tấn công
            res = detector.detect_from_text(prompt_text, attacked_text, tokenizer)
            attack_results[name].append(res.z_score)
            
            # In debug nhanh (giống log gốc của bạn) nếu chỉ có 1 mẫu
            if len(prompts) == 1 and name == "Paraphrase (LLM)":
                print(f"\n[DEBUG PARAPHRASE]")
                print(f"- Z-score bản gốc: {attack_results['No Attack'][0]:.2f}")
                print(f"- Z-score sau Paraphrase: {res.z_score:.2f}")

    print("\n[4] ROBUSTNESS METRICS (Under Attack)")
    baseline_z = np.mean(attack_results["No Attack"])
    
    for name, scores in attack_results.items():
        mean_z = np.mean(scores)
        survived_pct = sum(1 for z in scores if z >= Z_THRESHOLD) / len(scores) * 100
        z_drop = (1 - mean_z / baseline_z) * 100 if baseline_z > 0 else 0.0
        print(f"  {name:<25s} | Z= {mean_z:5.2f} | Drop: {z_drop:5.1f}% | Survived(Z>={Z_THRESHOLD}): {survived_pct:5.1f}%")
    # ... (sau đoạn print kết quả in ra màn hình)
    print("\n--- STEP 5: LƯU KẾT QUẢ VÀO CSV ---")
    
    get_attack_z = lambda name: f"{np.mean(attack_results[name]):.2f}" if name in attack_results else "N/A"

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval.csv')
    file_exists = os.path.exists(csv_path)
    
    # THÊM CỘT Paraphrase và Deletion 30%
    fieldnames = [
        "WM", "Delta_PPL", "Base_PPL", "Rep3_Score_%", "Mean_Z_Score", "UW_Z_Score", 
        "ROC_AUC", "No_Attack_Z", "Del_10_Z", "Del_30_Z", "Swap_10_Z", "Para_Z"
    ]
    
    row = {
        "WM": "DiPmark",
        "Delta_PPL": f"{mean_w_ppl - mean_uw_ppl:+.2f}",
        "Base_PPL": f"{mean_uw_ppl:.2f}",
        "Rep3_Score_%": f"{mean_w_rep*100:.1f}",
        "Mean_Z_Score": f"{mean_w_z:.2f}",
        "UW_Z_Score": f"{mean_uw_z:.2f}",
        "ROC_AUC": f"{auc_score:.4f}",
        "No_Attack_Z": get_attack_z("No Attack"),
        "Del_10_Z": get_attack_z("Deletion (10%)"),
        "Del_30_Z": get_attack_z("Deletion (30%)"),      # <--- Mới thêm
        "Swap_10_Z": get_attack_z("Swap (10%)"),
        "Para_Z": get_attack_z("Paraphrase (LLM)")       # <--- Mới thêm
    }

    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Đã lưu kết quả thành công vào: {os.path.abspath(csv_path)}")
    except Exception as e:
        print(f"Lỗi khi lưu file CSV: {e}")
if __name__ == "__main__":
    main()