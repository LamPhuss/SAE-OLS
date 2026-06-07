import torch
import sys
import os
from config import Config
from generator import WatermarkedGenerator
from detector import WatermarkDetector

class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.file = open(file_path, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def isatty(self):
        return self.terminal.isatty()
    def close(self):
        self.file.close()
        sys.stdout = self.terminal

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
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=safe_max_tokens, temperature=0.7, top_p=0.9, 
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] 
        )
    rewritten = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    if "\n\n" in rewritten:
        parts = rewritten.split("\n\n", 1)
        if len(parts[0]) < 100 and any(kw in parts[0].lower() for kw in ["here", "sure", "rewrite", "version", ":"]):
            rewritten = parts[1].strip()
            
    return rewritten

def print_token_table(title, res, is_generator=False):
    print(f"\n{'='*95}")
    if is_generator:
        print(f" {title} ")
    else:
        print(f" {title} | Z-SCORE TỔNG: {res.z_score:.2f} ")
    print(f"{'='*95}")
    
    if is_generator:
        print(f"{'Token (Từ vựng)':<18} | {'Entropy':<8} | {'Prob (%)':<8} | {'Valid?':<6} | {'Lý do loại':<12} | {'Logit Delta (Bơm)'}")
    else:
        print(f"{'Token (Từ vựng)':<18} | {'Entropy':<8} | {'Prob (%)':<8} | {'Valid?':<6} | {'Lý do loại':<12} | {'Net WM Score'}")
    
    print("-" * 95)
    
    valid_count = 0
    total_net_score = 0.0
    token_list = res if is_generator else res.token_details
    
    for d in token_list:
        valid_str = "YES" if d['valid'] else "NO"
        score_str = f"{d['wm_score']:>10.4f}" if d['valid'] else "       -"
        prob_pct = d.get('prob', 0.0) * 100
        reason = d.get('reason', '')
        
        marker = ""
        if d['valid']:
            if is_generator:
                marker = " <--- BƠM MẠNH" if d['wm_score'] > 0.5 else ""
            else:
                marker = " <--- TĂNG Z" if d['wm_score'] > 0.3 else (" <--- GIẢM Z" if d['wm_score'] < -0.3 else "")
                
            valid_count += 1
            total_net_score += d['wm_score']
            
        print(f"'{d['token']:<16}' | {d['entropy']:<8.4f} | {prob_pct:>6.2f}% | {valid_str:<6} | {reason:<12} | {score_str} {marker}")
        
    print("-" * 95)
    print(f"Tổng số từ hợp lệ : {valid_count} tokens")
    if not is_generator:
        print(f"Tổng Net WM Score  : {total_net_score:.4f} (Trước khi chia cho Phương sai)")

def main():
    config = Config()
    secret_key = "eval_secret_key_2026"
    
    print("Loading Generator...")
    generator = WatermarkedGenerator(config)
    
    print("Initializing Detector...")
    detector = WatermarkDetector.__new__(WatermarkDetector)
    detector.config, detector.device, detector.model = config, config.model.device, generator.model
    detector.tokenizer, detector.sae, detector.anchor = generator.tokenizer, generator.sae, generator.anchor

    # Khởi tạo Prompt Gốc
    prompt_raw = "Explain the concept of quantum computing."
    sys_inst = "You are a highly detailed academic assistant. Provide exhaustive, multi-paragraph answers (minimum 100 words)."
    formatted_prompt = generator.tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_inst}, {"role": "user", "content": prompt_raw}], 
        tokenize=False, add_generation_prompt=True
    )

    print("\n1. Đang sinh văn bản có Thủy vân (Original)...")
    w_text, gen_details = generator.generate(formatted_prompt, secret_key, max_new_tokens=400, return_token_details=True)
    
    print("\n2. Kẻ tấn công đang Paraphrase văn bản...")
    para_text = attack_llm_paraphrase(w_text, generator.model, generator.tokenizer, generator.device)

    # --- ĐỘT PHÁ: ĐỐI CHỨNG ÂM (NEGATIVE CONTROL) ---
    print("\n3. Đang sinh văn bản GIẢ (Unrelated Text) để đối chứng...")
    fake_prompt_raw = "Write a detailed recipe for baking a classic chocolate cake."
    formatted_fake_prompt = generator.tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_inst}, {"role": "user", "content": fake_prompt_raw}], 
        tokenize=False, add_generation_prompt=True
    )
    # Sinh một đoạn văn bản giả không liên quan, độ dài tương đương
    fake_text = generator.generate_unwatermarked(formatted_fake_prompt, max_new_tokens=400)

    print("\n4. Đang chấm điểm và nội soi Token...")
    orig_res = detector.detect(formatted_prompt, w_text, secret_key, return_token_details=True)
    para_res = detector.detect(formatted_prompt, para_text, secret_key, return_token_details=True)
    fake_res = detector.detect(formatted_fake_prompt, fake_text, secret_key, return_token_details=True)

    # In kết quả (Ẩn bớt bảng in cho gọn, tập trung vào Báo cáo cuối)
    print_token_table("MÁY DÒ TRÊN VĂN BẢN GỐC (CÓ THỦY VÂN)", orig_res)

    # =================================================================
    # ĐỘT PHÁ KIỂM CHỨNG: TỶ LỆ TRÙNG KHỚP SAU PARAPHRASE VÀ ĐỐI CHỨNG ÂM
    # =================================================================
    gen_clusters = generator.last_used_clusters
    det_orig_clusters = orig_res.used_clusters
    det_para_clusters = para_res.used_clusters
    det_fake_clusters = fake_res.used_clusters

    def get_lcs_length(list1, list2):
        m, n = len(list1), len(list2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if list1[i-1] == list2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    # Tính LCS cho Paraphrase (Văn bản cùng nghĩa)
    matched_para_clusters = get_lcs_length(det_orig_clusters, det_para_clusters)
    retention_rate = (matched_para_clusters / len(det_orig_clusters)) * 100 if det_orig_clusters else 0

    # Tính LCS cho Fake Text (Văn bản ĐỐI CHỨNG ÂM - Không cùng nghĩa)
    matched_fake_clusters = get_lcs_length(det_orig_clusters, det_fake_clusters)
    fake_retention_rate = (matched_fake_clusters / len(det_orig_clusters)) * 100 if det_orig_clusters else 0

    print(f"\n{'='*85}")
    print(" BÁO CÁO KIỂM CHỨNG SỨC MẠNH KHÁNG PARAPHRASE VÀ ĐỐI CHỨNG ÂM ")
    print(f"{'='*85}")
    print(f"- Số cụm Generator đã nhúng (Gốc)               : {len(gen_clusters)} cụm")
    print(f"- Số cụm Detector tìm thấy (Gốc)                : {len(det_orig_clusters)} cụm")
    print("-" * 85)
    print(f"- Số cụm Detector tìm thấy (Sau Paraphrase)     : {len(det_para_clusters)} cụm")
    print(f"- Số cụm TRÙNG KHỚP (Gốc vs Paraphrase)         : {matched_para_clusters} cụm")
    print(f"- Tỷ lệ bám sát của Máy dò (Paraphrase)         : {retention_rate:.2f} %")
    print("-" * 85)
    print(f"- Số cụm Detector tìm thấy (Văn bản Bánh Ngọt)  : {len(det_fake_clusters)} cụm")
    print(f"- Số cụm TRÙNG KHỚP (Gốc vs Bánh Ngọt)          : {matched_fake_clusters} cụm")
    print(f"- Tỷ lệ trùng khớp ảo (Negative Control)        : {fake_retention_rate:.2f} %")
    print(f"{'='*85}\n")
    print(f"\n[SỰ THẬT VỀ SỐ LƯỢNG CỤM DUY NHẤT]")
    print(f"- Số ID cụm khác nhau dùng trong bản Gốc       : {len(set(gen_clusters))}")
    print(f"- Số ID cụm khác nhau dùng trong bản Paraphrase: {len(set(det_para_clusters))}")
    print(f"- Số ID cụm khác nhau dùng trong bản Bánh Ngọt : {len(set(det_fake_clusters))}")
    if fake_retention_rate > 50.0:
        print("=> [BÁO ĐỘNG ĐỎ] Semantic Adapter bị 'Collapse'! Nó trả về cùng 1 ID cụm cho mọi văn bản.")
    elif retention_rate > 80.0 and fake_retention_rate < 20.0:
        print("=> [THÀNH CÔNG RỰC RỠ] Adapter phân biệt ngữ nghĩa hoàn hảo! Kháng Paraphrase cực tốt.")

if __name__ == "__main__":
    main()
    try:
        main()
    finally:
        if isinstance(sys.stdout, Tee):
            sys.stdout.close()