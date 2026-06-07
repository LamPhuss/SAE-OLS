import torch
from config import Config
from generator import WatermarkedGenerator
from prf import get_static_prompt_seed

def debug_synonyms():
    config = Config()
    print("Loading Generator...")
    gen = WatermarkedGenerator(config)
    
    # Một prompt dừng lại ngay trước một tính từ để LLM dự đoán các từ đồng nghĩa
    prompt = "The performance of the new algorithm was absolutely"
    secret_key = "eval_secret_key_2026"
    
    inputs = gen.tokenizer(prompt, return_tensors="pt").to(gen.device)
    prompt_ids = inputs["input_ids"][0].tolist()
    
    # 1. Tính toán v_target (như trong generator)
    seed = get_static_prompt_seed(prompt_ids, secret_key)
    rng = torch.Generator(device=gen.device)
    rng.manual_seed(seed)
    target_idx = torch.randint(0, gen.sae.d_sae, (1,), generator=rng, device=gen.device).item()
    v_target = gen.sae.get_feature_vector(target_idx).float()
    
    # 2. Chạy LLM để lấy Logits tự nhiên
    with torch.no_grad():
        outputs = gen.model(inputs.input_ids)
        original_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(original_logits, dim=-1)
        
    # 3. Tính toán hình học Thủy vân (delta_h)
    ctx_text = gen.tokenizer.decode(prompt_ids, skip_special_tokens=True)
    _, S_stable = gen.anchor.get_geometric_anchor(ctx_text)
    S_norm = S_stable / (S_stable.norm() + 1e-8)
    delta_h = v_target - torch.dot(v_target.squeeze(), S_norm.squeeze()) * S_norm
    delta_h = delta_h / (delta_h.norm() + 1e-8)
    
    # 4. Chấm điểm Thủy vân cho TOÀN BỘ từ vựng
    logit_delta_raw = (gen._W_U @ delta_h.to(gen._W_U.dtype)).float()
    mean_delta = logit_delta_raw.mean()
    std_delta = logit_delta_raw.std()
    logit_delta = (logit_delta_raw - mean_delta) / (std_delta + 1e-8)
    
    # ==========================================
    # 5. IN BẢNG BÁO CÁO TOP 20 TỪ TỰ NHIÊN NHẤT
    # ==========================================
    top_k = 20
    top_probs, top_indices = torch.topk(probs, top_k)
    
    print(f"\n[DEBUG] Prompt: '{prompt}'")
    print("-" * 65)
    print(f"{'Token (Từ vựng)':<25} | {'Natural Prob (%)':<18} | {'WM Score (Z-Delta)':<15}")
    print("-" * 65)
    
    for prob, idx in zip(top_probs, top_indices):
        token_str = gen.tokenizer.decode([idx.item()]).replace("\n", "\\n")
        wm_score = logit_delta[idx].item()
        
        # Đánh dấu các từ được Thủy vân hậu thuẫn mạnh (> 1.0)
        marker = " <--- THỦY VÂN CHỌN" if wm_score > 1.0 else ""
        print(f"'{token_str}'{marker:<23} | {prob.item()*100:16.2f}% | {wm_score:15.4f}")
    print("-" * 65)

if __name__ == "__main__":
    debug_synonyms()