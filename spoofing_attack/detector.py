import math
from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from sae import load_sae
from prf import get_static_prompt_seed
from semantic_anchor import SemanticAnchor
import re

@dataclass
class DetectionResult:
    is_watermarked: bool
    z_score: float
    p_value: float
    total_score: float
    num_tokens: int
    mean_score: float
    per_token_scores: Optional[List[float]] = None

class WatermarkDetector:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name_or_path, device_map=self.device, dtype=getattr(torch, config.model.torch_dtype),
        )
        self.model.eval()
        self.sae = load_sae(config.sae, device=self.device)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.d_model = self.model.config.hidden_size
        self.anchor = SemanticAnchor(self.d_model, self.device)
        
    @torch.no_grad()
    def detect(self, text: str, secret_key: str, prompt_len: int = 0, return_per_token: bool = False) -> DetectionResult:
        wm_cfg = self.config.watermark
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)
        token_ids = inputs["input_ids"][0].tolist()
        N = len(token_ids)

        if N <= prompt_len + 1: 
            return DetectionResult(False, 0.0, 1.0, 0.0, N, 0.0)

        if hasattr(self.model, 'lm_head'): W_U = self.model.lm_head.weight.detach().float()
        elif hasattr(self.model, 'embed_out'): W_U = self.model.embed_out.weight.detach().float()

        # 1. Khôi phục v_target tĩnh từ Prompt
        prompt_ids = token_ids[:prompt_len]
        seed = get_static_prompt_seed(prompt_ids, secret_key)
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        target_idx = torch.randint(0, self.sae.d_sae, (1,), generator=rng, device=self.device).item()
        v_target = self.sae.get_feature_vector(target_idx).float()

        # Chạy LLM lấy logits
        with torch.no_grad():
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits[0]

        scores, z_scores_t = [], []
        valid_tokens_count = 0
        sum_score = sum_mu = sum_var = 0.0

        # 2. Khởi tạo Dòng chảy đồng bộ
        S_state = W_U[prompt_ids].mean(dim=0).float()
        lam = 0.8

        start_idx = max(prompt_len, wm_cfg.context_window)

        for t in range(start_idx, N - 1):
            original_logits_t = logits[t-1]  
            
            # ==========================================================
            # 1. BẮT ĐÚNG NHỊP ĐỘ SINH CỦA LLM (TEMPERATURE = 0.7)
            # ==========================================================
            # Phân phối xác suất phải được tính với đúng Temperature lúc sinh
            # để đảm bảo mu_exact (Kỳ vọng) khớp 100% với thực tế.
            probs = torch.softmax(original_logits_t / 0.7, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            
            next_token_id = token_ids[t]

            # Cổng gác Entropy: Bỏ qua các token mà Máy phát cũng đã bỏ qua
            if entropy.item() < 2.0: 
                continue

            valid_tokens_count += 1
            W_gen = W_U[next_token_id].float()

            # ==========================================================
            # 2. MÁY DÒ LSH ĐA KÍNH LÚP (RADAR + MULTI-PROBE SIÊU TỐC)
            # ==========================================================
            window_sizes = [30, 40, 50, 60, 70]
            ctx_texts = []
            for w in window_sizes:
                ctx_start = max(0, t - w)
                ctx_ids = token_ids[ctx_start:t]
                ctx_texts.append(self.tokenizer.decode(ctx_ids, skip_special_tokens=True))
                
            # BƯỚC A: Gọi SBERT 1 lần duy nhất cho cả 5 cửa sổ
            base_h_bins = self.anchor.get_lsh_components_batch(ctx_texts, secret_key)
            
            unique_hashes = {}
            for j in range(len(window_sizes)):
                base_h_bin = base_h_bins[j]
                unique_hashes[tuple(base_h_bin.tolist())] = base_h_bin
                for i in range(self.anchor.n_bits):
                    flipped_h = base_h_bin.clone()
                    flipped_h[i] *= -1.0
                    unique_hashes[tuple(flipped_h.tolist())] = flipped_h
            
            # BƯỚC B: CHẤM ĐIỂM VECTOR HÀNG LOẠT TRÊN GPU (TENSORIZATION)
            # 1. Gom tất cả ứng viên thành 1 Ma trận (M x 12)
            h_cands = torch.stack(list(unique_hashes.values())) 
            
            # 2. Tính S_stable cho tất cả M ứng viên trong 1 phép tính
            _, R_map = self.anchor._init_lsh_matrices(secret_key)
            S_stables = torch.matmul(h_cands, R_map) 
            
            # 3. Chuẩn hóa hàng loạt
            S_norms = S_stables / (S_stables.norm(dim=1, keepdim=True) + 1e-8)
            
            # 4. Tính delta_h hàng loạt
            v_t_exp = v_target.squeeze().unsqueeze(0) # (1, d_model)
            dots = torch.sum(v_t_exp * S_norms, dim=1, keepdim=True) # (M, 1)
            delta_hs = v_t_exp - dots * S_norms # (M, d_model)
            delta_hs = delta_hs / (delta_hs.norm(dim=1, keepdim=True) + 1e-8)
            
            # 5. Chấm điểm thực tế hàng loạt
            scores = torch.matmul(delta_hs, W_gen.squeeze()) # (M,)
            
            # 6. Tính kỳ vọng hàng loạt
            S_alls = torch.matmul(W_U, delta_hs.T) # (vocab, M)
            mu_exacts = torch.sum(probs.unsqueeze(1) * S_alls, dim=0) # (M,)
            
            # 7. Tìm Excess Score cao nhất bằng CUDA
            excesses = scores - mu_exacts
            best_idx = torch.argmax(excesses)
            
            best_score = scores[best_idx].item()
            best_mu = mu_exacts[best_idx].item()
            best_var = S_alls[:, best_idx].std().item() ** 2

            # ==========================================================
            # 3. TOÁN HỌC CHỐT HẠ 
            # ==========================================================
            sum_score += best_score
            sum_mu += best_mu
            sum_var += best_var
            
            if return_per_token:
                sigma_t = math.sqrt(max(best_var, 1e-10))
                z_scores_t.append((best_score - best_mu) / sigma_t if sigma_t > 1e-8 else 0.0)
        if valid_tokens_count < 2:
            return DetectionResult(False, 0.0, 1.0, 0.0, N, 0.0)

        final_z_score = (sum_score - sum_mu) / math.sqrt(max(sum_var, 1e-10))

        return DetectionResult(
            is_watermarked=final_z_score > wm_cfg.z_threshold, 
            z_score=final_z_score,
            p_value=0.5 * math.erfc(final_z_score / math.sqrt(2)) if final_z_score > 0 else 1.0,
            total_score=sum_score, 
            num_tokens=N, 
            mean_score=sum_score/valid_tokens_count if valid_tokens_count else 0,
            per_token_scores=z_scores_t if return_per_token else None,
        )