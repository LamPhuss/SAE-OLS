import math
from dataclasses import dataclass
from typing import List, Optional
import torch
from config import Config
from sae import load_sae
from prf import get_static_prompt_seed
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_adapter import SemanticAdapter

@dataclass
class DetectionResult:
    is_watermarked: bool
    z_score: float
    p_value: float
    total_score: float
    num_tokens: int
    mean_score: float
    per_token_scores: Optional[List[float]] = None
    used_clusters: Optional[List[int]] = None
    token_details: Optional[List[dict]] = None # <--- Đã có

class WatermarkDetector:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name_or_path, 
            device_map=self.device, 
            quantization_config=bnb_config,
        )
        self.model.eval()
        self.sae = load_sae(config.sae, device=self.device)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.d_model = self.model.config.hidden_size
        
        self.anchor = SemanticAdapter(n_clusters=4096).to(self.device)
        adapter_path = os.path.join(os.path.dirname(__file__), "..", "models", "semantic_adapter.pth")
        
        if os.path.exists(adapter_path):
            self.anchor.load_state_dict(torch.load(adapter_path, map_location=self.device))
        else:
            print(f"CẢNH BÁO NGUY HIỂM: Không tìm thấy {adapter_path}. Adapter đang chạy bằng Random Weights!")
            
        self.anchor.init_inference_tools(self.device)
        self.anchor.eval()
        
    @torch.no_grad()
    # THÊM THAM SỐ return_token_details Ở ĐÂY:
    def detect(self, text: str, secret_key: str, prompt_len: int = 0, return_per_token: bool = False, return_token_details: bool = False) -> DetectionResult:
        wm_cfg = self.config.watermark
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)
        token_ids = inputs["input_ids"][0].tolist()
        N = len(token_ids)

        if N <= prompt_len + 1: 
            return DetectionResult(False, 0.0, 1.0, 0.0, N, 0.0)

        if not hasattr(self, 'W_U_cache'):
            if hasattr(self.model, 'get_output_embeddings') and self.model.get_output_embeddings() is not None:
                self.W_U_cache = self.model.get_output_embeddings().weight.detach().to(torch.bfloat16).to(self.device)
            elif hasattr(self.model, 'lm_head'):
                self.W_U_cache = self.model.lm_head.weight.detach().to(torch.bfloat16).to(self.device)
            else:
                self.W_U_cache = self.model.embed_out.weight.detach().to(torch.bfloat16).to(self.device)
        
        W_U = self.W_U_cache

        prompt_ids = token_ids[:prompt_len]
        seed = get_static_prompt_seed(prompt_ids, secret_key)
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        target_idx = torch.randint(0, self.sae.d_sae, (1,), generator=rng, device=self.device).item()
        v_target = self.sae.get_feature_vector(target_idx).float()

        with torch.no_grad():
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits[0]

        scores, z_scores_t = [], []
        used_clusters = []
        token_details_list = [] # LOG LIST
        valid_tokens_count = 0
        sum_score = sum_mu = sum_var = 0.0

        start_idx = max(prompt_len, wm_cfg.context_window)

        for t in range(start_idx, N - 1):
            original_logits_t = logits[t-1]  
            
            probs = torch.softmax(original_logits_t / self.config.model.temperature, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            next_token_id = token_ids[t]
            
            # --- ĐOẠN LOG MỚI ĐẦY ĐỦ ---
            token_str = self.tokenizer.decode([next_token_id]).replace('\n', '\\n')
            entropy_val = entropy.item()

            if entropy_val < 1.0: 
                if return_token_details:
                    token_details_list.append({"token": token_str, "entropy": entropy_val, "valid": False, "wm_score": 0.0})
                continue
            # --------------------

            valid_tokens_count += 1
            W_gen = W_U[next_token_id].float()

            window_sizes = [30, 40, 50, 60, 70]
            ctx_texts = []
            for w in window_sizes:
                ctx_start = max(0, t - w)
                ctx_ids = token_ids[ctx_start:t]
                ctx_texts.append(self.tokenizer.decode(ctx_ids, skip_special_tokens=True))
                
            c_ids, S_stables_raw = self.anchor.get_geometric_anchor_batch(ctx_texts)
            
            unique_S = {}
            unique_c_ids = []
            for i in range(len(window_sizes)):
                key_tuple = tuple(S_stables_raw[i].tolist())
                if key_tuple not in unique_S:
                    unique_S[key_tuple] = S_stables_raw[i]
                    unique_c_ids.append(c_ids[i].item())
            
            S_cands = torch.stack(list(unique_S.values())).to(self.device)
            S_norms = S_cands / (S_cands.norm(dim=1, keepdim=True) + 1e-8)
            
            v_t_exp = v_target.squeeze().unsqueeze(0)
            dots = torch.sum(v_t_exp * S_norms, dim=1, keepdim=True)
            delta_hs = v_t_exp - dots * S_norms
            delta_hs = delta_hs / (delta_hs.norm(dim=1, keepdim=True) + 1e-8)
            
            scores_tensor = torch.matmul(delta_hs, W_gen.squeeze())
            S_alls = torch.matmul(W_U, delta_hs.to(torch.bfloat16).T).float()
            mu_exacts = torch.sum(probs.unsqueeze(1) * S_alls, dim=0)
            
            excesses = scores_tensor - mu_exacts
            best_idx = torch.argmax(excesses).item()
            
            best_score = scores_tensor[best_idx].item()
            best_mu = mu_exacts[best_idx].item()
            best_var = S_alls[:, best_idx].std().item() ** 2

            used_clusters.append(unique_c_ids[best_idx])
            
            # --- THÊM LOG CHO TỪ HỢP LỆ ---
            if return_token_details:
                token_details_list.append({"token": token_str, "entropy": entropy_val, "valid": True, "wm_score": best_score - best_mu})
            # ------------------------------

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
            used_clusters=used_clusters,
            token_details=token_details_list if return_token_details else None # TRẢ VỀ Ở ĐÂY
        )