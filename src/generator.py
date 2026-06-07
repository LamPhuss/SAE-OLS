import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from config import Config
from prf import get_static_prompt_seed
from sae import load_sae
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_adapter import SemanticAdapter

class WatermarkedGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
        
        # Nếu muốn giảm thêm 1 nửa VRAM nữa, đổi load_in_8bit thành load_in_4bit=True
        bnb_config = BitsAndBytesConfig(load_in_8bit=True) 
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name_or_path,
            device_map=self.device,
            quantization_config=bnb_config,
        )
        self.model.eval()
        
        # [VÁ LỖI OOM 1]: Ép PyTorch nhả 5.1GB VRAM rác sinh ra trong quá trình tải HuggingFace
        torch.cuda.empty_cache() 

        self._W_U = self._get_unembedding_matrix()
        
        # [VÁ LỖI OOM 2]: Tải nguyên khối SAE 2.5GB lên RAM máy tính (CPU)
        self.sae = load_sae(config.sae, device="cpu") 
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.d_model = self.model.config.hidden_size
        self.anchor = SemanticAdapter(n_clusters=4096).to(self.device)
        self.anchor.load_state_dict(torch.load("models/semantic_adapter.pth", map_location=self.device))
        self.anchor.init_inference_tools(self.device) 
        self.anchor.eval()
        print("Đã kích hoạt Chế độ Hình Học Thanh Lịch (Geometric Mode) - Tối ưu VRAM!")

    def _get_unembedding_matrix(self) -> torch.Tensor:
        if hasattr(self.model, 'lm_head'): return self.model.lm_head.weight.detach()
        elif hasattr(self.model, 'embed_out'): return self.model.embed_out.weight.detach()
        else: raise ValueError("Cannot find unembedding matrix")

    @torch.no_grad()
    def generate(self, prompt: str, secret_key: str, max_new_tokens: Optional[int] = None, return_token_details: bool = False):
        max_tokens = max_new_tokens or self.config.model.max_new_tokens
        wm_cfg = self.config.watermark
        sae_cfg = self.config.sae
        steered_count = 0
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = inputs["input_ids"][0].tolist()
        generated_ids = prompt_ids.copy()
        
        self.last_used_clusters = []
        token_details_list = [] 

        seed = get_static_prompt_seed(prompt_ids, secret_key)
        
        # Generator phải ở trên CPU vì SAE đang ở CPU
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        target_idx = torch.randint(0, self.sae.d_sae, (1,), generator=rng).item()
        
        # Chỉ kéo duy nhất 1 vector v_target lên GPU. Tiết kiệm 2.5GB!
        v_target = self.sae.get_feature_vector(target_idx).float().to(self.device)

        past_key_values = None
        cur_ids = inputs.input_ids

        for step in range(max_tokens):
            outputs = self.model(cur_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            original_logits = outputs.logits[0, -1, :]

            probs = torch.softmax(original_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            is_steered_step = False
            logit_delta = None
            
            if entropy.item() < 0.7:
                steered_logits = original_logits.float()
            else:
                is_steered_step = True
                window_size = 50
                ctx_ids = generated_ids[-window_size:] if len(generated_ids) > window_size else generated_ids
                ctx_text = self.tokenizer.decode(ctx_ids, skip_special_tokens=True)
                
                c_id, S_stable = self.anchor.get_geometric_anchor(ctx_text)
                self.last_used_clusters.append(c_id.item())

                S_norm = S_stable / (S_stable.norm() + 1e-8)
                delta_h = v_target - torch.dot(v_target.squeeze(), S_norm.squeeze()) * S_norm
                delta_h = delta_h / (delta_h.norm() + 1e-8)

                logit_delta_raw = (self._W_U @ delta_h.to(self._W_U.dtype)).float()
                
                mean_delta = logit_delta_raw.mean()
                std_delta = logit_delta_raw.std()
                logit_delta = (logit_delta_raw - mean_delta) / (std_delta + 1e-8)
                logit_delta = torch.clamp(logit_delta, min=-1.0, max=2.0)

                top_k = 10000
                _, top_indices = torch.topk(original_logits, top_k)
                mask = torch.ones_like(original_logits, dtype=torch.bool)
                mask[top_indices] = False
                logit_delta[mask] = 0.0 

                steered_logits = original_logits.float() + (wm_cfg.alpha * logit_delta)

                window_size_wrp = 40
                recent_ids = list(set(generated_ids[-window_size_wrp:]))
                for tk in recent_ids:
                    if logit_delta[tk].item() > 0.5:
                        steered_logits[tk] -= (wm_cfg.alpha * logit_delta[tk].item())

            if self.config.model.do_sample:
                steered_logits = steered_logits / self.config.model.temperature
                probs_sampled = F.softmax(steered_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs_sampled, descending=True)

                mask_p = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > self.config.model.top_p
                sorted_probs[mask_p] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()

                next_token = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
            else:
                next_token = steered_logits.argmax().item()

            generated_ids.append(next_token)
            
            if return_token_details:
                token_str = self.tokenizer.decode([next_token]).replace('\n', '\\n')
                prob_val = probs[next_token].item()
                entropy_val = entropy.item()
                
                if not is_steered_step:
                    token_details_list.append({
                        "token": token_str, "entropy": entropy_val, "prob": prob_val,
                        "valid": False, "wm_score": 0.0, "reason": "Low Entropy"
                    })
                else:
                    wm_score_val = logit_delta[next_token].item()
                    token_details_list.append({
                        "token": token_str, "entropy": entropy_val, "prob": prob_val,
                        "valid": True, "wm_score": wm_score_val, "reason": ""
                    })

            if is_steered_step:
                natural_top_token = original_logits.argmax().item()
                if (next_token != natural_top_token) and (logit_delta[next_token].item() > 0):
                    steered_count += 1        
            
            cur_ids = torch.tensor([[next_token]], device=self.device)
            if next_token == self.tokenizer.eos_token_id:
                break
                
        total_gen = len(generated_ids) - len(prompt_ids)
        print(f"\n[DEBUG] Đã sinh {total_gen} tokens | Nhúng Thủy vân thành công: {steered_count}/{total_gen} tokens.")
        
        new_tokens = generated_ids[len(prompt_ids):]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if return_token_details:
            return generated_text, token_details_list
        return generated_text
        
    @torch.no_grad()
    def generate_unwatermarked(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        max_tokens = max_new_tokens or self.config.model.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=self.config.model.temperature,
            top_p=self.config.model.top_p, do_sample=self.config.model.do_sample
        )
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)