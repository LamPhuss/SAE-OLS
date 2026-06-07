import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from sae import JumpReLUSAE, load_sae
from prf import get_static_prompt_seed
from orthogonal import compute_orthogonal_steering_vector
from semantic_anchor import SemanticAnchor

class WatermarkedGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name_or_path,
            device_map=self.device,
            dtype=getattr(torch, config.model.torch_dtype),
        )
        self.model.eval()
        self._W_U = self._get_unembedding_matrix()
        self.sae = load_sae(config.sae, device=self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.d_model = self.model.config.hidden_size
        self.anchor = SemanticAnchor(self.d_model, self.device)

    def _get_unembedding_matrix(self) -> torch.Tensor:
        if hasattr(self.model, 'lm_head'): return self.model.lm_head.weight.detach()
        elif hasattr(self.model, 'embed_out'): return self.model.embed_out.weight.detach()
        else: raise ValueError("Cannot find unembedding matrix")

    @torch.no_grad()
    def generate(self, prompt: str, secret_key: str, max_new_tokens: Optional[int] = None) -> str:
        max_tokens = max_new_tokens or self.config.model.max_new_tokens
        wm_cfg = self.config.watermark
        sae_cfg = self.config.sae

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = inputs["input_ids"][0].tolist()
        generated_ids = prompt_ids.copy()

        # ==========================================================
        # 1. KHỞI TẠO MỤC TIÊU TĨNH & DÒNG CHẢY NGỮ NGHĨA
        # ==========================================================
        seed = get_static_prompt_seed(prompt_ids, secret_key)
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        target_idx = torch.randint(0, self.sae.d_sae, (1,), generator=rng, device=self.device).item()
        v_target = self.sae.get_feature_vector(target_idx).float()

        past_key_values = None
        cur_ids = inputs.input_ids

        macro_window = 200

        for step in range(max_tokens):
            # ==========================================================
            # 1. CHẠY LLM (Tối ưu siêu tốc với KV Cache)
            # ==========================================================
            outputs = self.model(cur_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            original_logits = outputs.logits[0, -1, :]

            # Cổng gác Entropy: Kiểm tra xem LLM có đang phân vân không
            probs = torch.softmax(original_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))

            # Nếu LLM quá chắc chắn (ví dụ: các từ nối), bỏ qua giấu Thủy vân
            if entropy.item() < 2.0:
                steered_logits = original_logits.float()
            else:
                # ==========================================================
                # 2. TOÁN HỌC LSH ANCHORING (NEO BẰNG BUCKET NGỮ NGHĨA)
                # ==========================================================
                window_size = 50 # Lấy 50 token liền trước
                ctx_ids = generated_ids[-window_size:] if len(generated_ids) > window_size else generated_ids
                ctx_text = self.tokenizer.decode(ctx_ids, skip_special_tokens=True)
                
                # Băm ngữ cảnh thành mã LSH và lấy luôn vector S_stable
                # Mọi câu đồng nghĩa rơi vào cùng Bucket sẽ có S_stable giống hệt nhau 100%
                _, S_stable = self.anchor.get_lsh_components(ctx_text, secret_key)

                # Nắn OLS vuông góc với S_stable
                S_norm = S_stable / (S_stable.norm() + 1e-8)
                delta_h = v_target - torch.dot(v_target.squeeze(), S_norm.squeeze()) * S_norm
                delta_h = delta_h / (delta_h.norm() + 1e-8)

                # ==========================================================
                # 3. LÁ CHẮN KÉP (DOUBLE-SHIELD) & KHUẾCH ĐẠI TÍN HIỆU
                # ==========================================================
                # Tính lượng thay đổi điểm số thô trên toàn bộ từ điển
                logit_delta_raw = (self._W_U @ delta_h.to(self._W_U.dtype)).float()

                # Trích xuất Top 10 từ tự nhiên nhất
                V_c = 10
                top_Vc_vals, top_Vc_indices = torch.topk(original_logits, V_c)
                max_delta = logit_delta_raw[top_Vc_indices].abs().max()

                # Khuếch đại tín hiệu sao cho biên độ lớn nhất đạt mức 2.0
                if max_delta > 1e-8:
                    logit_delta = (logit_delta_raw / max_delta) * 2.0
                else:
                    logit_delta = logit_delta_raw

                # Bơm Thủy vân vào Logits gốc
                steered_logits = original_logits.float() + (wm_cfg.alpha * logit_delta)

                # Lá chắn 1: Khóa điểm của tất cả các từ nằm ngoài Top 10 (trả về như cũ)
                mask = torch.ones_like(original_logits, dtype=torch.bool, device=self.device)
                mask[top_Vc_indices] = False
                steered_logits[mask] = original_logits[mask].float()

                # Lá chắn 2: Trần điểm số (Anti-Spike) - Không cho phép từ nào vọt quá điểm Top-1 gốc + 0.5
                steered_logits = torch.clamp(steered_logits, max=top_Vc_vals[0].float() + 0.5)

            # ==========================================================
            # 4. LẤY MẪU VÀ SINH TỪ (SAMPLING)
            # ==========================================================
            if self.config.model.do_sample:
                # Áp dụng Temperature
                steered_logits = steered_logits / self.config.model.temperature
                probs_sampled = F.softmax(steered_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs_sampled, descending=True)

                # Áp dụng Top-p (Nucleus Sampling)
                mask_p = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > self.config.model.top_p
                sorted_probs[mask_p] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()

                # Gieo xúc xắc chọn từ
                next_token = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
            else:
                next_token = steered_logits.argmax().item()

            generated_ids.append(next_token)

            # Cập nhật cur_ids cho bước tiếp theo (chỉ cần truyền 1 token vừa sinh vì đã dùng KV Cache)
            cur_ids = torch.tensor([[next_token]], device=self.device)

            # Kiểm tra xem câu đã kết thúc chưa
            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_unwatermarked(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        max_tokens = max_new_tokens or self.config.model.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=self.config.model.temperature,
            top_p=self.config.model.top_p, do_sample=self.config.model.do_sample
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
