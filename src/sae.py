import torch
import torch.nn as nn
from sae_lens import SAE

class SAELensWrapper(nn.Module):
    """
    Lớp bọc (Wrapper) biến object SAE của thư viện sae_lens 
    trở nên tương thích hoàn hảo với giao diện SAE-OLS cũ của bạn.
    """
    def __init__(self, sae: SAE):
        super().__init__()
        self.sae = sae
        
        # Đồng bộ các biến cấu hình để generator.py và detector.py gọi được
        self.d_sae = sae.cfg.d_sae
        self.d_model = sae.cfg.d_in
        self.device = sae.device

    def get_feature_vector(self, feature_idx: int) -> torch.Tensor:
        """
        Trích xuất vector ngữ nghĩa (Feature Direction) từ ma trận Decoder của SAE.
        Trong SAELens, ma trận W_dec có kích thước [d_sae, d_in].
        """
        # Lấy hàng thứ feature_idx, clone và detach để không ảnh hưởng gradient
        feature_vector = self.sae.W_dec[feature_idx].detach().clone()
        return feature_vector

def load_sae(config, device: str = "cuda") -> SAELensWrapper:
    """
    Tải SAE từ HuggingFace thông qua SAELens và bọc nó lại.
    """
    print(f"\n[SAE] Đang tải SAE qua SAELens: release='{config.release}', id='{config.sae_id}'...")
    
    # SAELens trả về tuple: (sae, cfg_dict, sparsity)
    # Ta chỉ cần lấy object sae đầu tiên
    sae, _, _ = SAE.from_pretrained(
        release=config.release,
        sae_id=config.sae_id,
        device=device
    )
    
    print(f"[SAE] Tải thành công! Kích thước: {sae.cfg.d_in} -> {sae.cfg.d_sae} features")
    
    return SAELensWrapper(sae)