import torch
import math
import hashlib
from sentence_transformers import SentenceTransformer

class SemanticAnchor:
    def __init__(self, d_model: int, device: str):
        print("\n[SemanticAnchor] Loading lightweight MiniLM-L6-v2 for LSH...")
        self.device = device
        self.d_model = d_model
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.lsh_cache = {}
        self.n_bits = 12 # Số bit LSH. 12 bits = 4096 buckets (Đủ an toàn, tìm lân cận nhanh)

    def _init_lsh_matrices(self, secret_key: str):
        """Khởi tạo Ma trận Siêu mặt phẳng (LSH) và Ma trận Ánh xạ (Map) tĩnh."""
        if secret_key not in self.lsh_cache:
            seed = int(hashlib.sha256(f"{secret_key}_LSH".encode()).hexdigest()[:8], 16)
            rng = torch.Generator(device=self.device)
            rng.manual_seed(seed)
            
            # R_lsh: Chiếu 384 chiều của SBERT xuống n_bits (Siêu mặt phẳng)
            R_lsh = torch.randn(384, self.n_bits, generator=rng, device=self.device)
            # R_map: Phóng n_bits ngược lên không gian d_model của LLM
            R_map = torch.randn(self.n_bits, self.d_model, generator=rng, device=self.device) / math.sqrt(self.n_bits)
            
            self.lsh_cache[secret_key] = (R_lsh, R_map)
        return self.lsh_cache[secret_key]

    def get_lsh_components(self, text_context: str, secret_key: str):
        """Trả về cả mã Hash nhị phân và Vector S_stable."""
        R_lsh, R_map = self._init_lsh_matrices(secret_key)
        
        if not text_context.strip():
            h_bin = torch.ones(self.n_bits, device=self.device)
            return h_bin, torch.matmul(h_bin, R_map)
            
        with torch.no_grad():
            emb = self.sbert.encode(text_context, convert_to_tensor=True, show_progress_bar=False).float()
            
        # LSH Toán học: Nhân ma trận và lấy Dấu (Sign)
        h_raw = torch.matmul(emb, R_lsh)
        h_bin = (h_raw > 0).float() * 2.0 - 1.0 # Tạo vector chứa -1.0 và 1.0
        
        s_stable = torch.matmul(h_bin, R_map)
        return h_bin, s_stable

    def get_vector_from_hash(self, h_bin: torch.Tensor, secret_key: str):
        """Tái tạo S_stable từ một mã Hash (Dùng cho Máy dò khi lật bit lân cận)."""
        _, R_map = self._init_lsh_matrices(secret_key)
        return torch.matmul(h_bin, R_map)
    
    def get_lsh_components_batch(self, text_contexts: list, secret_key: str) -> torch.Tensor:
        """Xử lý nhiều cửa sổ ngữ cảnh cùng lúc để tận dụng sức mạnh song song của GPU."""
        R_lsh, _ = self._init_lsh_matrices(secret_key)
        
        # Đảm bảo không có chuỗi rỗng gây lỗi SBERT
        safe_texts = [t if t.strip() else " " for t in text_contexts]
            
        with torch.no_grad():
            emb = self.sbert.encode(safe_texts, convert_to_tensor=True, show_progress_bar=False).float()
            
        # Nhân ma trận hàng loạt (Batch Matrix Multiplication)
        h_raw = torch.matmul(emb, R_lsh)
        h_bin = (h_raw > 0).float() * 2.0 - 1.0 
        
        return h_bin # Trả về tensor kích thước: (batch_size, n_bits)