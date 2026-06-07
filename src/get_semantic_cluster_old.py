import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import time

def kmeans_gpu(data_tensor, n_clusters, max_iter=30, batch_size=5000):
    """Thuật toán K-Means siêu tốc chạy hoàn toàn trên GPU bằng PyTorch"""
    N, D = data_tensor.shape
    
    # Khởi tạo tâm cụm ngẫu nhiên từ chính các điểm dữ liệu
    rand_indices = torch.randperm(N, device=data_tensor.device)[:n_clusters]
    centroids = data_tensor[rand_indices].clone()

    print(f"-> Bắt đầu ép xung K-Means trên GPU ({N} điểm, {n_clusters} cụm)...")
    for i in range(max_iter):
        t0 = time.time()
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_clusters, 1, device=data_tensor.device)

        # Xử lý theo lô (batch) để GPU không bị quá tải (OOM)
        for j in range(0, N, batch_size):
            batch = data_tensor[j:j+batch_size]
            
            # Tính khoảng cách Euclidean và tìm cụm gần nhất
            distances = torch.cdist(batch, centroids) # (batch_size, n_clusters)
            labels = torch.argmin(distances, dim=1)   # (batch_size,)

            # Cộng dồn tọa độ các điểm vào tâm cụm mới
            new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), batch)
            counts.scatter_add_(0, labels.unsqueeze(1), torch.ones_like(labels.unsqueeze(1), dtype=data_tensor.dtype))

        # Trung bình cộng để ra tâm cụm mới
        counts = torch.clamp(counts, min=1e-9) # Tránh lỗi chia cho 0
        centroids = new_centroids / counts
        
        print(f"   [Vòng lặp {i+1}/{max_iter}] Xong trong {time.time()-t0:.2f} giây")
        
    return centroids

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    print(f"1. Đang tải mô hình {model_name} (Tối ưu 16GB VRAM với 8-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("2. Đang trích xuất ma trận W_U...")
    if hasattr(model, 'get_output_embeddings') and model.get_output_embeddings() is not None:
        W_U_tensor = model.get_output_embeddings().weight.detach().float()
    elif hasattr(model, 'lm_head'):
        W_U_tensor = model.lm_head.weight.detach().float()
    else:
        W_U_tensor = model.model.embed_tokens.weight.detach().float()

    print(f"-> Kích thước thực tế của W_U: {W_U_tensor.shape}")

    # ==========================================================
    # CÚ HÍCH QUAN TRỌNG: ĐÁ LLAMA RA KHỎI VRAM ĐỂ LẤY CHỖ CHO K-MEANS
    # ==========================================================
    print("3. Đang giải phóng VRAM (Xóa Llama-3 khỏi bộ nhớ)...")
    # Đẩy ma trận W_U (chỉ khoảng 2GB) sang GPU để tính toán
    W_U_gpu = W_U_tensor.to(device) 
    
    # Xóa sạch model và dọn rác VRAM
    del model
    torch.cuda.empty_cache()
    print("-> Đã giải phóng thành công! Bắt đầu cất cánh.")

    # ==========================================================
    # CHẠY K-MEANS TRÊN GPU
    # ==========================================================
    print("4. Chạy K-Means bằng CUDA PyTorch...")
    # Chạy 30 vòng lặp là đủ để hội tụ cho bài toán này
    centroids = kmeans_gpu(W_U_gpu, n_clusters=4096, max_iter=30, batch_size=5000)

    print("5. Đang lưu Vector Tâm (Centroids)...")
    import os
    os.makedirs("data", exist_ok=True)
    torch.save(centroids.cpu(), "data/llm_centroids.pt")
    print(f"-> Đã tạo xong 4096 Mỏ neo hình học cho Llama-3! Kích thước file: {centroids.shape}")

if __name__ == "__main__":
    main()