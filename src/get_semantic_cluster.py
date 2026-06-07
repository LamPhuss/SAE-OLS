import torch
import time
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def kmeans_gpu(data_tensor, n_clusters, max_iter=50, batch_size=5000):
    """K-Means trên GPU với tối ưu hóa khoảng cách"""
    N, D = data_tensor.shape
    rand_indices = torch.randperm(N, device=data_tensor.device)[:n_clusters]
    centroids = data_tensor[rand_indices].clone()

    print(f"-> Bắt đầu ép xung K-Means trên GPU ({N} câu văn, {n_clusters} cụm)...")
    for i in range(max_iter):
        t0 = time.time()
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_clusters, 1, device=data_tensor.device)

        for j in range(0, N, batch_size):
            batch = data_tensor[j:j+batch_size]
            distances = torch.cdist(batch, centroids)
            labels = torch.argmin(distances, dim=1)

            new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), batch)
            counts.scatter_add_(0, labels.unsqueeze(1), torch.ones_like(labels.unsqueeze(1), dtype=data_tensor.dtype))

        counts = torch.clamp(counts, min=1e-9)
        centroids = new_centroids / counts
        print(f"   [Vòng lặp {i+1}/{max_iter}] Xong trong {time.time()-t0:.2f} giây")
        
    return centroids

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    print("1. Đang tải Dataset Wikipedia để tạo Mỏ neo Ngữ nghĩa...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:50000]")
    sentences = [text.strip() for text in dataset['text'] if len(text.strip()) > 30]
    
    print(f"2. Đang tải Tokenizer và Llama-3 8-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    model.eval()
    
    if hasattr(model, 'get_output_embeddings') and model.get_output_embeddings() is not None:
        W_U = model.get_output_embeddings().weight.detach().float()
    else:
        W_U = model.model.embed_tokens.weight.detach().float()

    print("3. Đang trích xuất Tọa độ Câu (Sentence Vectors) bằng Llama-3...")
    batch_size = 128
    all_sentence_vectors = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_texts = sentences[i : i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            
            input_ids = inputs.input_ids         
            attention_mask = inputs.attention_mask 
            
            token_vectors = W_U[input_ids] 
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(token_vectors * mask_expanded, dim=1) 
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            sentence_vectors = sum_embeddings / sum_mask 
            all_sentence_vectors.append(sentence_vectors.cpu()) # Cất về CPU cho khỏi tràn RAM

    # Giải phóng LLM
    del model
    torch.cuda.empty_cache()
    
    # Gộp tất cả các vector câu lại thành 1 ma trận siêu khổng lồ
    data_tensor = torch.cat(all_sentence_vectors, dim=0).to(device)
    print(f"-> Đã thu thập {data_tensor.shape[0]} vector câu trong không gian 4096 chiều.")

    print("\n4. Chạy K-Means trên Tọa độ Câu...")
    centroids = kmeans_gpu(data_tensor, n_clusters=4096, max_iter=30, batch_size=5000)

    print("5. Đang lưu Vector Tâm Ngữ nghĩa (Semantic Centroids)...")
    os.makedirs("data", exist_ok=True)
    torch.save(centroids.cpu(), "data/llm_centroids.pt")
    print(f"-> THÀNH CÔNG! Đã tạo xong 4096 Mỏ neo NGỮ NGHĨA chuẩn xác. Kích thước file: {centroids.shape}")

if __name__ == "__main__":
    main()