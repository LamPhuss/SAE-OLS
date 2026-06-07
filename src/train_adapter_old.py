import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import os

class SemanticAdapter(nn.Module):
    def __init__(self, sbert_dim=384, n_clusters=4096):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(sbert_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, n_clusters)
        )
        self.sbert = None
        self.centroids = None

    def init_inference_tools(self, device):
        from sentence_transformers import SentenceTransformer
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.centroids = torch.load("data/llm_centroids.pt", map_location=device)

    def forward(self, x):
        return self.adapter(x)

    def get_geometric_anchor(self, text_context: str):
        device = next(self.parameters()).device
        with torch.no_grad():
            emb = self.sbert.encode(text_context, convert_to_tensor=True, show_progress_bar=False).float().to(device)
            cluster_logits = self.adapter(emb)
            cluster_id = torch.argmax(cluster_logits, dim=-1)
            S_stable = self.centroids[cluster_id]
            return cluster_id, S_stable
            
    def get_geometric_anchor_batch(self, text_contexts: list):
        device = next(self.parameters()).device
        safe_texts = [t if t.strip() else " " for t in text_contexts]
        with torch.no_grad():
            emb = self.sbert.encode(safe_texts, convert_to_tensor=True, show_progress_bar=False).float().to(device)
            cluster_logits = self.adapter(emb)
            cluster_ids = torch.argmax(cluster_logits, dim=-1)
            S_stables = self.centroids[cluster_ids] 
            return cluster_ids, S_stables

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"Đang sử dụng thiết bị: {device}")

    print("Đang tải SBERT và Llama-3 Centroids...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    centroids = torch.load("data/llm_centroids.pt").to(device) 
    
    print(f"Đang trích xuất ma trận W_U từ {model_name}...")

    llama_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    
    if hasattr(llama_model, 'get_output_embeddings') and llama_model.get_output_embeddings() is not None:
        W_U = llama_model.get_output_embeddings().weight.detach().float().to(device)
    else:
        W_U = llama_model.model.embed_tokens.weight.detach().float().to(device)
        
    del llama_model
    torch.cuda.empty_cache()

    print("Đang tải Tokenizer của Llama-3...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Llama 3 không có pad token mặc định, phải set bằng eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    adapter = SemanticAdapter(n_clusters=4096).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    print("Đang tải tập dữ liệu Wikipedia (Mẫu nhỏ)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100000]")
    sentences = [text.strip() for text in dataset['text'] if len(text.strip()) > 30]
    
    batch_size = 128
    epochs = 10
    
    print(f"Bắt đầu huấn luyện trên {len(sentences)} câu văn...")

    adapter.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(range(0, len(sentences), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i in progress_bar:
            batch_texts = sentences[i : i + batch_size]
            if not batch_texts: continue

            optimizer.zero_grad()

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
            
            input_ids = inputs.input_ids         
            attention_mask = inputs.attention_mask 
            
            # Kéo vector 4096 chiều của Llama-3
            token_vectors = W_U[input_ids]       # (batch_size, seq_len, 4096)
            
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(token_vectors * mask_expanded, dim=1) 
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            # Tọa độ 4096 chiều của ngữ cảnh
            sentence_vectors = sum_embeddings / sum_mask 
            
            distances = torch.cdist(sentence_vectors, centroids) 
            target_cluster_ids = torch.argmin(distances, dim=-1) 

            with torch.no_grad():
                sbert_embeddings = sbert.encode(batch_texts, convert_to_tensor=True).float()
            sbert_embeddings = sbert_embeddings.clone().detach()
            
            logits = adapter(sbert_embeddings)

            loss = criterion(logits, target_cluster_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"-> Epoch {epoch+1} Hoàn tất! Average Loss: {total_loss / (len(sentences)//batch_size):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(adapter.state_dict(), "models/semantic_adapter.pth")
    print("\nĐã lưu thành công Lớp Cầu Nối tương thích Llama-3 tại 'models/semantic_adapter.pth'!")

if __name__ == "__main__":
    main()