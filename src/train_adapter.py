import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import os
import time

# --- NÂNG CẤP KIẾN TRÚC MẠNG NƠ-RON ---
class SemanticAdapter(nn.Module):
    def __init__(self, sbert_dim=384, n_clusters=4096):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(sbert_dim, 2048),
            nn.BatchNorm1d(2048), # Thêm BatchNorm để hội tụ siêu tốc
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, n_clusters)
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
            # VÁ LỖI: Bọc text_context trong list [] để ép SBERT trả về mảng 2D (1, 384)
            emb = self.sbert.encode([text_context], convert_to_tensor=True, show_progress_bar=False).float().to(device)
            
            cluster_logits = self.adapter(emb) # Output sẽ là 2D: (1, 4096)
            cluster_id = torch.argmax(cluster_logits, dim=-1)[0] # Lấy phần tử đầu tiên
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
    
    print("1. Đang tải SBERT và Llama-3 Centroids...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    centroids = torch.load("data/llm_centroids.pt").to(device) 
    
    print(f"2. Đang trích xuất ma trận W_U từ {model_name}...")
    llama_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    if hasattr(llama_model, 'get_output_embeddings') and llama_model.get_output_embeddings() is not None:
        W_U = llama_model.get_output_embeddings().weight.detach().float().to(device)
    else:
        W_U = llama_model.model.embed_tokens.weight.detach().float().to(device)
        
    del llama_model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100000]")
    sentences = [text.strip() for text in dataset['text'] if len(text.strip()) > 30]
    
    # =========================================================
    # ĐỘT PHÁ TỐI ƯU: TÍNH TOÁN TRƯỚC (PRE-COMPUTE) X VÀ Y
    # =========================================================
    print(f"\n3. Đang trích xuất Đặc trưng (X) và Nhãn (Y) cho {len(sentences)} câu...")
    batch_size_extract = 256
    all_X, all_Y = [], []
    
    for i in tqdm(range(0, len(sentences), batch_size_extract), desc="Pre-computing"):
        batch_texts = sentences[i : i + batch_size_extract]
        
        # Đầu ra (Target Y): Tìm cụm K-Means gốc của Llama
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        token_vectors = W_U[inputs.input_ids]
        mask = inputs.attention_mask.unsqueeze(-1).float()
        sentence_vectors = torch.sum(token_vectors * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        distances = torch.cdist(sentence_vectors, centroids) 
        target_ids = torch.argmin(distances, dim=-1)
        
        # Đầu vào (Feature X): Vector SBERT
        with torch.no_grad():
            sbert_emb = sbert.encode(batch_texts, convert_to_tensor=True).float()
            
        all_X.append(sbert_emb)
        all_Y.append(target_ids)
        
    tensor_X = torch.cat(all_X, dim=0)
    tensor_Y = torch.cat(all_Y, dim=0)
    
    train_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_Y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    # =========================================================
    # BẮT ĐẦU HUẤN LUYỆN
    # =========================================================
    print("\n4. Bắt đầu ép xung Huấn luyện Adapter...")
    adapter = SemanticAdapter(n_clusters=4096).to(device)
    
    epochs = 100 # Tăng gấp 10 lần số epoch
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-3, weight_decay=1e-4)
    # Bộ giảm tốc tự động: Giúp hội tụ sâu hơn về cuối
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    adapter.train()
    t_start = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = adapter(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        # In log mỗi 10 epoch cho đỡ rối mắt
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:>3}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\n-> Hoàn tất Huấn luyện sau {time.time() - t_start:.2f} giây!")
    
    os.makedirs("models", exist_ok=True)
    torch.save(adapter.state_dict(), "models/semantic_adapter.pth")
    print("-> Đã lưu thành công Semantic Adapter v2.0 tại 'models/semantic_adapter.pth'!")

if __name__ == "__main__":
    main()