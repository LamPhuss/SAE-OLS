"""
Test SAE Feature Fingerprinting Stability Under Paraphrase.

Câu hỏi cần trả lời:
1. Top-k SAE features có ổn định khi text bị paraphrase không? (Jaccard similarity)
2. Window cố định (50, 100, 200 tokens) vs toàn bộ text: cái nào ổn định hơn?

Cách chạy: python scripts/test_sae_fingerprint.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae import load_sae
from config import SAEConfig

# ==============================================================
# 1. CẶP TEXT GỐC vs PARAPHRASE (hardcoded để test nhanh)
# ==============================================================
PAIRS = [
    # Pair 1: Nhẹ (đổi vài từ)
    (
        "The future of artificial intelligence is incredibly promising. Machine learning models are becoming more powerful each year, and their applications span across healthcare, finance, and education. Researchers believe that within the next decade, AI will transform how we work and live.",
        "The outlook for artificial intelligence is very bright. Deep learning systems grow stronger every year, with uses ranging from medicine, banking, to schooling. Scientists predict that in the coming ten years, AI will revolutionize our work and daily lives."
    ),
    # Pair 2: Trung bình (đổi cấu trúc câu)
    (
        "Climate change poses significant threats to global ecosystems. Rising temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme. Governments around the world must take immediate action to reduce carbon emissions.",
        "Global ecosystems face serious dangers from climate change. The melting of polar ice, rising ocean levels, and increasingly severe weather are all consequences of warming temperatures. Immediate steps to cut carbon output are needed from governments worldwide."
    ),
    # Pair 3: Nặng (viết lại hoàn toàn)
    (
        "Quantum computing represents a paradigm shift in computational power. Unlike classical computers that use bits, quantum computers leverage qubits which can exist in superposition states. This allows them to solve certain problems exponentially faster than traditional machines.",
        "A revolutionary change in computing capability is emerging through quantum technology. Traditional machines rely on binary digits, but quantum processors use quantum bits that simultaneously occupy multiple states. For specific computational challenges, this provides an exponential speed advantage."
    ),
    # Pair 4: Rất nặng (paraphrase + thêm/bớt thông tin)
    (
        "The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections. This vast neural network enables complex cognitive functions including memory, language, and abstract reasoning. Understanding the brain remains one of science's greatest challenges.",
        "With roughly 86 billion nerve cells interconnected through trillions of synapses, the brain is extraordinarily complex. It supports sophisticated mental processes such as remembering, speaking, and thinking abstractly. Neuroscience continues to grapple with the enormous challenge of decoding how the brain truly works."
    ),
]

PAIR_LABELS = ["Light", "Medium", "Heavy", "Very Heavy"]


def get_hidden_states(model, tokenizer, text, target_layer, device):
    """Chạy LLM forward pass, trích xuất hidden states tại target_layer."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    captured = {}
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            captured['h'] = out[0].detach()
        else:
            captured['h'] = out.detach()

    layer = model.model.layers[target_layer]
    handle = layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    return captured['h']  # [1, seq_len, d_model]


def get_sae_fingerprint(sae, hidden_states, top_k):
    """
    Encode hidden states qua SAE, lấy top-k features có mean activation cao nhất.
    Returns: set of feature indices (fingerprint)
    """
    h = hidden_states.squeeze(0).to(sae.W_enc.dtype)  # [seq_len, d_model]
    with torch.no_grad():
        acts = sae.encode(h)  # [seq_len, d_sae]

    # Mean activation across all tokens
    mean_acts = acts.mean(dim=0)  # [d_sae]

    top_indices = torch.topk(mean_acts, k=top_k).indices.tolist()
    return set(top_indices), mean_acts


def jaccard_similarity(set_a, set_b):
    """Jaccard = |A ∩ B| / |A ∪ B|"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def overlap_ratio(set_a, set_b):
    """|A ∩ B| / min(|A|, |B|) — measures how much the smaller set is contained"""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def cosine_sim(acts_a, acts_b):
    """Cosine similarity between two activation vectors."""
    return torch.nn.functional.cosine_similarity(
        acts_a.unsqueeze(0), acts_b.unsqueeze(0)
    ).item()


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model & SAE
    model_name = "google/gemma-2-2b"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, torch_dtype=torch.bfloat16
    )
    model.eval()

    sae_config = SAEConfig()
    print("Loading SAE...")
    sae = load_sae(sae_config, device=device)
    target_layer = sae_config.target_layer  # 20

    # ==============================================================
    # TEST 1: Fingerprint stability across paraphrase levels
    # ==============================================================
    print("\n" + "="*70)
    print("TEST 1: SAE FINGERPRINT STABILITY (full text)")
    print("="*70)

    top_k_values = [10, 20, 50, 100, 200]

    for pair_idx, (text_orig, text_para) in enumerate(PAIRS):
        print(f"\n--- Pair {pair_idx+1}: {PAIR_LABELS[pair_idx]} paraphrase ---")

        h_orig = get_hidden_states(model, tokenizer, text_orig, target_layer, device)
        h_para = get_hidden_states(model, tokenizer, text_para, target_layer, device)

        fp_orig_sets = {}
        fp_para_sets = {}

        for k in top_k_values:
            fp_orig, acts_orig = get_sae_fingerprint(sae, h_orig, k)
            fp_para, acts_para = get_sae_fingerprint(sae, h_para, k)
            fp_orig_sets[k] = fp_orig
            fp_para_sets[k] = fp_para

            jacc = jaccard_similarity(fp_orig, fp_para)
            ovlp = overlap_ratio(fp_orig, fp_para)
            cos = cosine_sim(acts_orig, acts_para)

            print(f"  top-{k:>3d}: Jaccard={jacc:.3f}  Overlap={ovlp:.3f}  CosineSim(acts)={cos:.4f}")

    # ==============================================================
    # TEST 2: Window size comparison
    # ==============================================================
    print("\n" + "="*70)
    print("TEST 2: WINDOW SIZE COMPARISON")
    print("Câu hỏi: Cửa sổ cố định vs toàn bộ text, cái nào ổn định hơn?")
    print("="*70)

    # Dùng pair Medium để test
    text_orig, text_para = PAIRS[1]

    # Tokenize cả hai
    ids_orig = tokenizer.encode(text_orig, add_special_tokens=True)
    ids_para = tokenizer.encode(text_para, add_special_tokens=True)

    print(f"\nOriginal: {len(ids_orig)} tokens")
    print(f"Paraphrase: {len(ids_para)} tokens")

    h_orig_full = get_hidden_states(model, tokenizer, text_orig, target_layer, device)
    h_para_full = get_hidden_states(model, tokenizer, text_para, target_layer, device)

    # Test different windows: last N tokens, hoặc full
    window_sizes = [20, 50, 100, "full"]
    k_test = 50  # dùng top-50 cho test này

    print(f"\nUsing top-{k_test} features:")
    print(f"{'Window':<12} | {'Jaccard':>8} | {'Overlap':>8} | {'Cosine':>8}")
    print("-" * 45)

    for window in window_sizes:
        if window == "full":
            h_o = h_orig_full
            h_p = h_para_full
            label = "full"
        else:
            # Lấy last-N tokens
            h_o = h_orig_full[:, -min(window, h_orig_full.shape[1]):, :]
            h_p = h_para_full[:, -min(window, h_para_full.shape[1]):, :]
            label = f"last-{window}"

        fp_o, acts_o = get_sae_fingerprint(sae, h_o, k_test)
        fp_p, acts_p = get_sae_fingerprint(sae, h_p, k_test)

        jacc = jaccard_similarity(fp_o, fp_p)
        ovlp = overlap_ratio(fp_o, fp_p)
        cos = cosine_sim(acts_o, acts_p)

        print(f"{label:<12} | {jacc:>8.3f} | {ovlp:>8.3f} | {cos:>8.4f}")

    # ==============================================================
    # TEST 3: Stability of RANKING (không chỉ set, mà còn thứ tự)
    # ==============================================================
    print("\n" + "="*70)
    print("TEST 3: ACTIVATION RANKING STABILITY")
    print("Top features có giữ nguyên THỨ TỰ sau paraphrase không?")
    print("="*70)

    for pair_idx, (text_orig, text_para) in enumerate(PAIRS):
        print(f"\n--- Pair {pair_idx+1}: {PAIR_LABELS[pair_idx]} ---")

        h_orig = get_hidden_states(model, tokenizer, text_orig, target_layer, device)
        h_para = get_hidden_states(model, tokenizer, text_para, target_layer, device)

        _, acts_orig = get_sae_fingerprint(sae, h_orig, 10)
        _, acts_para = get_sae_fingerprint(sae, h_para, 10)

        # Top-20 features by activation (ranked)
        rank_orig = torch.topk(acts_orig, 20).indices.tolist()
        rank_para = torch.topk(acts_para, 20).indices.tolist()

        # Spearman-like: how many of top-20 original are in top-20 paraphrase?
        common = set(rank_orig) & set(rank_para)

        # Kendall tau trên common features
        print(f"  Top-20 overlap: {len(common)}/20 features in common")
        print(f"  Original  top-10: {rank_orig[:10]}")
        print(f"  Paraphrase top-10: {rank_para[:10]}")

    print("\n" + "="*70)
    print("KẾT LUẬN")
    print("="*70)
    print("- Nếu Jaccard > 0.7 với top-50: Fingerprinting khả thi")
    print("- Nếu Jaccard < 0.5: Cần dùng soft matching (cosine trên activation vector)")
    print("- Window lớn hơn thường ổn định hơn (pha loãng noise)")


if __name__ == "__main__":
    main()
