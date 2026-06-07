"""
Test LSH on SAE Activation Vectors.

Câu hỏi cần trả lời:
1. LSH có map text gốc và paraphrase vào cùng bucket không?
2. Bao nhiêu hyperplanes là tối ưu? (trade-off: ít → collision cao nhưng false positive cao)
3. Multi-probe LSH có cải thiện không?
4. So sánh: dùng activation vector trực tiếp (cosine → delta_h) vs hash thành bucket rời rạc

Cách chạy: python scripts/test_lsh_fingerprint.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae import load_sae
from config import SAEConfig

# ==============================================================
# TEXT PAIRS (same as fingerprint test)
# ==============================================================
PAIRS = [
    (
        "The future of artificial intelligence is incredibly promising. Machine learning models are becoming more powerful each year, and their applications span across healthcare, finance, and education. Researchers believe that within the next decade, AI will transform how we work and live.",
        "The outlook for artificial intelligence is very bright. Deep learning systems grow stronger every year, with uses ranging from medicine, banking, to schooling. Scientists predict that in the coming ten years, AI will revolutionize our work and daily lives."
    ),
    (
        "Climate change poses significant threats to global ecosystems. Rising temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme. Governments around the world must take immediate action to reduce carbon emissions.",
        "Global ecosystems face serious dangers from climate change. The melting of polar ice, rising ocean levels, and increasingly severe weather are all consequences of warming temperatures. Immediate steps to cut carbon output are needed from governments worldwide."
    ),
    (
        "Quantum computing represents a paradigm shift in computational power. Unlike classical computers that use bits, quantum computers leverage qubits which can exist in superposition states. This allows them to solve certain problems exponentially faster than traditional machines.",
        "A revolutionary change in computing capability is emerging through quantum technology. Traditional machines rely on binary digits, but quantum processors use quantum bits that simultaneously occupy multiple states. For specific computational challenges, this provides an exponential speed advantage."
    ),
    (
        "The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections. This vast neural network enables complex cognitive functions including memory, language, and abstract reasoning. Understanding the brain remains one of science's greatest challenges.",
        "With roughly 86 billion nerve cells interconnected through trillions of synapses, the brain is extraordinarily complex. It supports sophisticated mental processes such as remembering, speaking, and thinking abstractly. Neuroscience continues to grapple with the enormous challenge of decoding how the brain truly works."
    ),
]

PAIR_LABELS = ["Light", "Medium", "Heavy", "Very Heavy"]

# Thêm text hoàn toàn khác topic để test false positive
NEGATIVE_TEXTS = [
    "The recipe for chocolate cake requires flour, sugar, cocoa powder, and eggs. Mix the dry ingredients first, then add the wet ingredients gradually. Bake at 350 degrees for about 30 minutes until a toothpick comes out clean.",
    "Basketball was invented by James Naismith in 1891. The sport quickly gained popularity across the United States and eventually became one of the most watched sports globally. The NBA finals attract millions of viewers each year.",
    "The stock market experienced significant volatility during the first quarter. Investors were cautious due to rising interest rates and geopolitical tensions. Technology stocks were particularly affected by the changing economic landscape.",
]


def get_hidden_states(model, tokenizer, text, target_layer, device):
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
    return captured['h']


def get_mean_sae_activation(sae, hidden_states):
    """Returns mean activation vector [d_sae]"""
    h = hidden_states.squeeze(0).to(sae.W_enc.dtype)
    with torch.no_grad():
        acts = sae.encode(h)
    return acts.mean(dim=0).float()


# ==============================================================
# LSH IMPLEMENTATIONS
# ==============================================================

class SimHashLSH:
    """
    SimHash LSH: sign(R @ x) → binary hash.
    Vectors with high cosine similarity → same hash with high probability.

    P(same bit) = 1 - arccos(cos_sim) / π
    For cos_sim=0.95: P ≈ 0.90 per bit
    For cos_sim=0.50: P ≈ 0.67 per bit
    """
    def __init__(self, d_input, n_bits, secret_key="default", device="cpu"):
        self.n_bits = n_bits
        self.device = device
        # Random hyperplanes seeded by secret_key
        seed = int(hashlib.sha256(f"{secret_key}_LSH".encode()).hexdigest()[:8], 16)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        self.hyperplanes = torch.randn(n_bits, d_input, generator=rng, device=device)
        # Normalize each hyperplane
        self.hyperplanes = self.hyperplanes / self.hyperplanes.norm(dim=1, keepdim=True)

    def hash(self, x):
        """Returns binary hash as tuple of 0/1"""
        projections = self.hyperplanes @ x  # [n_bits]
        return tuple((projections > 0).int().tolist())

    def hamming_distance(self, hash_a, hash_b):
        """Number of differing bits"""
        return sum(a != b for a, b in zip(hash_a, hash_b))

    def hash_match_ratio(self, hash_a, hash_b):
        """Fraction of matching bits"""
        return 1.0 - self.hamming_distance(hash_a, hash_b) / self.n_bits


class MultiBandLSH:
    """
    Multi-band LSH: Chia n_bits thành B bands, mỗi band R bits.
    Hai vector match nếu BẤT KỲ band nào trùng hoàn toàn.

    Với n_bits=64, bands=8, rows=8:
    - cos_sim=0.95: P(match) ≈ 1 - (1 - 0.90^8)^8 ≈ 0.997
    - cos_sim=0.50: P(match) ≈ 1 - (1 - 0.67^8)^8 ≈ 0.297
    """
    def __init__(self, d_input, n_bands, rows_per_band, secret_key="default", device="cpu"):
        self.n_bands = n_bands
        self.rows_per_band = rows_per_band
        n_bits = n_bands * rows_per_band
        self.simhash = SimHashLSH(d_input, n_bits, secret_key, device)

    def get_band_hashes(self, x):
        """Returns list of band hashes"""
        full_hash = self.simhash.hash(x)
        bands = []
        for b in range(self.n_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            bands.append(full_hash[start:end])
        return bands

    def match(self, x1, x2):
        """True if ANY band matches exactly"""
        bands1 = self.get_band_hashes(x1)
        bands2 = self.get_band_hashes(x2)
        for b1, b2 in zip(bands1, bands2):
            if b1 == b2:
                return True
        return False

    def num_matching_bands(self, x1, x2):
        bands1 = self.get_band_hashes(x1)
        bands2 = self.get_band_hashes(x2)
        return sum(b1 == b2 for b1, b2 in zip(bands1, bands2))


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    target_layer = sae_config.target_layer
    d_sae = sae.d_sae  # 16384

    # ==============================================================
    # Compute all activation vectors
    # ==============================================================
    print("\nComputing SAE activation vectors...")

    pair_acts = []  # [(act_orig, act_para), ...]
    for text_orig, text_para in PAIRS:
        h_orig = get_hidden_states(model, tokenizer, text_orig, target_layer, device)
        h_para = get_hidden_states(model, tokenizer, text_para, target_layer, device)
        act_orig = get_mean_sae_activation(sae, h_orig)
        act_para = get_mean_sae_activation(sae, h_para)
        pair_acts.append((act_orig, act_para))

    neg_acts = []
    for text in NEGATIVE_TEXTS:
        h = get_hidden_states(model, tokenizer, text, target_layer, device)
        neg_acts.append(get_mean_sae_activation(sae, h))

    # ==============================================================
    # TEST 1: SimHash — varying number of bits
    # ==============================================================
    print("\n" + "="*70)
    print("TEST 1: SimHash — Bit count sweep")
    print("Bao nhiêu bits tối ưu để paraphrase pairs match, negatives don't?")
    print("="*70)

    bit_counts = [8, 16, 32, 48, 64, 96, 128]
    secret_key = "test_key_2026"

    print(f"\n{'Bits':>6} | {'Pair':>12} | {'Match%':>7} | {'Hamming':>8} | {'Cosine':>7}")
    print("-" * 55)

    for n_bits in bit_counts:
        lsh = SimHashLSH(d_sae, n_bits, secret_key, device)

        # Positive pairs (should match)
        for i, (act_o, act_p) in enumerate(pair_acts):
            h_o = lsh.hash(act_o)
            h_p = lsh.hash(act_p)
            match_ratio = lsh.hash_match_ratio(h_o, h_p)
            hamming = lsh.hamming_distance(h_o, h_p)
            cos = torch.nn.functional.cosine_similarity(act_o.unsqueeze(0), act_p.unsqueeze(0)).item()
            exact = "EXACT" if h_o == h_p else ""
            print(f"{n_bits:>6} | {PAIR_LABELS[i]:>12} | {match_ratio*100:>6.1f}% | {hamming:>8d} | {cos:>7.4f} {exact}")

        # Negative pairs (should NOT match)
        for j, neg_act in enumerate(neg_acts):
            h_orig = lsh.hash(pair_acts[0][0])  # Compare against Pair 1 original
            h_neg = lsh.hash(neg_act)
            match_ratio = lsh.hash_match_ratio(h_orig, h_neg)
            hamming = lsh.hamming_distance(h_orig, h_neg)
            cos = torch.nn.functional.cosine_similarity(pair_acts[0][0].unsqueeze(0), neg_act.unsqueeze(0)).item()
            print(f"{n_bits:>6} | {'NEG-'+str(j+1):>12} | {match_ratio*100:>6.1f}% | {hamming:>8d} | {cos:>7.4f}")
        print()

    # ==============================================================
    # TEST 2: Multi-Band LSH — best config search
    # ==============================================================
    print("\n" + "="*70)
    print("TEST 2: Multi-Band LSH — Config sweep")
    print("Tìm (bands, rows) tối ưu: paraphrase=match, negative=no match")
    print("="*70)

    configs = [
        (4, 2),   # 8 bits, very coarse
        (4, 4),   # 16 bits
        (8, 4),   # 32 bits
        (8, 8),   # 64 bits
        (16, 4),  # 64 bits (more bands, fewer rows)
        (16, 8),  # 128 bits
    ]

    print(f"\n{'Config':>12} | {'Bits':>5} | ", end="")
    for label in PAIR_LABELS:
        print(f"{label:>8} | ", end="")
    print(f"{'NEG-1':>8} | {'NEG-2':>8} | {'NEG-3':>8}")
    print("-" * 110)

    for n_bands, rows in configs:
        mb_lsh = MultiBandLSH(d_sae, n_bands, rows, secret_key, device)
        n_bits = n_bands * rows
        label = f"B={n_bands},R={rows}"

        print(f"{label:>12} | {n_bits:>5} | ", end="")

        # Positive pairs
        for act_o, act_p in pair_acts:
            matched = mb_lsh.match(act_o, act_p)
            n_match = mb_lsh.num_matching_bands(act_o, act_p)
            print(f"{'YES':>4}({n_match:>2})" if matched else f"{'NO':>4}({n_match:>2})", end=" | ")

        # Negative pairs
        for neg_act in neg_acts:
            matched = mb_lsh.match(pair_acts[0][0], neg_act)
            n_match = mb_lsh.num_matching_bands(pair_acts[0][0], neg_act)
            print(f"{'YES':>4}({n_match:>2})" if matched else f"{'NO':>4}({n_match:>2})", end=" | ")
        print()

    # ==============================================================
    # TEST 3: Phương án thay thế — Dùng cosine trực tiếp (không hash)
    # ==============================================================
    print("\n" + "="*70)
    print("TEST 3: DIRECT COSINE (không cần LSH)")
    print("Nếu cosine đủ cao → dùng activation vector trực tiếp làm anchor")
    print("Không cần hash rời rạc, delta_h sẽ gần giống nhau tự nhiên")
    print("="*70)

    print(f"\n{'Pair':>12} | {'Cos(act)':>12} | {'Cos(S_proj)':>12} | {'Cos(delta_h)':>16} | {'|delta_h diff|':>15}")
    print("-" * 80)

    # Simulate: dùng activation vector làm S_stable, tính delta_h
    # activation [16384] → chiếu về d_model [2304] qua SAE decoder: S = act @ W_dec
    # Giả lập v_target
    seed = int(hashlib.sha256(f"{secret_key}_target".encode()).hexdigest()[:8], 16)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    target_idx = torch.randint(0, d_sae, (1,), generator=rng, device=device).item()
    v_target = sae.get_feature_vector(target_idx).float()  # [d_model=2304]

    W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]

    def act_to_stable(act):
        """Chiếu activation [d_sae] về d_model [2304] qua W_dec"""
        return (act @ W_dec)  # [d_model]

    for i, (act_o, act_p) in enumerate(pair_acts):
        # Chiếu về d_model space
        S_o = act_to_stable(act_o)
        S_p = act_to_stable(act_p)

        # Tính delta_h từ S gốc
        S_norm_o = S_o / (S_o.norm() + 1e-8)
        delta_h_o = v_target - torch.dot(v_target, S_norm_o) * S_norm_o
        delta_h_o = delta_h_o / (delta_h_o.norm() + 1e-8)

        # Tính delta_h từ S paraphrase
        S_norm_p = S_p / (S_p.norm() + 1e-8)
        delta_h_p = v_target - torch.dot(v_target, S_norm_p) * S_norm_p
        delta_h_p = delta_h_p / (delta_h_p.norm() + 1e-8)

        cos_act = torch.nn.functional.cosine_similarity(act_o.unsqueeze(0), act_p.unsqueeze(0)).item()
        cos_s = torch.nn.functional.cosine_similarity(S_o.unsqueeze(0), S_p.unsqueeze(0)).item()
        cos_dh = torch.nn.functional.cosine_similarity(delta_h_o.unsqueeze(0), delta_h_p.unsqueeze(0)).item()
        diff_norm = (delta_h_o - delta_h_p).norm().item()

        print(f"{PAIR_LABELS[i]:>12} | {cos_act:>12.4f} | {cos_s:>12.4f} | {cos_dh:>16.4f} | {diff_norm:>15.6f}")

    # Negatives
    for j, neg_act in enumerate(neg_acts):
        act_o = pair_acts[0][0]
        S_o = act_to_stable(act_o)
        S_norm_o = S_o / (S_o.norm() + 1e-8)
        delta_h_o = v_target - torch.dot(v_target, S_norm_o) * S_norm_o
        delta_h_o = delta_h_o / (delta_h_o.norm() + 1e-8)

        S_n = act_to_stable(neg_act)
        S_norm_n = S_n / (S_n.norm() + 1e-8)
        delta_h_n = v_target - torch.dot(v_target, S_norm_n) * S_norm_n
        delta_h_n = delta_h_n / (delta_h_n.norm() + 1e-8)

        cos_act = torch.nn.functional.cosine_similarity(act_o.unsqueeze(0), neg_act.unsqueeze(0)).item()
        cos_s = torch.nn.functional.cosine_similarity(S_o.unsqueeze(0), S_n.unsqueeze(0)).item()
        cos_dh = torch.nn.functional.cosine_similarity(delta_h_o.unsqueeze(0), delta_h_n.unsqueeze(0)).item()
        diff_norm = (delta_h_o - delta_h_n).norm().item()

        print(f"{'NEG-'+str(j+1):>12} | {cos_act:>12.4f} | {cos_s:>12.4f} | {cos_dh:>16.4f} | {diff_norm:>15.6f}")

    # ==============================================================
    # SUMMARY
    # ==============================================================
    print("\n" + "="*70)
    print("TÓM TẮT & KHUYẾN NGHỊ")
    print("="*70)
    print("""
Có 2 đường đi:

[A] LSH Bucket (rời rạc):
    + Generator & Detector luôn đồng bộ 100% nếu cùng bucket
    + Có thể hash bucket → seed → deterministic
    - Cần tune (bands, rows) cẩn thận
    - Paraphrase nặng có thể rơi khác bucket → mất tín hiệu hoàn toàn

[B] Direct Cosine (liên tục):
    + Không cần tune hyperparameters
    + Graceful degradation: paraphrase nặng → cosine giảm nhẹ → delta_h lệch nhẹ
      → score giảm nhẹ (thay vì mất hoàn toàn)
    - Generator & Detector tính delta_h hơi khác nhau
      → nhưng nếu cosine(delta_h) > 0.99 thì sai số rất nhỏ

Nếu cosine(delta_h) > 0.99 cho paraphrase pairs → Phương án B là tối ưu.
Nếu cosine(delta_h) < 0.95 → Cần LSH (Phương án A) để đồng bộ chính xác.
""")


if __name__ == "__main__":
    main()
