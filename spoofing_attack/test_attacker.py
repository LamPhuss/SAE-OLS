#!/usr/bin/env python3
"""
test_attacker.py — Test Spoofing Attack on SAE-OLS Watermark (Llama-3-8B).
Attacker learns token transition patterns from watermarked vs baseline texts,
then generates spoofed text that mimics the watermark signature.
"""

import sys, os, time, csv

# Set HF_HOME TRƯỚC mọi import huggingface
os.environ["HF_HOME"] = "/media/ics-security/Data/PhuPham/huggingface_cache"

# Đường dẫn an toàn
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_this_dir, '..', 'src')
sys.path.insert(0, _src_dir)
sys.path.insert(1, _this_dir)

from attacker import WatermarkLearner, AttackerLLM
from config import Config
from detector import WatermarkDetector

# ── Config ──
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" # Chuyển sang Llama-3
DEVICE = "cuda"
SECRET_KEY = "eval_secret_key_2026"
NUM_LEARN = 500       # Giảm xuống 500 để test nhanh, có thể tăng lên nếu RAM chịu được
NUM_TEST = 10         
PREVCTX_WIDTH = 1     # Giảm xuống 1 vì Llama-3 không gian vocab rất lớn (128k), để 3 sẽ bị Sparse Data
SPOOFER_STRENGTH = 7.5

print("=" * 60)
print("TEST: SAE-OLS Spoofing Attack Feasibility (Llama-3-8B)")
print("=" * 60)

# ── 1. Load data from CSV ──
CSV_PATH = os.path.join(_this_dir, "..", "eval_data_saeols_10000.csv")
print(f"\nLoading data from {CSV_PATH}...")

prompts, wm_suffixes, base_suffixes = [], [], []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompts.append(row["prompt"])
        wm_suffixes.append(row["wm_text"])
        base_suffixes.append(row["uwm_text"])

print(f"  Total rows: {len(prompts)}")

# ── 2. Load SAE-OLS detector ──
print("\nLoading SAE-OLS detector...")
config = Config()
detector = WatermarkDetector(config)
tokenizer = detector.tokenizer

# ── Ghép nối Dữ liệu cho Learner ──
# Bắt buộc phải nối Prompt và Suffix để Learner học được context ranh giới
system_instruction = "You are a highly detailed academic assistant. You MUST provide exhaustive, multi-paragraph answers (minimum 250 words) to every prompt. Do not be brief."

wm_learn_full, base_learn_full = [], []
print("Preparing learning data (Applying Chat Templates)...")
for i in range(min(NUM_LEARN, len(prompts))):
    messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompts[i]}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    wm_learn_full.append(formatted_prompt + wm_suffixes[i])
    base_learn_full.append(formatted_prompt + base_suffixes[i])

print(f"  Using {len(wm_learn_full)} texts for learning")

# ── 3. Learn watermark patterns ──
print("\n--- Learning Phase ---")
learner = WatermarkLearner(tokenizer=tokenizer, prevctx_width=PREVCTX_WIDTH)

t0 = time.time()
learner.learn_from_watermarked(wm_learn_full)
learner.learn_from_baseline(base_learn_full)
learn_time = time.time() - t0

print(f"  WM counts:   {learner.counts_wm.total_counts():>12,}")
print(f"  Base counts: {learner.counts_base.total_counts():>12,}")
print(f"  Learning time: {learn_time:.1f}s")

# ── 4. Build spoofer & attacker ──
print("\nBuilding spoofer...")
# Chia sẻ model từ Detector sang Attacker để tiết kiệm VRAM
attacker = AttackerLLM(
    model_name=MODEL_NAME, device=DEVICE,
    model=detector.model, tokenizer=tokenizer,
)
vocab_size = detector.model.config.vocab_size
spoofer = learner.build_spoofer(vocab_size, spoofer_strength=SPOOFER_STRENGTH)

# ── 5. Generate spoofed text & verify detection ──
print(f"\n--- Spoofer Output ({NUM_TEST} samples) ---")
spf_z_scores = []
test_idx_start = NUM_LEARN
test_prompts_raw = prompts[test_idx_start:test_idx_start + NUM_TEST]

if len(test_prompts_raw) < NUM_TEST:
    test_prompts_raw = prompts[:NUM_TEST]

for i, raw_prompt in enumerate(test_prompts_raw):
    # Đóng gói Chat Template
    messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": raw_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_len_ids = len(tokenizer.encode(formatted_prompt, add_special_tokens=True))

    # Tấn công (Sinh văn bản giả mạo)
    _, outputs = attacker.generate(
        [formatted_prompt],
        max_length=250,
        temperature=0.7,
        static_spoofer=spoofer,
    )
    
    # Chỉ trích xuất phần Suffix sinh ra
    spoofed_suffix = tokenizer.decode(outputs[0][p_len_ids:], skip_special_tokens=True)

    # Chấm điểm bằng API Detector chuẩn xác của Llama-3
    result = detector.detect(formatted_prompt, spoofed_suffix, SECRET_KEY)
    z = result.z_score
    spf_z_scores.append(z)
    print(f"  [SPF {i:2d}] z={z:6.2f} | detected={result.is_watermarked}")

# ── 6. Results ──
avg_z = sum(spf_z_scores) / len(spf_z_scores)
detected_count = sum(1 for z in spf_z_scores if z > 4.0)

print(f"\n{'=' * 60}")
print(f"RESULTS:")
print(f"  Avg Spoofer z-score : {avg_z:.2f}")
print(f"  Detected as WM      : {detected_count}/{NUM_TEST} ({detected_count/NUM_TEST:.0%})")
print(f"  WM counts collected : {learner.counts_wm.total_counts():,}")
print(f"  Spoofer strength    : {SPOOFER_STRENGTH}")
print(f"  PrevCtx width       : {PREVCTX_WIDTH}")

if avg_z > 4.0:
    print(f"\n  [!] Spoofer SUCCESSFUL -- SAE-OLS is vulnerable to spoofing!")
elif avg_z > 2.0:
    print(f"\n  [~] Spoofer shows WEAK signal. May need more data or tuning.")
else:
    print(f"\n  [OK] Spoofer FAILED -- SAE-OLS resists spoofing attack.")
print(f"{'=' * 60}")