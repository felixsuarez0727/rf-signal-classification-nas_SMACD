#!/usr/bin/env python3
"""
Extract test vectors from the real dataset and quantize to Q8.8 for simulation.

Outputs (in verilog_output/):
  test_vectors.hex   – IQ samples: one 32-bit word per sample (real[31:16] | imag[15:0])
  test_labels.txt    – expected class index (0=LTE, 1=DVB-T, 2=WiFi), one per sample
  test_info.txt      – human-readable summary
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

OUT_DIR   = "verilog_output"
DW        = 16
FRAC      = 8
SCALE     = 1 << FRAC
MAXV      = (1 << (DW - 1)) - 1
MINV      = -(1 << (DW - 1))
SEQ       = 512
N_SAMPLES = 9   # 3 samples per class

CLASSES = ["LTE", "DVB-T", "WiFi"]


def quantize(v: float) -> int:
    return int(np.clip(round(float(v) * SCALE), MINV, MAXV))


sys.path.insert(0, ".")
import tensorflow as tf
from train import load_dataset, CLASSES as TRAIN_CLASSES

print("Loading test dataset...")
X_test, y_test = load_dataset(os.path.join("split_dataset", "test"), chunk_samples=1024)
y_labels = np.argmax(y_test, axis=1)

# Decimate 1024→512 (same as training preprocessing)
step = 1024 // SEQ
X_test = X_test[:, ::step, :]

# Pick 3 samples per class
rng = np.random.default_rng(42)
selected_X, selected_y = [], []
for cls in range(len(CLASSES)):
    idx = np.where(y_labels == cls)[0]
    chosen = rng.choice(idx, 3, replace=False)
    selected_X.extend(X_test[chosen])
    selected_y.extend([cls] * 3)

selected_X = np.array(selected_X)   # (9, 512, 2)
selected_y = np.array(selected_y)   # (9,)

os.makedirs(OUT_DIR, exist_ok=True)

# Write test_vectors.hex
# Format per line: one 32-bit word = {real[15:0], imag[15:0]}
# 512 lines per test sample, N_SAMPLES samples total
vec_path = os.path.join(OUT_DIR, "test_vectors.hex")
with open(vec_path, "w") as f:
    for s in range(N_SAMPLES):
        for t in range(SEQ):
            r = quantize(selected_X[s, t, 0])
            im = quantize(selected_X[s, t, 1])
            # Pack as 32-bit: upper 16 = real, lower 16 = imag
            word = ((r & 0xFFFF) << 16) | (im & 0xFFFF)
            f.write(f"{word:08X}\n")

# Write expected labels
lbl_path = os.path.join(OUT_DIR, "test_labels.txt")
with open(lbl_path, "w") as f:
    for y in selected_y:
        f.write(f"{y}\n")

# Human-readable summary
info_path = os.path.join(OUT_DIR, "test_info.txt")
with open(info_path, "w") as f:
    f.write(f"Test vectors: {N_SAMPLES} samples ({N_SAMPLES//len(CLASSES)} per class)\n")
    f.write(f"Sequence length: {SEQ} IQ samples per inference\n")
    f.write(f"Fixed-point: Q8.8 (signed {DW}-bit, {FRAC} fractional bits)\n\n")
    f.write("Index | Class  | Label\n")
    f.write("------+--------+------\n")
    for i, y in enumerate(selected_y):
        f.write(f"  {i:3d} | {CLASSES[y]:6s} |   {y}\n")

print(f"Written {N_SAMPLES} test vectors ({N_SAMPLES//3} per class) → {vec_path}")
print(f"Labels → {lbl_path}")
print(f"Info   → {info_path}")

for i, (x, y) in enumerate(zip(selected_X, selected_y)):
    print(f"  Sample {i}: {CLASSES[y]:6s}  "
          f"real∈[{x[:,0].min():.3f},{x[:,0].max():.3f}]  "
          f"imag∈[{x[:,1].min():.3f},{x[:,1].max():.3f}]")
