#!/usr/bin/env python3
"""
Full test-set evaluation of the Q8.8 hardware model.
Loads all .bin files from split_dataset/test/, runs the same
Q8.8 arithmetic as nas_classifier_top.sv, and reports accuracy.
No TensorFlow required.
"""

import os, sys, glob, numpy as np

BASE   = os.path.join(os.path.dirname(__file__), "..")
TEST   = os.path.join(BASE, "split_dataset", "test")
CLASSES   = ["LTE", "DVB-T", "WiFi"]
PREFIXES  = ["lte", "dvbt", "wf"]

DW   = 16;  FRAC = 8;  SCALE = 1 << FRAC
SEQ  = 512; K = 5;     PAD   = K // 2
IN_CH = 2;  C1F = 16;  C2F = 32;  D1U = 16;  NC = 3

# ── helpers ──────────────────────────────────────────────────────────────────
def read_hex_s16(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                v = int(line, 16) & 0xFFFF
                vals.append(v if v < 0x8000 else v - 0x10000)
    return np.array(vals, dtype=np.int32)

def elu_q88(acc):
    x = int(acc >> FRAC) & 0xFFFF
    x = x if x < 0x8000 else x - 0x10000
    ELU_MIN = -SCALE
    if   x >= 0:        return x
    elif x >= ELU_MIN:  return x >> 1
    else:               return ELU_MIN

def quantize(v):
    return int(np.clip(round(float(v) * SCALE), -(1<<(DW-1)), (1<<(DW-1))-1))

# ── load weights ─────────────────────────────────────────────────────────────
WD = os.path.dirname(__file__)
c1w = read_hex_s16(f"{WD}/c1w.hex")
c1b = read_hex_s16(f"{WD}/c1b.hex")
c2w = read_hex_s16(f"{WD}/c2w.hex")
c2b = read_hex_s16(f"{WD}/c2b.hex")
d1w = read_hex_s16(f"{WD}/d1w.hex")
d1b = read_hex_s16(f"{WD}/d1b.hex")
d2w = read_hex_s16(f"{WD}/d2w.hex")
d2b = read_hex_s16(f"{WD}/d2b.hex")

# ── inference ─────────────────────────────────────────────────────────────────
def infer(iq_q88):
    """iq_q88: (SEQ, 2) int32 array of Q8.8 samples. Returns predicted class 0-2."""
    # Conv1
    c1_mem = np.zeros((SEQ, C1F), dtype=np.int32)
    for t in range(SEQ):
        acc = np.array([int(c1b[f]) << FRAC for f in range(C1F)], dtype=np.int64)
        for k in range(K):
            ts = t + k - PAD
            if 0 <= ts < SEQ:
                for ch in range(IN_CH):
                    inp = int(iq_q88[ts, ch])
                    for f in range(C1F):
                        acc[f] += inp * int(c1w[k*IN_CH*C1F + ch*C1F + f])
        for f in range(C1F):
            c1_mem[t, f] = elu_q88(int(acc[f]))
    # Conv2
    c2_mem = np.zeros((SEQ, C2F), dtype=np.int32)
    for t in range(SEQ):
        acc = np.array([int(c2b[f]) << FRAC for f in range(C2F)], dtype=np.int64)
        for k in range(K):
            ts = t + k - PAD
            if 0 <= ts < SEQ:
                for ch in range(C1F):
                    inp = int(c1_mem[ts, ch])
                    for f in range(C2F):
                        acc[f] += inp * int(c2w[k*C1F*C2F + ch*C2F + f])
        for f in range(C2F):
            c2_mem[t, f] = elu_q88(int(acc[f]))
    # GAP
    gap_acc = np.zeros(C2F, dtype=np.int64)
    for t in range(SEQ):
        gap_acc += c2_mem[t, :]
    gap_out = (gap_acc >> 9).astype(np.int32)
    # Dense1
    d1_acc = np.array([int(d1b[n]) << FRAC for n in range(D1U)], dtype=np.int64)
    for i in range(C2F):
        for n in range(D1U):
            d1_acc[n] += int(gap_out[i]) * int(d1w[i*D1U + n])
    d1_out = np.array([elu_q88(int(d1_acc[n])) for n in range(D1U)], dtype=np.int32)
    # Dense2
    d2_acc = np.array([int(d2b[k]) << FRAC for k in range(NC)], dtype=np.int64)
    for i in range(D1U):
        for k in range(NC):
            d2_acc[k] += int(d1_out[i]) * int(d2w[i*NC + k])
    return int(np.argmax(d2_acc))

# ── load test set ─────────────────────────────────────────────────────────────
def read_iq_file(path):
    data = np.fromfile(path, dtype=np.float32)
    iq = data[0::2] + 1j * data[1::2]
    iq = (iq - np.mean(iq)) / np.std(iq)
    return iq

correct = {c: 0 for c in CLASSES}
total   = {c: 0 for c in CLASSES}
n_total = 0

print("Evaluating Q8.8 hardware model on full test set...")
print("(This may take a few minutes — pure Python loops)\n")

for cls_idx, (cls_name, prefix) in enumerate(zip(CLASSES, PREFIXES)):
    files = sorted(glob.glob(os.path.join(TEST, f"{prefix}*.bin")))
    for fpath in files:
        iq = read_iq_file(fpath)
        n_chunks = len(iq) // (1024)          # 1024-sample windows (decimated to 512)
        for i in range(n_chunks):
            chunk = iq[i*1024 : i*1024 + 1024]
            chunk = chunk[::2]                 # decimate 1024→512
            iq_re = np.array([quantize(v.real) for v in chunk], dtype=np.int32)
            iq_im = np.array([quantize(v.imag) for v in chunk], dtype=np.int32)
            iq_q88 = np.stack([iq_re, iq_im], axis=1)  # (512, 2)
            pred = infer(iq_q88)
            total[cls_name]   += 1
            n_total           += 1
            if pred == cls_idx:
                correct[cls_name] += 1
        print(f"  {cls_name}: {fpath.split('/')[-1]}  "
              f"({correct[cls_name]}/{total[cls_name]} so far)", flush=True)

print("\n" + "="*50)
print("  Per-class accuracy:")
for cls_name in CLASSES:
    acc = correct[cls_name] / total[cls_name] * 100 if total[cls_name] > 0 else 0
    print(f"    {cls_name:<8}: {correct[cls_name]:>5}/{total[cls_name]:<5}  ({acc:.1f}%)")
overall = sum(correct.values()) / n_total * 100
print(f"\n  Overall Q8.8 hardware accuracy: {sum(correct.values())}/{n_total} = {overall:.1f}%")
print("="*50)
