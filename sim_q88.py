#!/usr/bin/env python3
"""
Pure-numpy Q4.12 reference simulation.
Mirrors the exact hardware arithmetic in nas_classifier_top.sv so we can
compare expected vs. actual Icarus Verilog output without TensorFlow.
"""

import numpy as np

DW   = 16
FRAC = 12
SCALE = 1 << FRAC     # 4096
SEQ  = 512
K    = 5
PAD  = K // 2         # 2
IN_CH = 2
C1F  = 16
C2F  = 32
D1U  = 16
NC   = 3
N_SAMPLES = 9
CLASSES = ["LTE  ", "DVB-T", "WiFi "]

MINV = -(1 << (DW-1))
MAXV =  (1 << (DW-1)) - 1

def to_s16(v):
    """Interpret a numpy int as signed 16-bit."""
    v = int(v) & 0xFFFF
    return v if v < 0x8000 else v - 0x10000

def read_hex_s16(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                v = int(line, 16) & 0xFFFF
                vals.append(v if v < 0x8000 else v - 0x10000)
    return np.array(vals, dtype=np.int32)

def elu_q88(acc_int):
    """ELU on 40-bit accumulator integer → 16-bit fixed-point result."""
    x = (acc_int >> FRAC) & 0xFFFF
    x = x if x < 0x8000 else x - 0x10000   # to signed
    ELU_MIN = -SCALE   # -1.0 in the chosen Q-format
    if   x >= 0:        return x
    elif x >= ELU_MIN:  return x >> 1
    else:               return ELU_MIN

import os as _os
_WD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results_verilog")
def _p(name): return _os.path.join(_WD, name)

# ── Load weights ──────────────────────────────────────────────────────────────
c1w = read_hex_s16(_p("c1w.hex"))   # K*IN_CH*C1F = 160
c1b = read_hex_s16(_p("c1b.hex"))   # C1F = 16
c2w = read_hex_s16(_p("c2w.hex"))   # K*C1F*C2F = 2560
c2b = read_hex_s16(_p("c2b.hex"))   # C2F = 32
d1w = read_hex_s16(_p("d1w.hex"))   # C2F*D1U = 512
d1b = read_hex_s16(_p("d1b.hex"))   # D1U = 16
d2w = read_hex_s16(_p("d2w.hex"))   # D1U*NC = 48
d2b = read_hex_s16(_p("d2b.hex"))   # NC = 3

# ── Load test vectors ─────────────────────────────────────────────────────────
words = []
with open(_p("test_vectors.hex")) as f:
    for line in f:
        words.append(int(line.strip(), 16))

labels = []
with open(_p("test_labels.txt")) as f:
    for line in f:
        labels.append(int(line.strip()))

print("=" * 62)
print(f"  NAS Q{DW-FRAC}.{FRAC} Reference (numpy) vs Hardware (Icarus Verilog)")
print("=" * 62)
print(f"  {'#':>2}  {'Expected':<10}  {'HW result':<10}  Result")
print(f"  {'-':>2}  {'--------':<10}  {'---------':<10}  ------")

hw_results = [2,2,2, 0,2,2, 2,2,2]   # from Icarus simulation

pass_hw = 0
pass_np = 0
for s in range(N_SAMPLES):
    # Unpack IQ samples for this test sample
    sample = []
    for t in range(SEQ):
        w = words[s * SEQ + t]
        r  = w >> 16
        im = w & 0xFFFF
        r  = r  if r  < 0x8000 else r  - 0x10000
        im = im if im < 0x8000 else im - 0x10000
        sample.append((r, im))

    # ── Conv1 ────────────────────────────────────────────────────────────────
    c1_mem = np.zeros((SEQ, C1F), dtype=np.int32)
    for t in range(SEQ):
        acc = np.array([int(c1b[f]) << FRAC for f in range(C1F)], dtype=np.int64)
        for k in range(K):
            ts = t + k - PAD
            for ch in range(IN_CH):
                if 0 <= ts < SEQ:
                    inp = sample[ts][ch]
                    for f in range(C1F):
                        w_idx = k * IN_CH * C1F + ch * C1F + f
                        acc[f] += inp * int(c1w[w_idx])
        for f in range(C1F):
            c1_mem[t, f] = elu_q88(int(acc[f]))

    # ── Conv2 ────────────────────────────────────────────────────────────────
    c2_mem = np.zeros((SEQ, C2F), dtype=np.int32)
    for t in range(SEQ):
        acc = np.array([int(c2b[f]) << FRAC for f in range(C2F)], dtype=np.int64)
        for k in range(K):
            ts = t + k - PAD
            for ch in range(C1F):
                if 0 <= ts < SEQ:
                    inp = c1_mem[ts, ch]
                    for f in range(C2F):
                        w_idx = k * C1F * C2F + ch * C2F + f
                        acc[f] += inp * int(c2w[w_idx])
        for f in range(C2F):
            c2_mem[t, f] = elu_q88(int(acc[f]))

    # ── GAP ──────────────────────────────────────────────────────────────────
    gap_acc = np.zeros(C2F, dtype=np.int64)
    for t in range(SEQ):
        gap_acc += c2_mem[t, :]
    gap_out = (gap_acc >> 9).astype(np.int32)   # /512, preserve fixed-point scaling

    # ── Dense1 ───────────────────────────────────────────────────────────────
    d1_acc = np.array([int(d1b[n]) << FRAC for n in range(D1U)], dtype=np.int64)
    for i in range(C2F):
        for n in range(D1U):
            d1_acc[n] += int(gap_out[i]) * int(d1w[i * D1U + n])
    d1_out = np.array([elu_q88(int(d1_acc[n])) for n in range(D1U)], dtype=np.int32)

    # ── Dense2 ───────────────────────────────────────────────────────────────
    d2_acc = np.array([int(d2b[k]) << FRAC for k in range(NC)], dtype=np.int64)
    for i in range(D1U):
        for k in range(NC):
            d2_acc[k] += int(d1_out[i]) * int(d2w[i * NC + k])

    pred_np = int(np.argmax(d2_acc))
    pred_hw = hw_results[s]
    expected = labels[s]

    ok_np = "PASS" if pred_np == expected else "FAIL"
    ok_hw = "PASS" if pred_hw == expected else "FAIL"
    if ok_np == "PASS": pass_np += 1
    if ok_hw == "PASS": pass_hw += 1

    print(f"  {s:>2}  {CLASSES[expected]:<10}  {CLASSES[pred_hw]:<10}  "
          f"HW:{ok_hw}  NP:{ok_np}  "
          f"scores=[{d2_acc[0]:+10d}, {d2_acc[1]:+10d}, {d2_acc[2]:+10d}]  np_pred={CLASSES[pred_np]}")

print("=" * 62)
print(f"  Hardware: {pass_hw}/9    Numpy Q{DW-FRAC}.{FRAC}: {pass_np}/9")
print("=" * 62)
