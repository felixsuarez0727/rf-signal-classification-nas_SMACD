#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) for the NAS classifier.

Loads float32 weights from the pruned Keras model, rebuilds the network
in PyTorch with BN already folded, then trains with fake-quantization
(straight-through estimator) at Q4.12 precision.

After training, exports weights to results_verilog/*.hex
"""

import os, glob, zipfile, tempfile, h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "results_nas_v2_paper_pruning/nas_paper_model_pruned_55pct_1571weights.keras"
OUT_DIR    = "results_verilog"
TRAIN_DIR  = "split_dataset/train"
VAL_DIR    = "split_dataset/validation"

DW    = 16
FRAC  = 12
SCALE = 1 << FRAC          # 4096
MAXV  = (1 << (DW-1)) - 1  # 32767
MINV  = -(1 << (DW-1))     # -32768

SEQ   = 512
K     = 5
PAD   = K // 2
IN_CH = 2
C1F   = 16
C2F   = 32
D1U   = 16
NC    = 3

CLASSES  = ["LTE", "DVB-T", "WiFi"]
PREFIXES = ["lte",  "dvbt",  "wf"]

EPOCHS     = 10
BATCH_SIZE = 64
LR         = 1e-4

DEVICE = "cpu"   # MPS can be used if available
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
print(f"Device: {DEVICE}")


# ── Fake-quantization (STE) ───────────────────────────────────────────────────
class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, minv, maxv):
        return (x * scale).clamp(minv, maxv).round() / scale

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None   # straight-through

def fq(x):
    return FakeQuant.apply(x, float(SCALE), float(MINV), float(MAXV))


# ── ELU approximation matching hardware ──────────────────────────────────────
def elu_hw(x):
    """Hardware ELU: x≥0→x, -1≤x<0→x/2, x<-1→-1."""
    pos   = x.clamp(min=0)
    neg1  = (x / 2).clamp(min=-1.0, max=0)
    neg2  = torch.full_like(x, -1.0)
    out   = torch.where(x >= 0, pos,
            torch.where(x >= -1.0, neg1, neg2))
    return out


# ── BN folding helper ─────────────────────────────────────────────────────────
def fold_bn(W, b, gamma, beta, mean, var, eps=1e-3):
    s = gamma / np.sqrt(var + eps)
    return W * s, (b - mean) * s + beta


# ── Load weights from .keras ──────────────────────────────────────────────────
def load_keras_weights():
    with zipfile.ZipFile(MODEL_PATH) as z:
        wdata = z.read("model.weights.h5")
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.write(wdata); tmp.close()

    with h5py.File(tmp.name) as h:
        def _w(p): return np.array(h[p], dtype=np.float32)
        conv1_W = _w("layers/conv1d/vars/0")       # (5, 2, 16)
        conv1_b = _w("layers/conv1d/vars/1")        # (16,)
        bn1_g   = _w("layers/batch_normalization/vars/0")
        bn1_b   = _w("layers/batch_normalization/vars/1")
        bn1_m   = _w("layers/batch_normalization/vars/2")
        bn1_v   = _w("layers/batch_normalization/vars/3")
        conv2_W = _w("layers/conv1d_1/vars/0")      # (5, 16, 32)
        conv2_b = _w("layers/conv1d_1/vars/1")      # (32,)
        bn2_g   = _w("layers/batch_normalization_1/vars/0")
        bn2_b   = _w("layers/batch_normalization_1/vars/1")
        bn2_m   = _w("layers/batch_normalization_1/vars/2")
        bn2_v   = _w("layers/batch_normalization_1/vars/3")
        d1_W    = _w("layers/dense/vars/0")         # (32, 16)
        d1_b    = _w("layers/dense/vars/1")         # (16,)
        d2_W    = _w("layers/dense_1/vars/0")       # (16, 3)
        d2_b    = _w("layers/dense_1/vars/1")       # (3,)
    os.unlink(tmp.name)

    c1W, c1b = fold_bn(conv1_W, conv1_b, bn1_g, bn1_b, bn1_m, bn1_v)
    c2W, c2b = fold_bn(conv2_W, conv2_b, bn2_g, bn2_b, bn2_m, bn2_v)

    return c1W, c1b, c2W, c2b, d1_W, d1_b, d2_W, d2_b


# ── QAT Model ────────────────────────────────────────────────────────────────
class NASClassifierQAT(nn.Module):
    def __init__(self, c1W, c1b, c2W, c2b, d1W, d1b, d2W, d2b):
        super().__init__()
        # Keras Conv1D weight shape: (K, in_ch, out_ch)
        # PyTorch Conv1d weight shape: (out_ch, in_ch, K)
        self.conv1 = nn.Conv1d(IN_CH, C1F, K, padding=PAD, bias=True)
        self.conv2 = nn.Conv1d(C1F,  C2F, K, padding=PAD, bias=True)
        self.fc1   = nn.Linear(C2F, D1U, bias=True)
        self.fc2   = nn.Linear(D1U, NC,  bias=True)

        with torch.no_grad():
            # Keras: (K, in, out) → PyTorch: (out, in, K)
            self.conv1.weight.copy_(torch.from_numpy(c1W.transpose(2, 1, 0)))
            self.conv1.bias.copy_(torch.from_numpy(c1b))
            self.conv2.weight.copy_(torch.from_numpy(c2W.transpose(2, 1, 0)))
            self.conv2.bias.copy_(torch.from_numpy(c2b))
            # Keras Dense: (in, out) → PyTorch Linear: (out, in)
            self.fc1.weight.copy_(torch.from_numpy(d1W.T))
            self.fc1.bias.copy_(torch.from_numpy(d1b))
            self.fc2.weight.copy_(torch.from_numpy(d2W.T))
            self.fc2.bias.copy_(torch.from_numpy(d2b))

    def forward(self, x, quantize=True):
        # x: (B, SEQ, 2) → (B, 2, SEQ)
        x = x.permute(0, 2, 1)
        if quantize: x = fq(x)

        # Conv1
        w1 = fq(self.conv1.weight) if quantize else self.conv1.weight
        b1 = fq(self.conv1.bias)   if quantize else self.conv1.bias
        x = F.conv1d(x, w1, b1, padding=PAD)
        if quantize: x = fq(x)
        x = elu_hw(x)
        if quantize: x = fq(x)

        # Conv2
        w2 = fq(self.conv2.weight) if quantize else self.conv2.weight
        b2 = fq(self.conv2.bias)   if quantize else self.conv2.bias
        x = F.conv1d(x, w2, b2, padding=PAD)
        if quantize: x = fq(x)
        x = elu_hw(x)
        if quantize: x = fq(x)

        # GAP: mean over time dim → (B, C2F)
        x = x.mean(dim=2)
        if quantize: x = fq(x)

        # Dense1
        w3 = fq(self.fc1.weight) if quantize else self.fc1.weight
        b3 = fq(self.fc1.bias)   if quantize else self.fc1.bias
        x = F.linear(x, w3, b3)
        if quantize: x = fq(x)
        x = elu_hw(x)
        if quantize: x = fq(x)

        # Dense2
        w4 = fq(self.fc2.weight) if quantize else self.fc2.weight
        b4 = fq(self.fc2.bias)   if quantize else self.fc2.bias
        x = F.linear(x, w4, b4)

        return x


# ── Dataset ───────────────────────────────────────────────────────────────────
def read_iq(path):
    data = np.fromfile(path, dtype=np.float32)
    return data[0::2] + 1j * data[1::2]

def normalize(iq):
    return (iq - iq.mean()) / (iq.std() + 1e-8)

class IQDataset(Dataset):
    def __init__(self, folder):
        self.samples, self.labels = [], []
        for cls_idx, (cls_name, prefix) in enumerate(zip(CLASSES, PREFIXES)):
            for fpath in sorted(glob.glob(os.path.join(folder, f"{prefix}*.bin"))):
                iq = normalize(read_iq(fpath))
                for i in range(len(iq) // 1024):
                    chunk = iq[i*1024 : i*1024 + 1024][::2]   # decimate 1024→512
                    x = np.stack([chunk.real, chunk.imag], axis=1).astype(np.float32)
                    self.samples.append(x)
                    self.labels.append(cls_idx)
        self.samples = np.array(self.samples)   # (N, 512, 2)
        self.labels  = np.array(self.labels)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return torch.from_numpy(self.samples[i]), int(self.labels[i])


# ── Weight export ─────────────────────────────────────────────────────────────
def quantize_arr(a):
    return np.clip(np.round(a * SCALE), MINV, MAXV).astype(np.int32)

def write_hex(fname, flat):
    mask = (1 << DW) - 1
    with open(os.path.join(OUT_DIR, fname), "w") as f:
        for v in flat:
            f.write(f"{int(v) & mask:04X}\n")

def export_weights(model):
    with torch.no_grad():
        c1W = model.conv1.weight.cpu().numpy()   # (C1F, IN_CH, K)
        c1b = model.conv1.bias.cpu().numpy()     # (C1F,)
        c2W = model.conv2.weight.cpu().numpy()   # (C2F, C1F, K)
        c2b = model.conv2.bias.cpu().numpy()     # (C2F,)
        d1W = model.fc1.weight.cpu().numpy()     # (D1U, C2F)
        d1b = model.fc1.bias.cpu().numpy()       # (D1U,)
        d2W = model.fc2.weight.cpu().numpy()     # (NC, D1U)
        d2b = model.fc2.bias.cpu().numpy()       # (NC,)

    # Convert back to Keras layout for hex export
    # Conv: (out, in, K) → (K, in, out)
    c1W_k = c1W.transpose(2, 1, 0)   # (K, IN_CH, C1F)
    c2W_k = c2W.transpose(2, 1, 0)   # (K, C1F, C2F)
    # Dense: (out, in) → (in, out)
    d1W_k = d1W.T                    # (C2F, D1U)
    d2W_k = d2W.T                    # (D1U, NC)

    C1W = quantize_arr(c1W_k); C1B = quantize_arr(c1b)
    C2W = quantize_arr(c2W_k); C2B = quantize_arr(c2b)
    D1W = quantize_arr(d1W_k); D1B = quantize_arr(d1b)
    D2W = quantize_arr(d2W_k); D2B = quantize_arr(d2b)

    write_hex("c1w.hex", C1W.flatten())
    write_hex("c1b.hex", C1B.flatten())
    write_hex("c2w.hex", C2W.flatten())
    write_hex("c2b.hex", C2B.flatten())
    write_hex("d1w.hex", D1W.flatten())
    write_hex("d1b.hex", D1B.flatten())
    write_hex("d2w.hex", D2W.flatten())
    write_hex("d2b.hex", D2B.flatten())
    print("Hex files exported.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading float32 weights...")
    c1W, c1b, c2W, c2b, d1W, d1b, d2W, d2b = load_keras_weights()

    print("Building QAT model...")
    model = NASClassifierQAT(c1W, c1b, c2W, c2b, d1W, d1b, d2W, d2b).to(DEVICE)

    # Quick sanity check: float accuracy before QAT
    print("\nLoading validation set...")
    val_ds  = IQDataset(VAL_DIR)
    val_dl  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb, quantize=False).argmax(1)
            correct += (pred == yb).sum().item()
    print(f"Float32 val accuracy (pre-QAT): {correct/len(val_ds)*100:.1f}%")

    correct = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb, quantize=True).argmax(1)
            correct += (pred == yb).sum().item()
    print(f"Q4.12 val accuracy  (pre-QAT): {correct/len(val_ds)*100:.1f}%")

    print("\nLoading training set (may take a moment)...")
    train_ds = IQDataset(TRAIN_DIR)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\nStarting QAT — {EPOCHS} epochs, lr={LR}\n")
    best_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb, quantize=True)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb, quantize=True).argmax(1)
                correct += (pred == yb).sum().item()
        val_acc = correct / len(val_ds) * 100
        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest val accuracy: {best_acc:.1f}%")
    model.load_state_dict(best_state)

    print("Exporting QAT weights to hex files...")
    export_weights(model)
    print(f"\nDone. Re-run eval_q88_full.py to measure test-set accuracy.")
