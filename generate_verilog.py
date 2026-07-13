#!/usr/bin/env python3
"""
Generate synthesizable SystemVerilog from the pruned NAS model.

Network (after BatchNorm folding):
  Conv1D(k=5, 2→16, ELU, same) → Conv1D(k=5, 16→32, ELU, same)
  → GlobalAvgPool1D → Dense(32→16, ELU) → Dense(16→3, argmax)

Fixed-point : Q8.8  (signed 16-bit, 8 fractional bits)
Accumulator : 40-bit signed

Output → verilog_output/
  nas_classifier_top.sv   synthesizable top-level
  *.hex                   quantized weight files
  README.md               usage guide
"""

import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

MODEL_PATH = "results_nas_v2_paper_pruning/nas_paper_model_pruned_55pct_1571weights.keras"
OUT_DIR    = "results_verilog"

DW    = 16
FRAC  = 12          # Q4.12 — 4 integer bits, 12 fractional bits (resolution ≈ 0.000244)
SCALE = 1 << FRAC
MAXV  = (1 << (DW - 1)) - 1
MINV  = -(1 << (DW - 1))
ACC_W = 40

SEQ   = 512
IN_CH = 2
K     = 5
C1F   = 16
C2F   = 32
D1U   = 16
NC    = 3
PAD   = K // 2          # 2
C1_MAC = K * IN_CH      # 10
C2_MAC = K * C1F        # 80


# ── helpers ───────────────────────────────────────────────────────────────────
def quantize(v):
    return int(np.clip(round(float(v) * SCALE), MINV, MAXV))

def quantize_arr(a):
    return np.vectorize(quantize)(a)

def fold_bn(W, b, gamma, beta, mean, var, eps=1e-3):
    s = gamma / np.sqrt(var + eps)
    return W * s, (b - mean) * s + beta

def write_hex(fname, flat, width=DW):
    nibbles = (width + 3) // 4
    mask    = (1 << width) - 1
    with open(os.path.join(OUT_DIR, fname), "w") as f:
        for v in flat:
            f.write(f"{int(v) & mask:0{nibbles}X}\n")


# ── load & process model (via h5py — no TensorFlow needed) ────────────────────
import zipfile, h5py, tempfile
print("Loading model weights via h5py...")

with zipfile.ZipFile(MODEL_PATH) as z:
    wdata = z.read("model.weights.h5")
_tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
_tmp.write(wdata); _tmp.close()

with h5py.File(_tmp.name) as h:
    def _w(path): return np.array(h[path])
    # Keras saves BN vars as [gamma, beta, moving_mean, moving_variance]
    conv1_W = _w("layers/conv1d/vars/0")        # (5, 2, 16)
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

os.unlink(_tmp.name)

conv1_W, conv1_b = fold_bn(conv1_W, conv1_b, bn1_g, bn1_b, bn1_m, bn1_v)
conv2_W, conv2_b = fold_bn(conv2_W, conv2_b, bn2_g, bn2_b, bn2_m, bn2_v)

C1W = quantize_arr(conv1_W)   # (5, 2, 16)
C1B = quantize_arr(conv1_b)   # (16,)
C2W = quantize_arr(conv2_W)   # (5, 16, 32)
C2B = quantize_arr(conv2_b)   # (32,)
D1W = quantize_arr(d1_W)      # (32, 16)
D1B = quantize_arr(d1_b)      # (16,)
D2W = quantize_arr(d2_W)      # (16, 3)
D2B = quantize_arr(d2_b)      # (3,)

print(f"Weight ranges (raw Q{DW-FRAC}.{FRAC} integers):")
for n, a in [("C1W",C1W),("C1B",C1B),("C2W",C2W),("C2B",C2B),
             ("D1W",D1W),("D1B",D1B),("D2W",D2W),("D2B",D2B)]:
    print(f"  {n:4s}: [{a.min():6d}, {a.max():6d}]  "
          f"({a.min()/SCALE:.4f}, {a.max()/SCALE:.4f})")

os.makedirs(OUT_DIR, exist_ok=True)
write_hex("c1w.hex", C1W.flatten())   # 160  entries  (k, cin, fout)
write_hex("c1b.hex", C1B.flatten())   # 16
write_hex("c2w.hex", C2W.flatten())   # 2560 entries  (k, cin, fout)
write_hex("c2b.hex", C2B.flatten())   # 32
write_hex("d1w.hex", D1W.flatten())   # 512  entries  (in, out)
write_hex("d1b.hex", D1B.flatten())   # 16
write_hex("d2w.hex", D2W.flatten())   # 48   entries  (in, out)
write_hex("d2b.hex", D2B.flatten())   # 3
print("Hex files written.\n")


# ── SystemVerilog generation ─────────────────────────────────────────────────
# Uses a plain string template (not f-string) to avoid conflicts between
# Python's {var} interpolation and Verilog's {a,b} concatenation syntax.
# Python variables are substituted via str.replace() at the end.
#
# Timing per inference (cycles):
#   RECV   : SEQ                   =    512
#   CONV1  : SEQ * (C1_MAC + 2)    =  6,144   (1 bias + 10 MACs + 1 ELU-write)
#   CONV2  : SEQ * (C2_MAC + 2)    = 42,496   (1 bias + 80 MACs + 1 ELU-write)
#   GAP    : SEQ + 1               =    513
#   DENSE1 : C2F + 1               =     33   (32 MACs + 1 ELU-write)
#   DENSE2 : D1U + 1               =     17   (16 MACs + 1 write)
#   ARGMAX : 1
#   Total  : ~49 k  →  ~0.49 ms @ 100 MHz
#
# Design notes:
#   - No `automatic` variables (Icarus Verilog compatible)
#   - Conv indices computed combinatorially (assign wires, not inside always)
#   - ELU write deferred one cycle after the last MAC (correct accumulation)
#   - BN folded into conv weights → zero BN hardware
#   - ELU approximation: x>=0→x,  -1≤x<0→x/2,  x<-1→-1.0
#   - Argmax: hardcoded for NC=3 (no loop)

sv = """// =============================================================================
// NAS Wireless Signal Classifier – Synthesizable SystemVerilog
//
// Conv1D(5,2->16,ELU) -> Conv1D(5,16->32,ELU) -> GAP -> Dense(32->16,ELU)
//                     -> Dense(16->3,argmax)
// Fixed-point Q8.8  |  Classes: 0=LTE  1=DVB-T  2=WiFi
// Compatible: Icarus Verilog / Vivado / Quartus
// =============================================================================
`timescale 1ns / 1ps

module nas_classifier_top #(
    parameter int DW    = __DW__,
    parameter int FRAC  = __FRAC__,
    parameter int ACC_W = __ACC_W__,
    parameter int SEQ   = __SEQ__,
    parameter int IN_CH = __IN_CH__,
    parameter int K     = __K__,
    parameter int C1F   = __C1F__,
    parameter int C2F   = __C2F__,
    parameter int D1U   = __D1U__,
    parameter int NC    = __NC__
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              start,
    input  logic              sample_valid,
    input  logic signed [DW-1:0] iq_real,
    input  logic signed [DW-1:0] iq_imag,
    output logic [1:0]        class_out,
    output logic              result_valid
);

    // ── Local parameters ──────────────────────────────────────────────────────
    localparam int PAD    = K / 2;       // 2
    localparam int C1_MAC = K * IN_CH;   // 10
    localparam int C2_MAC = K * C1F;     // 80
    localparam int KC1_END = C1_MAC + 1; // 11  (bias + 10 MACs + ELU-write)
    localparam int KC2_END = C2_MAC + 1; // 81

    // ── States ────────────────────────────────────────────────────────────────
    typedef enum logic [3:0] {
        S_IDLE, S_RECV, S_CONV1, S_CONV2,
        S_GAP, S_GAP_DIV, S_DENSE1, S_DENSE2, S_ARGMAX, S_DONE
    } state_t;
    state_t state;

    // ── Weight ROMs ───────────────────────────────────────────────────────────
    // Loaded from *.hex files at elaboration time via $readmemh.
    // Indexing: conv kernel[k, cin, fout] -> flat = k*IN_CH*C1F + cin*C1F + fout
    //           dense    W[in, out]       -> flat = in*D1U + out
    logic signed [DW-1:0] c1w [0:K*IN_CH*C1F-1];
    logic signed [DW-1:0] c1b [0:C1F-1];
    logic signed [DW-1:0] c2w [0:K*C1F*C2F-1];
    logic signed [DW-1:0] c2b [0:C2F-1];
    logic signed [DW-1:0] d1w [0:C2F*D1U-1];
    logic signed [DW-1:0] d1b [0:D1U-1];
    logic signed [DW-1:0] d2w [0:D1U*NC-1];
    logic signed [DW-1:0] d2b [0:NC-1];

    initial begin
        $readmemh("c1w.hex", c1w);  $readmemh("c1b.hex", c1b);
        $readmemh("c2w.hex", c2w);  $readmemh("c2b.hex", c2b);
        $readmemh("d1w.hex", d1w);  $readmemh("d1b.hex", d1b);
        $readmemh("d2w.hex", d2w);  $readmemh("d2b.hex", d2b);
    end

    // ── Feature map memories (synthesis tool infers BRAMs) ────────────────────
    logic signed [DW-1:0] input_mem [0:SEQ-1][0:IN_CH-1];
    logic signed [DW-1:0] c1_mem   [0:SEQ-1][0:C1F-1];
    logic signed [DW-1:0] c2_mem   [0:SEQ-1][0:C2F-1];

    // ── Accumulators and output registers ─────────────────────────────────────
    logic signed [ACC_W-1:0] c1_acc [0:C1F-1];
    logic signed [ACC_W-1:0] c2_acc [0:C2F-1];
    logic signed [ACC_W-1:0] gap_acc[0:C2F-1];
    logic signed [DW-1:0]    gap_out[0:C2F-1];
    logic signed [ACC_W-1:0] d1_acc [0:D1U-1];
    logic signed [DW-1:0]    d1_out [0:D1U-1];
    logic signed [ACC_W-1:0] d2_acc [0:NC-1];

    // ── Counters ──────────────────────────────────────────────────────────────
    logic [8:0]  t_cnt;    // time-step counter (0..511)
    logic [6:0]  kc_cnt;   // MAC counter       (0..81)

    // ── Combinatorial index wires for Conv1 ───────────────────────────────────
    // kc_cnt=1..C1_MAC maps kc=0..9  (k*IN_CH + ch)
    // IN_CH=2 is a power of 2: k = kc>>1,  ch = kc[0]
    logic [3:0]  c1_kc;
    logic [2:0]  c1_k;
    logic        c1_ch;
    logic signed [9:0] c1_ts;

    assign c1_kc = kc_cnt - 1;
    assign c1_k  = c1_kc[3:1];
    assign c1_ch = c1_kc[0];
    assign c1_ts = $signed({1'b0, t_cnt}) + $signed({7'b0, c1_k}) - $signed(10'(PAD));

    // ── Combinatorial index wires for Conv2 ───────────────────────────────────
    // C1F=16 is a power of 2: k = kc>>4,  ch = kc[3:0]
    logic [6:0]  c2_kc;
    logic [2:0]  c2_k;
    logic [3:0]  c2_ch;
    logic signed [9:0] c2_ts;

    assign c2_kc = kc_cnt - 1;
    assign c2_k  = c2_kc[6:4];
    assign c2_ch = c2_kc[3:0];
    assign c2_ts = $signed({1'b0, t_cnt}) + $signed({7'b0, c2_k}) - $signed(10'(PAD));

    // ── ELU activation function ───────────────────────────────────────────────
    // Input: 40-bit raw accumulator.  Output: Q8.8 16-bit result.
    // x >= 0         ->  x
    // -1.0 <= x < 0  ->  x >> 1   (linear fit of alpha*(e^x-1), alpha=1)
    // x < -1.0       ->  -1.0     (saturate)
    localparam logic signed [DW-1:0] ELU_MIN = __ELU_MIN__; // -1.0 in Q-format

    function automatic logic signed [DW-1:0] elu(
        input logic signed [ACC_W-1:0] acc
    );
        logic signed [DW-1:0] x;
        x = acc[FRAC +: DW];           // extract fixed-point result (shift right by FRAC)
        if      (x >= 0)        return x;
        else if (x >= ELU_MIN)  return x >>> 1;
        else                    return ELU_MIN;
    endfunction

    // ── Main state machine ─────────────────────────────────────────────────────
    integer f, n, kk;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            result_valid <= 1'b0;
            class_out    <= 2'd0;
            t_cnt        <= '0;
            kc_cnt       <= '0;
        end else begin
            result_valid <= 1'b0;
            case (state)

                // ── Idle ─────────────────────────────────────────────────
                S_IDLE: if (start) begin
                    state  <= S_RECV;
                    t_cnt  <= '0;
                end

                // ── Receive 512 IQ samples ────────────────────────────────
                S_RECV: if (sample_valid) begin
                    input_mem[t_cnt][0] <= iq_real;
                    input_mem[t_cnt][1] <= iq_imag;
                    if (t_cnt == SEQ - 1) begin
                        state  <= S_CONV1;
                        t_cnt  <= '0;
                        kc_cnt <= '0;
                    end else
                        t_cnt <= t_cnt + 1;
                end

                // ── Conv1: kernel(5, 2->16, ELU, same-padding) ────────────
                // kc_cnt=0       : load biases into accumulators
                // kc_cnt=1..10  : accumulate MAC (k, ch) for all 16 filters
                // kc_cnt=11     : write ELU output to c1_mem (acc is final here)
                S_CONV1: begin
                    case (kc_cnt)
                        0: begin
                            for (f=0; f<C1F; f=f+1)
                                c1_acc[f] <= {{(ACC_W-DW){c1b[f][DW-1]}}, c1b[f]} <<< FRAC;
                            kc_cnt <= kc_cnt + 1;
                        end
                        KC1_END: begin
                            for (f=0; f<C1F; f=f+1)
                                c1_mem[t_cnt][f] <= elu(c1_acc[f]);
                            kc_cnt <= '0;
                            if (t_cnt == SEQ - 1) begin
                                state <= S_CONV2;  t_cnt <= '0;
                            end else
                                t_cnt <= t_cnt + 1;
                        end
                        default: begin
                            if (c1_ts >= 0 && c1_ts < SEQ)
                                for (f=0; f<C1F; f=f+1)
                                    c1_acc[f] <= c1_acc[f]
                                        + {{(ACC_W-DW){input_mem[c1_ts][c1_ch][DW-1]},
                                           input_mem[c1_ts][c1_ch]}
                                        * {{(ACC_W-DW){c1w[c1_k*IN_CH*C1F + c1_ch*C1F + f][DW-1]}},
                                           c1w[c1_k*IN_CH*C1F + c1_ch*C1F + f]};
                            kc_cnt <= kc_cnt + 1;
                        end
                    endcase
                end

                // ── Conv2: kernel(5, 16->32, ELU, same-padding) ───────────
                S_CONV2: begin
                    case (kc_cnt)
                        0: begin
                            for (f=0; f<C2F; f=f+1)
                                c2_acc[f] <= {{(ACC_W-DW){c2b[f][DW-1]}}, c2b[f]} <<< FRAC;
                            kc_cnt <= kc_cnt + 1;
                        end
                        KC2_END: begin
                            for (f=0; f<C2F; f=f+1)
                                c2_mem[t_cnt][f] <= elu(c2_acc[f]);
                            kc_cnt <= '0;
                            if (t_cnt == SEQ - 1) begin
                                state  <= S_GAP;  t_cnt <= '0;
                                for (f=0; f<C2F; f=f+1)  gap_acc[f] <= '0;
                            end else
                                t_cnt <= t_cnt + 1;
                        end
                        default: begin
                            if (c2_ts >= 0 && c2_ts < SEQ)
                                for (f=0; f<C2F; f=f+1)
                                    c2_acc[f] <= c2_acc[f]
                                        + {{(ACC_W-DW){c1_mem[c2_ts][c2_ch][DW-1]}},
                                           c1_mem[c2_ts][c2_ch]}
                                        * {{(ACC_W-DW){c2w[c2_k*C1F*C2F + c2_ch*C2F + f][DW-1]}},
                                           c2w[c2_k*C1F*C2F + c2_ch*C2F + f]};
                            kc_cnt <= kc_cnt + 1;
                        end
                    endcase
                end

                // ── GlobalAveragePooling1D ────────────────────────────────
                S_GAP: begin
                    for (f=0; f<C2F; f=f+1)
                        gap_acc[f] <= gap_acc[f]
                                    + {{(ACC_W-DW){c2_mem[t_cnt][f][DW-1]}},
                                       c2_mem[t_cnt][f]};
                    if (t_cnt == SEQ - 1)
                        state <= S_GAP_DIV;
                    else
                        t_cnt <= t_cnt + 1;
                end

                // Divide by 512 (shift right 9), init Dense1 biases
                S_GAP_DIV: begin
                    for (f=0; f<C2F; f=f+1)
                        gap_out[f] <= DW'(gap_acc[f] >>> 9);
                    for (n=0; n<D1U; n=n+1)
                        d1_acc[n] <= {{(ACC_W-DW){d1b[n][DW-1]}}, d1b[n]} <<< FRAC;
                    state <= S_DENSE1;
                    t_cnt <= '0;
                end

                // ── Dense1: (32->16, ELU) ─────────────────────────────────
                // Iterate t_cnt over 32 inputs; all 16 outputs accumulate in parallel.
                S_DENSE1: begin
                    for (n=0; n<D1U; n=n+1)
                        d1_acc[n] <= d1_acc[n]
                                   + {{(ACC_W-DW){gap_out[t_cnt][DW-1]}}, gap_out[t_cnt]}
                                   * {{(ACC_W-DW){d1w[t_cnt*D1U + n][DW-1]}},
                                      d1w[t_cnt*D1U + n]};
                    if (t_cnt == C2F - 1) begin
                        for (n=0; n<D1U; n=n+1)
                            d1_out[n] <= elu(d1_acc[n]);
                        for (kk=0; kk<NC; kk=kk+1)
                            d2_acc[kk] <= {{(ACC_W-DW){d2b[kk][DW-1]}}, d2b[kk]} <<< FRAC;
                        state <= S_DENSE2;  t_cnt <= '0;
                    end else
                        t_cnt <= t_cnt + 1;
                end

                // ── Dense2: (16->3, no activation) ────────────────────────
                S_DENSE2: begin
                    for (kk=0; kk<NC; kk=kk+1)
                        d2_acc[kk] <= d2_acc[kk]
                                    + {{(ACC_W-DW){d1_out[t_cnt][DW-1]}}, d1_out[t_cnt]}
                                    * {{(ACC_W-DW){d2w[t_cnt*NC + kk][DW-1]}},
                                       d2w[t_cnt*NC + kk]};
                    if (t_cnt == D1U - 1)
                        state <= S_ARGMAX;
                    else
                        t_cnt <= t_cnt + 1;
                end

                // ── Argmax (unrolled for NC=3) ────────────────────────────
                S_ARGMAX: begin
                    if      (d2_acc[0] >= d2_acc[1] && d2_acc[0] >= d2_acc[2])
                        class_out <= 2'd0;
                    else if (d2_acc[1] >= d2_acc[2])
                        class_out <= 2'd1;
                    else
                        class_out <= 2'd2;
                    state <= S_DONE;
                end

                // ── Done: pulse result_valid ───────────────────────────────
                S_DONE: begin
                    result_valid <= 1'b1;
                    state        <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
"""

# Substitute Python parameters into the template
elu_min_raw = -(1 << FRAC)  # -1.0 in the chosen Q-format
elu_min_hex = f"-{DW}'sh{(-elu_min_raw):04X}"  # e.g. -16'sh1000 for Q4.12

sv = (sv
    .replace("__DW__",      str(DW))
    .replace("__FRAC__",    str(FRAC))
    .replace("__ACC_W__",   str(ACC_W))
    .replace("__SEQ__",     str(SEQ))
    .replace("__IN_CH__",   str(IN_CH))
    .replace("__K__",       str(K))
    .replace("__C1F__",     str(C1F))
    .replace("__C2F__",     str(C2F))
    .replace("__D1U__",     str(D1U))
    .replace("__NC__",      str(NC))
    .replace("__ELU_MIN__", elu_min_hex)
)


sv_path = os.path.join(OUT_DIR, "nas_classifier_top.sv")
with open(sv_path, "w") as f:
    f.write(sv)
print(f"SystemVerilog  →  {sv_path}")


# ── README ─────────────────────────────────────────────────────────────────────
readme = f"""\
# verilog_output

Synthesizable SystemVerilog generated from
`results_nas_v2_paper_pruning/nas_paper_model_pruned_55pct_1571weights.keras`

## Files

| File | Description |
|------|-------------|
| `nas_classifier_top.sv` | Synthesizable top-level module |
| `nas_classifier_tb.sv`  | Simulation testbench |
| `c1w.hex` / `c1b.hex`   | Conv1 weights / biases (Q8.8, hex) |
| `c2w.hex` / `c2b.hex`   | Conv2 weights / biases |
| `d1w.hex` / `d1b.hex`   | Dense1 weights / biases |
| `d2w.hex` / `d2b.hex`   | Dense2 weights / biases |
| `test_vectors.hex`       | 9 real IQ test vectors (3 per class) |
| `test_labels.hex`        | Expected labels (0=LTE 1=DVB-T 2=WiFi) |

## Fixed-point format

**Q8.8** — signed 16-bit, 8 fractional bits.
Real value = raw_integer / 256.
Accumulator: 40-bit signed.

## Network topology (BN folded into conv weights)

```
Input  →  Conv1D(k=5, 2→16, ELU, same)
       →  Conv1D(k=5, 16→32, ELU, same)
       →  GlobalAveragePooling1D  →  (32,)
       →  Dense(32→16, ELU)
       →  Dense(16→3, argmax)
       →  class_out [1:0]   (0=LTE  1=DVB-T  2=WiFi)
```

## Port list

| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| `clk` | in | 1 | System clock |
| `rst_n` | in | 1 | Active-low synchronous reset |
| `start` | in | 1 | Single-cycle pulse: begin inference |
| `sample_valid` | in | 1 | IQ sample present on inputs |
| `iq_real` | in | 16 | Q8.8 real part |
| `iq_imag` | in | 16 | Q8.8 imaginary part |
| `class_out` | out | 2 | Predicted class (valid with result_valid) |
| `result_valid` | out | 1 | Single-cycle pulse: inference complete |

## Cycle budget @ 100 MHz → ≈ 0.49 ms

| Stage | Cycles |
|-------|--------|
| RECV  | 512 |
| CONV1 | 512 × 12 = 6,144 |
| CONV2 | 512 × 82 = 41,984 |
| GAP   | 512 + 1 = 513 |
| DENSE1 | 32 + 1 = 33 |
| DENSE2 | 16 |
| ARGMAX | 1 |
| **Total** | **~49 k** |

## Run simulation (Icarus Verilog)

```bash
# Install (macOS)
brew install icarus-verilog

# Compile & simulate
cd verilog_output
iverilog -g2012 -o sim nas_classifier_top.sv nas_classifier_tb.sv
vvp sim
```

## Regenerate at any time

```bash
source venv/bin/activate
python generate_verilog.py        # regenerates .sv and .hex
python generate_test_vectors.py   # regenerates test vectors
```
"""

with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
    f.write(readme)
print(f"README         →  {OUT_DIR}/README.md")
print("\nDone.")
