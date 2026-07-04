#!/usr/bin/env python3
"""
Generate synthesizable SystemVerilog from the pruned NAS model.

Network (after BatchNorm folding):
  Conv1D(k=5, 2→16, ELU, same) → Conv1D(k=5, 16→32, ELU, same)
  → GlobalAvgPool1D → Dense(32→16, ELU) → Dense(16→3, argmax)

Fixed-point format: Q8.8  (signed 16-bit, 8 fractional bits)
Accumulator width : 40-bit signed (prevents overflow during MAC chains)

Output directory: verilog_output/
  nas_classifier_top.sv   – synthesizable top-level module
  c1w.hex / c1b.hex       – Conv1 weights / biases
  c2w.hex / c2b.hex       – Conv2 weights / biases
  d1w.hex / d1b.hex       – Dense1 weights / biases
  d2w.hex / d2b.hex       – Dense2 weights / biases
  README.md               – usage notes
"""

import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

MODEL_PATH = "results_nas_v2_paper_pruning/nas_paper_model_pruned_55pct_1571weights.keras"
OUT_DIR    = "verilog_output"

# ── Fixed-point configuration ─────────────────────────────────────────────────
DW    = 16          # data / weight width (bits)
FRAC  = 8           # fractional bits
SCALE = 1 << FRAC   # 256
MAXV  = (1 << (DW - 1)) - 1   #  32767
MINV  = -(1 << (DW - 1))      # -32768
ACC_W = 40          # accumulator width

# Network constants
SEQ   = 512   # input sequence length
IN_CH = 2     # IQ channels (real, imag)
K     = 5     # conv kernel size
C1F   = 16    # conv1 filters
C2F   = 32    # conv2 filters
D1U   = 16    # dense1 units
NC    = 3     # number of classes


# ── Helpers ───────────────────────────────────────────────────────────────────
def quantize(v: float) -> int:
    return int(np.clip(round(v * SCALE), MINV, MAXV))

def quantize_arr(a: np.ndarray) -> np.ndarray:
    return np.vectorize(quantize)(a)

def fold_bn(W, b, gamma, beta, mean, var, eps=1e-3):
    s = gamma / np.sqrt(var + eps)
    return W * s, (b - mean) * s + beta

def write_hex(fname: str, flat, width: int = DW) -> None:
    nibbles = (width + 3) // 4
    mask    = (1 << width) - 1
    path    = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v) & mask:0{nibbles}X}\n")


# ── Load and process model ────────────────────────────────────────────────────
print("Loading model...")
try:
    import tensorflow as tf
except ImportError:
    sys.exit("TensorFlow not found. Activate the venv first: source venv/bin/activate")

model = tf.keras.models.load_model(MODEL_PATH)
L = model.layers

conv1_W, conv1_b             = L[0].get_weights()   # (5, 2, 16), (16,)
bn1_g,   bn1_b, bn1_m, bn1_v = L[1].get_weights()
conv2_W, conv2_b             = L[3].get_weights()   # (5, 16, 32), (32,)
bn2_g,   bn2_b, bn2_m, bn2_v = L[4].get_weights()
d1_W,    d1_b               = L[7].get_weights()    # (32, 16), (16,)
d2_W,    d2_b               = L[9].get_weights()    # (16, 3), (3,)

# Fold BatchNorm into conv weights (eliminates BN hardware entirely)
conv1_W, conv1_b = fold_bn(conv1_W, conv1_b, bn1_g, bn1_b, bn1_m, bn1_v)
conv2_W, conv2_b = fold_bn(conv2_W, conv2_b, bn2_g, bn2_b, bn2_m, bn2_v)

# Quantize
C1W = quantize_arr(conv1_W)   # (5, 2, 16)   kernel[k, cin, fout]
C1B = quantize_arr(conv1_b)   # (16,)
C2W = quantize_arr(conv2_W)   # (5, 16, 32)
C2B = quantize_arr(conv2_b)   # (32,)
D1W = quantize_arr(d1_W)      # (32, 16)     W[in, out]
D1B = quantize_arr(d1_b)      # (16,)
D2W = quantize_arr(d2_W)      # (16, 3)
D2B = quantize_arr(d2_b)      # (3,)

print("Weight ranges after quantization (raw integer values):")
for name, arr in [("C1W", C1W), ("C1B", C1B), ("C2W", C2W), ("C2B", C2B),
                  ("D1W", D1W), ("D1B", D1B), ("D2W", D2W), ("D2B", D2B)]:
    print(f"  {name:4s}: [{arr.min():6d}, {arr.max():6d}]  "
          f"({arr.min()/SCALE:.4f}, {arr.max()/SCALE:.4f}) float")

os.makedirs(OUT_DIR, exist_ok=True)

# Write hex files (one word per line, MSB first)
# Indexing convention: weight[k, cin, fout] → flat index k*IN_CH*C1F + cin*C1F + fout
write_hex("c1w.hex", C1W.reshape(-1))   # 160 entries
write_hex("c1b.hex", C1B.reshape(-1))   # 16
write_hex("c2w.hex", C2W.reshape(-1))   # 2560
write_hex("c2b.hex", C2B.reshape(-1))   # 32
write_hex("d1w.hex", D1W.reshape(-1))   # 512 (row-major: W[in, out])
write_hex("d1b.hex", D1B.reshape(-1))   # 16
write_hex("d2w.hex", D2W.reshape(-1))   # 48
write_hex("d2b.hex", D2B.reshape(-1))   # 3

print("\nHex weight files written.")


# ── Generate SystemVerilog ────────────────────────────────────────────────────
# Timing (cycles per inference at 100 MHz):
#   RECV  :   512  cycles  (receive IQ samples)
#   CONV1 :   512 × (1 + 5×2) = 5,632  cycles  (11 cycles/t: 1 bias + 10 MACs)
#   CONV2 :   512 × (1 + 5×16) = 41,472 cycles  (81 cycles/t: 1 bias + 80 MACs)
#   GAP   :   512  cycles  (all C2F in parallel)
#   DENSE1:    32  cycles  (all D1U in parallel)
#   DENSE2:    16  cycles  (all NC in parallel)
#   ARGMAX:     1  cycle
# Total : ~48k cycles → ~0.48 ms @ 100 MHz

sv = f"""\
// =============================================================================
// NAS Wireless Signal Classifier – Synthesizable SystemVerilog
//
// Architecture : Conv1D(5,2→16,ELU) → Conv1D(5,16→32,ELU)
//                → GlobalAvgPool1D → Dense(32→16,ELU) → Dense(16→3,argmax)
// Fixed-point  : Q8.8  (signed {DW}-bit, {FRAC} fractional bits)
// Classes      : 0=LTE  1=DVB-T  2=WiFi
//
// Interfaces
//   clk          : system clock
//   rst_n        : active-low synchronous reset
//   start        : single-cycle pulse to begin a new inference
//   sample_valid : asserted while IQ samples are being streamed in
//   iq_real/iq_imag : one {DW}-bit Q8.8 sample per clock (512 total)
//   class_out    : 2-bit result (valid when result_valid is high)
//   result_valid : single-cycle pulse when inference is complete
//
// Weight ROMs are loaded via $readmemh at elaboration time.
// Place the *.hex files in the simulation/synthesis working directory.
// =============================================================================

`timescale 1ns / 1ps

module nas_classifier_top #(
    parameter int DW    = {DW},    // data / weight width
    parameter int FRAC  = {FRAC},    // fractional bits
    parameter int ACC_W = {ACC_W},   // accumulator width
    parameter int SEQ   = {SEQ},  // input sequence length
    parameter int IN_CH = {IN_CH},     // IQ channels
    parameter int K     = {K},     // conv kernel size (must be odd)
    parameter int C1F   = {C1F},    // conv1 output filters
    parameter int C2F   = {C2F},    // conv2 output filters
    parameter int D1U   = {D1U},    // dense1 units
    parameter int NC    = {NC}      // number of classes
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

    // ── Local parameters ─────────────────────────────────────────────────────
    localparam int PAD     = K / 2;          // same-padding half-width (2)
    localparam int C1_MAC  = K * IN_CH;      // MACs per t for conv1 (10)
    localparam int C2_MAC  = K * C1F;        // MACs per t for conv2 (80)
    localparam int KC1_MAX = C1_MAC;         // kc counter max for conv1
    localparam int KC2_MAX = C2_MAC;         // kc counter max for conv2

    // ── State machine ────────────────────────────────────────────────────────
    typedef enum logic [3:0] {{
        S_IDLE, S_RECV,
        S_CONV1, S_CONV2,
        S_GAP, S_DENSE1, S_DENSE2,
        S_ARGMAX, S_DONE
    }} state_t;
    state_t state;

    // ── Weight ROMs ──────────────────────────────────────────────────────────
    // conv1: kernel[k, cin, fout]  flat = k*IN_CH*C1F + cin*C1F + fout
    logic signed [DW-1:0] c1w [0:K*IN_CH*C1F-1];
    logic signed [DW-1:0] c1b [0:C1F-1];
    // conv2: kernel[k, cin, fout]  flat = k*C1F*C2F + cin*C2F + fout
    logic signed [DW-1:0] c2w [0:K*C1F*C2F-1];
    logic signed [DW-1:0] c2b [0:C2F-1];
    // dense1: W[in, out]           flat = in*D1U + out
    logic signed [DW-1:0] d1w [0:C2F*D1U-1];
    logic signed [DW-1:0] d1b [0:D1U-1];
    // dense2: W[in, out]           flat = in*NC + out
    logic signed [DW-1:0] d2w [0:D1U*NC-1];
    logic signed [DW-1:0] d2b [0:NC-1];

    initial begin
        $readmemh("c1w.hex", c1w);
        $readmemh("c1b.hex", c1b);
        $readmemh("c2w.hex", c2w);
        $readmemh("c2b.hex", c2b);
        $readmemh("d1w.hex", d1w);
        $readmemh("d1b.hex", d1b);
        $readmemh("d2w.hex", d2w);
        $readmemh("d2b.hex", d2b);
    end

    // ── Feature map memories (inferred as BRAMs by synthesis) ────────────────
    logic signed [DW-1:0] input_mem [0:SEQ-1][0:IN_CH-1];
    logic signed [DW-1:0] c1_mem   [0:SEQ-1][0:C1F-1];
    logic signed [DW-1:0] c2_mem   [0:SEQ-1][0:C2F-1];

    // ── Intermediate registers ───────────────────────────────────────────────
    logic signed [ACC_W-1:0] c1_acc [0:C1F-1];
    logic signed [ACC_W-1:0] c2_acc [0:C2F-1];
    logic signed [ACC_W-1:0] gap_acc[0:C2F-1];  // wider: accumulates 512 values
    logic signed [DW-1:0]   gap_out [0:C2F-1];
    logic signed [ACC_W-1:0] d1_acc [0:D1U-1];
    logic signed [DW-1:0]   d1_out  [0:D1U-1];
    logic signed [ACC_W-1:0] d2_acc [0:NC-1];

    // ── Counters ─────────────────────────────────────────────────────────────
    logic [$clog2(SEQ)-1:0]   t_cnt;
    logic [$clog2(C2_MAC):0]  kc_cnt;   // sized for max(C1_MAC, C2_MAC) = 80

    // ── ELU approximation ────────────────────────────────────────────────────
    // ELU(x) = x            if x >= 0
    //        = x / 2        if -1.0 <= x < 0   (linear approx of e^x - 1)
    //        = -1.0         if x < -1.0         (saturate)
    // The input `acc` is the raw accumulator value; shift right by FRAC to get
    // the Q8.8 result, then apply the activation.
    function automatic logic signed [DW-1:0] elu(
        input logic signed [ACC_W-1:0] acc
    );
        logic signed [DW-1:0] x;
        x = acc[FRAC +: DW];  // arithmetic shift right by FRAC (extract Q8.8)
        if (x >= 0)
            return x;
        else if (x >= -$signed(DW'(SCALE)))   // -1.0 in Q8.8 = -256
            return x >>> 1;                    // ≈ (e^x - 1) for small |x|
        else
            return -$signed(DW'(SCALE));       // saturate at -1.0
    endfunction

    // ── Main state machine ───────────────────────────────────────────────────
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

                // ── Wait for start pulse ──────────────────────────────────
                S_IDLE: begin
                    if (start) begin
                        state  <= S_RECV;
                        t_cnt  <= '0;
                    end
                end

                // ── Receive 512 IQ samples ────────────────────────────────
                S_RECV: begin
                    if (sample_valid) begin
                        input_mem[t_cnt][0] <= iq_real;
                        input_mem[t_cnt][1] <= iq_imag;
                        if (t_cnt == SEQ - 1) begin
                            state  <= S_CONV1;
                            t_cnt  <= '0;
                            kc_cnt <= '0;
                        end else begin
                            t_cnt <= t_cnt + 1;
                        end
                    end
                end

                // ── Conv1: kernel(5,2→16) with same padding ───────────────
                // kc_cnt = 0        : load bias into accumulator
                // kc_cnt = 1..C1_MAC: accumulate MAC for (k,c) pair
                S_CONV1: begin
                    if (kc_cnt == 0) begin
                        // Bias initialisation – scale up so accumulator is Q.2*FRAC
                        for (int f = 0; f < C1F; f++)
                            c1_acc[f] <= ACC_W'($signed(c1b[f])) <<< FRAC;
                        kc_cnt <= kc_cnt + 1;
                    end else begin
                        automatic int kc  = kc_cnt - 1;
                        automatic int k   = kc / IN_CH;
                        automatic int ch  = kc % IN_CH;
                        automatic int t_s = int'(t_cnt) + k - PAD;

                        for (int f = 0; f < C1F; f++) begin
                            if (t_s >= 0 && t_s < SEQ) begin
                                c1_acc[f] <= c1_acc[f] +
                                    ACC_W'($signed(input_mem[t_s][ch])) *
                                    ACC_W'($signed(c1w[k*IN_CH*C1F + ch*C1F + f]));
                            end
                        end

                        if (kc_cnt == C1_MAC) begin
                            // Apply ELU and write feature map
                            for (int f = 0; f < C1F; f++)
                                c1_mem[t_cnt][f] <= elu(c1_acc[f]);
                            if (t_cnt == SEQ - 1) begin
                                state  <= S_CONV2;
                                t_cnt  <= '0;
                            end else begin
                                t_cnt <= t_cnt + 1;
                            end
                            kc_cnt <= '0;
                        end else begin
                            kc_cnt <= kc_cnt + 1;
                        end
                    end
                end

                // ── Conv2: kernel(5,16→32) with same padding ──────────────
                S_CONV2: begin
                    if (kc_cnt == 0) begin
                        for (int f = 0; f < C2F; f++)
                            c2_acc[f] <= ACC_W'($signed(c2b[f])) <<< FRAC;
                        kc_cnt <= kc_cnt + 1;
                    end else begin
                        automatic int kc  = kc_cnt - 1;
                        automatic int k   = kc / C1F;
                        automatic int ch  = kc % C1F;
                        automatic int t_s = int'(t_cnt) + k - PAD;

                        for (int f = 0; f < C2F; f++) begin
                            if (t_s >= 0 && t_s < SEQ) begin
                                c2_acc[f] <= c2_acc[f] +
                                    ACC_W'($signed(c1_mem[t_s][ch])) *
                                    ACC_W'($signed(c2w[k*C1F*C2F + ch*C2F + f]));
                            end
                        end

                        if (kc_cnt == C2_MAC) begin
                            for (int f = 0; f < C2F; f++)
                                c2_mem[t_cnt][f] <= elu(c2_acc[f]);
                            if (t_cnt == SEQ - 1) begin
                                state  <= S_GAP;
                                t_cnt  <= '0;
                                for (int f = 0; f < C2F; f++)
                                    gap_acc[f] <= '0;
                            end else begin
                                t_cnt <= t_cnt + 1;
                            end
                            kc_cnt <= '0;
                        end else begin
                            kc_cnt <= kc_cnt + 1;
                        end
                    end
                end

                // ── GlobalAveragePooling1D: accumulate over SEQ ───────────
                // gap_out[f] = sum(c2_mem[t][f], t=0..SEQ-1) / SEQ
                // Division by 512 = arithmetic right-shift by 9.
                S_GAP: begin
                    for (int f = 0; f < C2F; f++)
                        gap_acc[f] <= gap_acc[f] + ACC_W'($signed(c2_mem[t_cnt][f]));

                    if (t_cnt == SEQ - 1) begin
                        // Divide by SEQ (512 = 2^9) via arithmetic shift
                        for (int f = 0; f < C2F; f++)
                            gap_out[f] <= DW'(gap_acc[f] >>> 9);
                        state  <= S_DENSE1;
                        t_cnt  <= '0;
                        for (int n = 0; n < D1U; n++)
                            d1_acc[n] <= ACC_W'($signed(d1b[n])) <<< FRAC;
                    end else begin
                        t_cnt <= t_cnt + 1;
                    end
                end

                // ── Dense1: (32→16) with ELU ──────────────────────────────
                // Iterate over the 32 input neurons; all 16 output neurons
                // accumulate in parallel.
                S_DENSE1: begin
                    for (int n = 0; n < D1U; n++)
                        d1_acc[n] <= d1_acc[n] +
                            ACC_W'($signed(gap_out[t_cnt])) *
                            ACC_W'($signed(d1w[int'(t_cnt)*D1U + n]));

                    if (t_cnt == C2F - 1) begin
                        for (int n = 0; n < D1U; n++)
                            d1_out[n] <= elu(d1_acc[n]);
                        state  <= S_DENSE2;
                        t_cnt  <= '0;
                        for (int k = 0; k < NC; k++)
                            d2_acc[k] <= ACC_W'($signed(d2b[k])) <<< FRAC;
                    end else begin
                        t_cnt <= t_cnt + 1;
                    end
                end

                // ── Dense2: (16→3) – no activation, argmax follows ────────
                S_DENSE2: begin
                    for (int k = 0; k < NC; k++)
                        d2_acc[k] <= d2_acc[k] +
                            ACC_W'($signed(d1_out[t_cnt])) *
                            ACC_W'($signed(d2w[int'(t_cnt)*NC + k]));

                    if (t_cnt == D1U - 1) begin
                        state <= S_ARGMAX;
                    end else begin
                        t_cnt <= t_cnt + 1;
                    end
                end

                // ── Argmax: find class with highest score ─────────────────
                S_ARGMAX: begin
                    begin
                        automatic logic signed [ACC_W-1:0] best     = d2_acc[0];
                        automatic logic [1:0]              best_idx = 2'd0;
                        for (int k = 1; k < NC; k++) begin
                            if (d2_acc[k] > best) begin
                                best     = d2_acc[k];
                                best_idx = 2'(k);
                            end
                        end
                        class_out <= best_idx;
                    end
                    state <= S_DONE;
                end

                // ── Output result ─────────────────────────────────────────
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

sv_path = os.path.join(OUT_DIR, "nas_classifier_top.sv")
with open(sv_path, "w") as f:
    f.write(sv)

print(f"SystemVerilog written → {sv_path}")


# ── README ────────────────────────────────────────────────────────────────────
readme = f"""\
# verilog_output — Synthesizable SystemVerilog

Generated from `results_nas_v2_paper_pruning/nas_paper_model_pruned_55pct_1571weights.keras`.

## Files

| File | Description |
|------|-------------|
| `nas_classifier_top.sv` | Synthesizable top-level SystemVerilog module |
| `c1w.hex` / `c1b.hex` | Conv1 weights / biases (Q8.8 fixed-point, hex) |
| `c2w.hex` / `c2b.hex` | Conv2 weights / biases |
| `d1w.hex` / `d1b.hex` | Dense1 weights / biases |
| `d2w.hex` / `d2b.hex` | Dense2 weights / biases |

## Fixed-point format

**Q8.8** — signed 16-bit, 8 fractional bits.
Real value = raw_integer / 256.
Accumulator: 40-bit signed (prevents overflow across MAC chains).

## Network topology (after BN folding)

```
Input  : 512 × 2  (IQ: real + imag, streamed in sample-by-sample)
Conv1  : kernel (5, 2→16),  ELU,  same padding
Conv2  : kernel (5, 16→32), ELU,  same padding
GAP    : GlobalAveragePooling1D  →  (32,)
Dense1 : (32→16),  ELU
Dense2 : (16→3),   argmax
Output : 2-bit class label  (0 = LTE,  1 = DVB-T,  2 = WiFi)
```

BatchNorm layers are absorbed into the preceding conv weights at generation time —
no BN hardware is needed.

## Interface

```systemverilog
module nas_classifier_top (
    input  logic        clk,
    input  logic        rst_n,         // active-low synchronous reset
    input  logic        start,         // single-cycle pulse: begin inference
    input  logic        sample_valid,  // IQ sample present on iq_real/iq_imag
    input  logic signed [15:0] iq_real,
    input  logic signed [15:0] iq_imag,
    output logic [1:0]  class_out,     // 0=LTE  1=DVB-T  2=WiFi
    output logic        result_valid   // single-cycle pulse when done
);
```

## Cycle budget (@ 100 MHz → ≈ 0.48 ms per inference)

| Stage | Cycles |
|-------|--------|
| RECV  (512 samples) | 512 |
| CONV1 (512 × 11)    | 5,632 |
| CONV2 (512 × 81)    | 41,472 |
| GAP   (512 steps)   | 512 |
| DENSE1 (32 inputs)  | 32 |
| DENSE2 (16 inputs)  | 16 |
| ARGMAX              | 1 |
| **Total**           | **~48 k** |

## ELU approximation

Hardware ELU is approximated as:
- x ≥ 0   → x
- −1 ≤ x < 0  → x / 2   (linear fit of e^x − 1)
- x < −1  → −1.0  (saturation)

## Usage

```bash
# 1. Activate the project venv and regenerate at any time:
source venv/bin/activate
python generate_verilog.py

# 2. Run simulation (example with Icarus Verilog):
iverilog -g2012 -o sim nas_classifier_top.sv your_tb.sv
vvp sim

# 3. Synthesis (Vivado example):
#    - Add nas_classifier_top.sv as design source
#    - Add *.hex files to the project directory (or set the path in $readmemh)
#    - Set top module: nas_classifier_top
```
"""

readme_path = os.path.join(OUT_DIR, "README.md")
with open(readme_path, "w") as f:
    f.write(readme)

print(f"README written       → {readme_path}")
print("\nDone. Run with: python generate_verilog.py")
