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
