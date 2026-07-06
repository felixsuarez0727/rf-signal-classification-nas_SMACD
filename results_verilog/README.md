# results_verilog

Hardware simulation of the pruned NAS classifier in synthesizable SystemVerilog.

**Tool:** Icarus Verilog v13 (`iverilog -g2012`)  
**Clock:** 100 MHz  
**Arithmetic:** Fixed-point Q8.8 (16-bit signed, 8 fractional bits, scale = 256)  
**Source model:** `results_nas_v2_paper_pruning/nas_paper_model_pruned_55pct_1571weights.keras`

---

## Architecture

```
Conv1D(5, 2→16, ELU) → Conv1D(5, 16→32, ELU) → GAP → Dense(32→16, ELU) → Dense(16→3, argmax)
```

BatchNorm layers are folded into the convolutional weights before quantization.  
Accumulator width: 40-bit signed (prevents overflow across MAC chains).

---

## Accuracy Results

| Class  | Correct / Total     | Accuracy |
|--------|---------------------|----------|
| LTE    | 1,019 / 12,888      | 7.9%     |
| DVB-T  | 6 / 11,814          | 0.1%     |
| WiFi   | 9,613 / 9,666       | 99.5%    |
| **Overall** | **10,638 / 34,368** | **31.0%** |

Float32 reference accuracy: **89.8%**

The accuracy drop is due to Q8.8 precision loss — the LTE/DVB-T decision boundary collapses at 8 fractional bits. WiFi has sufficiently distinct spectral features to survive quantization. Increasing to Q16 or applying quantization-aware training (QAT) would recover accuracy.

---

## Port Interface

| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| `clk` | in | 1 | 100 MHz system clock |
| `rst_n` | in | 1 | Active-low synchronous reset |
| `start` | in | 1 | Single-cycle pulse: begin inference |
| `sample_valid` | in | 1 | IQ sample present on inputs |
| `iq_real` | in | 16 | Q8.8 real component |
| `iq_imag` | in | 16 | Q8.8 imaginary component |
| `class_out` | out | 2 | Predicted class (0=LTE, 1=DVB-T, 2=WiFi) |
| `result_valid` | out | 1 | Single-cycle pulse: output valid |

---

## Cycle Budget @ 100 MHz (~0.49 ms per inference)

| Stage  | Cycles |
|--------|--------|
| RECV   | 512 |
| CONV1  | 512 × 12 = 6,144 |
| CONV2  | 512 × 82 = 41,984 |
| GAP    | 513 |
| DENSE1 | 33 |
| DENSE2 | 16 |
| ARGMAX | 1 |
| **Total** | **~49,203** |

---

## Files

| File | Description |
|------|-------------|
| `nas_classifier_top.sv` | Synthesizable SystemVerilog module |
| `nas_classifier_tb.sv` | Testbench — streams 9 real IQ test vectors |
| `c1w.hex`, `c1b.hex` | Conv1 weights and biases (Q8.8) |
| `c2w.hex`, `c2b.hex` | Conv2 weights and biases (Q8.8) |
| `d1w.hex`, `d1b.hex` | Dense1 weights and biases (Q8.8) |
| `d2w.hex`, `d2b.hex` | Dense2 weights and biases (Q8.8) |
| `test_vectors.hex` | 9 IQ test samples (3 per class), Q8.8 |
| `test_labels.hex` | Expected class labels |
| `test_info.txt` | Human-readable test sample summary |

---

## How to Run

```bash
cd results_verilog/
iverilog -g2012 -o sim nas_classifier_top.sv nas_classifier_tb.sv
vvp sim
```

---

## Scripts (project root)

| Script | Description |
|--------|-------------|
| `generate_verilog.py` | Loads pruned model, folds BatchNorm, quantizes to Q8.8, generates `.sv` and `.hex` files |
| `generate_test_vectors.py` | Extracts test samples from dataset, exports to Q8.8 hex format |
| `eval_q88_full.py` | Evaluates Q8.8 accuracy on the full test set (pure numpy, no TensorFlow) |
| `sim_q88.py` | Verifies 9-sample hardware output against numpy Q8.8 reference |
