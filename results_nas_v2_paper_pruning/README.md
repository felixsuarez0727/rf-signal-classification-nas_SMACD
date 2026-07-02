# results_nas_v2_paper_pruning

Results of applying magnitude pruning to the paper model (`results_nas_v2_paper`).

**Method:** 3-stage progressive magnitude pruning (30% → 45% → 60% sparsity) with fine-tuning  
**Result:** 55.6% effective sparsity — 89.83% test accuracy (−0.92% vs. unpruned)

---

## Files

| File | Description |
|------|-------------|
| `nas_paper_model_pruned_55pct_1571weights.keras` | Pruned Keras model (1,571 non-zero weights out of 3,539) |
| `nas_paper_pruned_float32.tflite` | TFLite conversion (FP32) |
| `nas_paper_pruned_float16.tflite` | TFLite conversion (FP16) |
| `nas_paper_pruned_int8.tflite` | TFLite conversion (INT8 dynamic range) |
| `nas_paper_pruning_results.json` | Full pruning report (accuracy, sparsity, size) |

---

> Note: TFLite quantization yielded no meaningful size reduction due to the model being extremely compact (< 0.1 MB). The Keras model with zero-masked weights is the primary pruning artifact.  
> The original unpruned model is in `results_nas_v2_paper/models/`.
