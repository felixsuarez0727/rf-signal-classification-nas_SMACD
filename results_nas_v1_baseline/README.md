# results_nas_v1_baseline

Results of the initial NAS search run used as a baseline before the paper configuration.

**Configuration:** population=8, generations=5  
**Result:** 4,715 parameters — 86.6% test accuracy

---

## models/

| File | Description |
|------|-------------|
| `nas_optimized_wireless_classifier.keras` | Best model found during baseline search |
| `nas_optimized_wireless_classifier_pruned.keras` | Pruned variant of the baseline model |
| `nas_model.mlpackage` | Core ML conversion for iOS deployment |
| `nas_model.tflite` | TFLite conversion (dynamic range) |
| `nas_model_float16.tflite` | TFLite conversion (FP16) |
| `nas_model_float32.tflite` | TFLite conversion (FP32) |
| `saved_model/` | TensorFlow SavedModel format |
| `ios_nas_model.keras` | Model exported for iOS integration |
| `nas_coreml_model.keras` | Model used during Core ML conversion |
| `nas_real_model.keras` | Intermediate model checkpoint |
| `simple_nas_model.keras` | Simplified architecture variant |
| `nas_results.json` | Metrics and architecture of the best candidate found |
| `nas_pruning_results.json` | Pruning metrics for the baseline model |
| `nas_training_log.txt` | Final training log |

## figures/

| File | Description |
|------|-------------|
| `nas_confusion_matrix_absolute.png` | Confusion matrix (absolute values) |
| `nas_confusion_matrix_combined.png` | Confusion matrix (absolute + percentage) |
| `nas_confusion_matrix_percentage.png` | Confusion matrix (percentages) |
| `nas_search_progress.png` | Fitness evolution across generations |

---

> The definitive paper model (3,539 params, 90.1% accuracy) is in `results_nas_v2_paper/`.
