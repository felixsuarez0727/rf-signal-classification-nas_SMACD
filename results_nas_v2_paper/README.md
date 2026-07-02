# results_nas_v2_paper

NAS search results that produced the model presented in the SMACD paper.

**Configuration:** population=16, generations=10, eval_epochs=8, train_epochs=60  
**Result:** 3,539 parameters — 90.1% test accuracy

---

## models/

| File | Description |
|------|-------------|
| `nas_optimized_wireless_classifier.keras` | Final trained model (paper model) |
| `nas_model.mlpackage` | Core ML conversion for iOS deployment |
| `nas_results.json` | Metrics and architecture of the best candidate found |
| `nas_training_log.txt` | Final training log |

## figures/

| File | Description |
|------|-------------|
| `nas_confusion_matrix_absolute.png` | Confusion matrix (absolute values) |
| `nas_confusion_matrix_combined.png` | Confusion matrix (absolute + percentage) |
| `nas_confusion_matrix_percentage.png` | Confusion matrix (percentages) |
| `nas_search_progress.png` | Fitness evolution across generations |

---

> The pruned version of this model (55.6% sparsity) is located in `results_nas_v2_paper_pruning/`.
