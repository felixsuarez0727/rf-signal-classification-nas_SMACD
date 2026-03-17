"""
Magnitude pruning + fine-tuning for the best NAS model.

Compatible with current TensorFlow/Keras stack in this project.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from train import CLASSES, load_dataset


def get_balanced_subset(X, y, n_samples_per_class, seed=42):
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y)
    indices = []
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        n_samples = min(n_samples_per_class, len(class_indices))
        selected = rng.choice(class_indices, n_samples, replace=False)
        indices.extend(selected.tolist())
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def reduce_chunks_1024_to_512(X):
    step = 1024 // 512
    return X[:, ::step, :]


def evaluate_accuracy(model, X, y):
    preds = model.predict(X, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    return float(np.mean(pred_labels == y))


def count_nonzero_weights(model):
    nonzero = 0
    total = 0
    for w in model.get_weights():
        total += w.size
        nonzero += int(np.count_nonzero(w))
    return nonzero, total


def build_global_masks_for_target_sparsity(model, target_sparsity):
    """
    Create a global magnitude mask so that approximately target_sparsity
    of pruneable weights are zeroed.
    """
    pruneable = []
    for w in model.get_weights():
        # Prune tensors with rank >= 2 (kernels); keep biases/BN params untouched.
        if w.ndim >= 2:
            pruneable.append(np.abs(w).reshape(-1))

    if not pruneable:
        return [None for _ in model.get_weights()]

    flat_all = np.concatenate(pruneable)
    threshold = np.percentile(flat_all, target_sparsity * 100.0)

    masks = []
    for w in model.get_weights():
        if w.ndim >= 2:
            mask = (np.abs(w) > threshold).astype(w.dtype)
            masks.append(mask)
        else:
            masks.append(None)
    return masks


def apply_masks_to_model_weights(model, masks):
    new_weights = []
    for w, m in zip(model.get_weights(), masks):
        if m is None:
            new_weights.append(w)
        else:
            new_weights.append(w * m)
    model.set_weights(new_weights)


class HardMaskCallback(tf.keras.callbacks.Callback):
    """
    Enforce fixed masks after each batch so pruned weights stay at zero.
    """
    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def on_train_batch_end(self, batch, logs=None):
        apply_masks_to_model_weights(self.model, self.masks)


def main():
    parser = argparse.ArgumentParser(description="Prune NAS model and fine-tune it.")
    parser.add_argument(
        "--model-path",
        default="results_nas/nas_optimized_wireless_classifier.keras",
        help="Path to trained NAS model."
    )
    parser.add_argument(
        "--output-path",
        default="results_nas/nas_optimized_wireless_classifier_pruned.keras",
        help="Path to save pruned model."
    )
    parser.add_argument("--epochs", type=int, default=12, help="Total fine-tuning epochs across pruning stages.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--final-sparsity", type=float, default=0.6, help="Final target sparsity [0, 1].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset sampling.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}. Run nas_fast_demo.py first."
        )

    print("📂 Loading dataset...")
    X_train, y_train = load_dataset(os.path.join("split_dataset", "train"), chunk_samples=1024)
    X_val, y_val = load_dataset(os.path.join("split_dataset", "validation"), chunk_samples=1024)
    X_test, y_test = load_dataset(os.path.join("split_dataset", "test"), chunk_samples=1024)

    y_train_sparse = np.argmax(y_train, axis=1)
    y_val_sparse = np.argmax(y_val, axis=1)
    y_test_sparse = np.argmax(y_test, axis=1)

    X_train_subset, y_train_subset = get_balanced_subset(X_train, y_train_sparse, 1000, seed=args.seed)
    X_val_subset, y_val_subset = get_balanced_subset(X_val, y_val_sparse, 400, seed=args.seed + 1)
    X_test_subset, y_test_subset = get_balanced_subset(X_test, y_test_sparse, 400, seed=args.seed + 2)

    X_train_opt = reduce_chunks_1024_to_512(X_train_subset)
    X_val_opt = reduce_chunks_1024_to_512(X_val_subset)
    X_test_opt = reduce_chunks_1024_to_512(X_test_subset)

    print(f"✅ Train: {X_train_opt.shape} | Val: {X_val_opt.shape} | Test: {X_test_opt.shape}")
    print("🧠 Loading baseline NAS model...")
    baseline_model = tf.keras.models.load_model(args.model_path)
    baseline_accuracy = evaluate_accuracy(baseline_model, X_test_opt, y_test_subset)
    baseline_params = int(baseline_model.count_params())

    print(f"📊 Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"📦 Baseline params: {baseline_params:,}")

    # Progressive pruning stages to reduce abrupt accuracy drops
    stage_sparsities = [0.30, 0.45, args.final_sparsity]
    stage_sparsities = sorted(set(min(max(s, 0.0), 0.95) for s in stage_sparsities))
    epochs_per_stage = max(2, args.epochs // max(1, len(stage_sparsities)))

    final_model = baseline_model
    all_histories = []

    for i, stage_sparsity in enumerate(stage_sparsities, start=1):
        print(f"✂️ Pruning stage {i}/{len(stage_sparsities)} to sparsity={stage_sparsity:.2f}")
        masks = build_global_masks_for_target_sparsity(final_model, stage_sparsity)
        apply_masks_to_model_weights(final_model, masks)

        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = [
            HardMaskCallback(masks),
            tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-6, monitor="val_loss"),
        ]

        print("🏋️ Fine-tuning pruned model...")
        history = final_model.fit(
            X_train_opt,
            y_train_subset,
            validation_data=(X_val_opt, y_val_subset),
            epochs=epochs_per_stage,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        all_histories.append(history.history)
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    pruned_accuracy = evaluate_accuracy(final_model, X_test_opt, y_test_subset)
    nonzero_params, total_weights = count_nonzero_weights(final_model)
    sparsity = 1.0 - (nonzero_params / total_weights if total_weights else 0.0)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    final_model.save(args.output_path)
    saved_size_mb = os.path.getsize(args.output_path) / (1024 * 1024)

    print("\n✅ PRUNING COMPLETED")
    print(f"   Baseline test accuracy: {baseline_accuracy:.4f}")
    print(f"   Pruned test accuracy:   {pruned_accuracy:.4f}")
    print(f"   Accuracy delta:         {pruned_accuracy - baseline_accuracy:+.4f}")
    print(f"   Nominal params:         {baseline_params:,}")
    print(f"   Non-zero weights:       {nonzero_params:,}/{total_weights:,}")
    print(f"   Effective sparsity:     {sparsity:.2%}")
    print(f"   Saved model size:       {saved_size_mb:.2f} MB")
    print(f"   Saved model path:       {args.output_path}")

    results_path = os.path.join(os.path.dirname(args.output_path), "nas_pruning_results.json")
    payload = {
        "model_path": args.model_path,
        "pruned_model_path": args.output_path,
        "classes": CLASSES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "target_final_sparsity": args.final_sparsity,
        "baseline": {
            "test_accuracy": baseline_accuracy,
            "nominal_params": baseline_params,
        },
        "pruned": {
            "test_accuracy": pruned_accuracy,
            "accuracy_delta": pruned_accuracy - baseline_accuracy,
            "nominal_params": baseline_params,
            "nonzero_weights": nonzero_params,
            "total_weights": total_weights,
            "effective_sparsity": sparsity,
            "saved_model_size_mb": saved_size_mb,
        },
        "training_history": {
            "stages": len(stage_sparsities),
            "epochs_per_stage": epochs_per_stage,
            "histories": all_histories,
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"   Saved pruning report:   {results_path}")


if __name__ == "__main__":
    main()
