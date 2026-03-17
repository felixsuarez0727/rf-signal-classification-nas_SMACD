"""
NAS Fast Demo - Optimized version for quick results
"""

import numpy as np
import sys
import os
import random
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append('.')

from neural_architecture_search.nas_optimization import WirelessSignalNAS
from train import load_dataset, CLASSES


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def main(args):
    """
    Fast NAS demonstration with optimized settings
    """
    print("🚀 NAS FAST DEMO - OPTIMIZED VERSION")
    print("=" * 50)
    
    try:
        set_global_seed(args.seed)

        # 1. Load SMALL subset of data
        print("📂 Loading SMALL dataset subset...")
        X_train, y_train = load_dataset(os.path.join("split_dataset", "train"), chunk_samples=1024)
        X_val, y_val = load_dataset(os.path.join("split_dataset", "validation"), chunk_samples=1024)
        X_test, y_test = load_dataset(os.path.join("split_dataset", "test"), chunk_samples=1024)
        
        # Convert to sparse labels
        y_train_sparse = np.argmax(y_train, axis=1)
        y_val_sparse = np.argmax(y_val, axis=1)
        y_test_sparse = np.argmax(y_test, axis=1)
        
        # Use BALANCED smaller subset for speed
        def get_balanced_subset(X, y, n_samples_per_class=500):
            """Get balanced subset with equal samples per class"""
            unique_classes = np.unique(y)
            indices = []
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                # Take min of requested samples or available samples
                n_samples = min(n_samples_per_class, len(class_indices))
                selected_indices = np.random.choice(class_indices, n_samples, replace=False)
                indices.extend(selected_indices)
            
            indices = np.array(indices)
            np.random.shuffle(indices)
            return X[indices], y[indices]
        
        # Use configurable balanced subsets
        X_train_subset, y_train_subset = get_balanced_subset(
            X_train, y_train_sparse, args.train_samples_per_class
        )
        X_val_subset, y_val_subset = get_balanced_subset(
            X_val, y_val_sparse, args.val_samples_per_class
        )
        X_test_subset, y_test_subset = get_balanced_subset(
            X_test, y_test_sparse, args.test_samples_per_class
        )
        
        print(f"✅ Data loaded (BALANCED SUBSET):")
        print(f"   Train: {X_train_subset.shape}, Val: {X_val_subset.shape}, Test: {X_test_subset.shape}")
        
        # Print class distribution
        print("\n📊 Class distribution in subsets:")
        for dataset_name, y_data in [("Train", y_train_subset), ("Val", y_val_subset), ("Test", y_test_subset)]:
            unique, counts = np.unique(y_data, return_counts=True)
            print(f"   {dataset_name}: ", end="")
            for i, count in zip(unique, counts):
                print(f"{CLASSES[i]}={count} ", end="")
            print()
        
        # 2. Simple chunk size reduction (no feature optimization)
        print("\n🔧 Applying simple chunk reduction...")
        step = 1024 // 512
        X_train_opt = X_train_subset[:, ::step, :]
        X_val_opt = X_val_subset[:, ::step, :]
        X_test_opt = X_test_subset[:, ::step, :]
        
        print(f"   ✅ Chunks reduced: {X_train_opt.shape}")
        
        # 3. Initialize IMPROVED NAS
        print("\n🧬 Initializing IMPROVED NAS...")
        nas = WirelessSignalNAS(
            input_shape=X_train_opt.shape[1:],
            num_classes=len(CLASSES),
            population_size=args.population_size,
            generations=args.generations,
            eval_epochs=args.eval_epochs
        )
        
        # 4. Run FAST NAS search
        print("\n🔍 Running FAST NAS search...")
        best_architecture = nas.search(X_train_opt, y_train_subset, X_val_opt, y_val_subset)
        
        # 5. Build and train best model QUICKLY
        print("\n🏗️ Building and training best model...")
        best_model = nas._build_model_from_architecture(best_architecture)
        
        # Train with BETTER regularization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),  # Better monitoring
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, monitor='val_loss')
        ]
        
        print("🎯 Training optimized model with REGULARIZATION...")
        history = best_model.fit(
            X_train_opt, y_train_subset,
            validation_data=(X_val_opt, y_val_subset),
            epochs=args.train_epochs,
            batch_size=32,  # Stable batch size
            callbacks=callbacks,
            verbose=1
        )
        
        # 6. Quick evaluation
        print("\n📊 Quick evaluation...")
        predictions = best_model.predict(X_test_opt, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == y_test_subset)
        
        # Get metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Fix labels issue - ensure all classes are represented
        unique_labels = sorted(list(set(y_test_subset) | set(predicted_classes)))
        
        class_report = classification_report(
            y_test_subset, predicted_classes,
            target_names=CLASSES,
            labels=unique_labels,
            zero_division=0,
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(y_test_subset, predicted_classes, labels=unique_labels)
        
        # 7. Create results directory
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
        print(f"\n📁 Creating results directory: {results_dir}/")
        
        # 8. Save model
        print("\n💾 Saving NAS-optimized model...")
        model_path = os.path.join(results_dir, "nas_optimized_wireless_classifier.keras")
        best_model.save(model_path)
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   ✅ Saved: {model_path}")
        
        # 9. Generate confusion matrices
        print("\n📊 Generating confusion matrices...")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        
        # Create display labels that match the actual classes found
        display_labels = [CLASSES[i] for i in unique_labels if i < len(CLASSES)]
        
        # Absolute numbers
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)
        disp1.plot(ax=ax1, cmap='Blues')
        ax1.set_title('NAS Model - Confusion Matrix (Absolute Numbers)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        confusion_abs_path = os.path.join(results_dir, "nas_confusion_matrix_absolute.png")
        plt.savefig(confusion_abs_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {confusion_abs_path}")
        
        # Percentages
        cm_percent = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=display_labels)
        disp2.plot(ax=ax2, cmap='Blues', values_format='.1f')
        ax2.set_title('NAS Model - Confusion Matrix (Percentages %)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        confusion_percent_path = os.path.join(results_dir, "nas_confusion_matrix_percentage.png")
        plt.savefig(confusion_percent_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {confusion_percent_path}")
        
        # Combined
        fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
        disp3 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)
        disp3.plot(ax=ax3, cmap='Blues')
        ax3.set_title('NAS Model - Absolute Numbers', fontsize=12, fontweight='bold')
        
        disp4 = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=display_labels)
        disp4.plot(ax=ax4, cmap='Blues', values_format='.1f')
        ax4.set_title('NAS Model - Percentages (%)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        confusion_combined_path = os.path.join(results_dir, "nas_confusion_matrix_combined.png")
        plt.savefig(confusion_combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {confusion_combined_path}")
        
        # 10. Save detailed logs
        print("\n📝 Saving detailed logs...")
        log_path = os.path.join(results_dir, "nas_training_log.txt")
        with open(log_path, 'w') as f:
            f.write("=== NAS FAST DEMO RESULTS ===\n\n")
            f.write(f"Model: NAS-Optimized Wireless Classifier (FAST VERSION)\n")
            f.write(f"Date: {np.datetime64('now')}\n\n")

            f.write("=== CONFIGURATION ===\n")
            f.write(f"Input shape: {X_train_opt.shape[1:]}\n")
            f.write(f"Dataset subset: 500 samples per class\n")
            f.write(f"Epochs: 10 (fast training)\n")
            f.write(f"Batch size: 32\n")
            f.write(f"Population size: {nas.population_size}\n")
            f.write(f"Generations: {nas.generations}\n\n")

            f.write("=== TRAINING HISTORY ===\n")
            for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history['loss'], history.history['accuracy'],
                history.history['val_loss'], history.history['val_accuracy']
            )):
                f.write(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}\n")

            f.write("\n=== FINAL RESULTS ===\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"Parameters: {best_model.count_params():,}\n")
            f.write(f"Model Size: {model_size:.2f} MB\n\n")

            f.write("=== METRICS PER CLASS ===\n")
            for class_name in CLASSES:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    f.write(f"{class_name}: Precision={metrics['precision']:.3f}, "
                            f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\n")

            f.write("\n=== NAS SEARCH DETAILS ===\n")
            search_results = nas.get_search_results()
            f.write(f"Search space size: {search_results['search_space_size']:,}\n")
            f.write(f"Architectures evaluated: {search_results['population_size'] * search_results['generations']:,}\n")
            f.write(f"Best fitness score: {best_architecture['fitness']:.4f}\n")
            f.write(f"Best architecture parameters: {best_architecture['metrics']['parameters']:,}\n")
        print(f"   ✅ Saved: {log_path}")
        
        # 11. Save results JSON
        print("\n📋 Saving results to JSON...")
        import json
        
        results_json = {
            "model_info": {
                "name": "nas_optimized_wireless_classifier_fast",
                "parameters": int(best_model.count_params()),
                "model_size_mb": float(model_size),
                "input_shape": list(X_train_opt.shape[1:]),
                "classes": list(CLASSES)
            },
            "training_history": {
                "epochs": len(history.history['loss']),
                "final_accuracy": float(history.history['accuracy'][-1]),
                "final_val_accuracy": float(history.history['val_accuracy'][-1]),
                "final_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_accuracy'][-1])
            },
            "test_results": {
                "overall_accuracy": float(accuracy),
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": {k: {m: float(v) if isinstance(v, (int, float, np.number)) else v
                                           for m, v in metrics.items()}
                                       for k, metrics in class_report.items()
                                       if k not in ['accuracy', 'macro avg', 'weighted avg']}
            },
            "nas_search": {
                "population_size": nas.population_size,
                "generations": nas.generations,
                "best_fitness": float(best_architecture['fitness']),
                "best_parameters": int(best_architecture['metrics']['parameters']),
                "search_space_size": search_results['search_space_size']
            }
        }
        
        json_path = os.path.join(results_dir, "nas_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"   ✅ Saved: {json_path}")
        
        # 12. Generate NAS progress
        print("\n📈 Generating NAS progress visualization...")
        nas.visualize_search_progress(os.path.join(results_dir, "nas_search_progress.png"))
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 NAS FAST DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   NAS-Optimized Parameters: {best_model.count_params():,}")
        print(f"   Model Size: {model_size:.2f} MB")
        print(f"   Test Accuracy: {accuracy:.4f}")
        
        print(f"\n🧬 NAS SEARCH METRICS:")
        print(f"   Search space size: {search_results['search_space_size']:,}")
        print(f"   Architectures evaluated: {search_results['population_size'] * search_results['generations']:,}")
        print(f"   Best fitness score: {best_architecture['fitness']:.4f}")
        
        print(f"\n📁 Files generated in '{results_dir}/':")
        print(f"   - nas_optimized_wireless_classifier.keras (model)")
        print(f"   - nas_confusion_matrix_absolute.png (absolute numbers)")
        print(f"   - nas_confusion_matrix_percentage.png (percentages)")
        print(f"   - nas_confusion_matrix_combined.png (combined)")
        print(f"   - nas_training_log.txt (detailed logs)")
        print(f"   - nas_results.json (complete results)")
        print(f"   - nas_search_progress.png (search visualization)")
        
        print(f"\n💡 NAS FAST DEMO SUCCESS!")
        print(f"   Parameter Count: {best_model.count_params():,}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Search Time: ~5-10 minutes (vs hours)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during NAS fast demo: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NAS search and training with configurable settings.")
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--eval-epochs", type=int, default=5, help="Epochs per architecture during NAS search.")
    parser.add_argument("--train-epochs", type=int, default=30, help="Epochs for final training.")
    parser.add_argument("--train-samples-per-class", type=int, default=1000)
    parser.add_argument("--val-samples-per-class", type=int, default=400)
    parser.add_argument("--test-samples-per-class", type=int, default=400)
    parser.add_argument("--results-dir", type=str, default="results_nas")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    success = main(args)
    if success:
        print("\n🚀 NAS Fast Demo completed successfully!")
    else:
        print("\n❌ NAS Fast Demo failed!")
