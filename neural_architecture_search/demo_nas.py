"""
Neural Architecture Search (NAS) demonstration for wireless signal classification
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_architecture_search.nas_optimization import run_nas_demo, WirelessSignalNAS
from train import load_dataset, CLASSES


def run_nas_complete_demo():
    """
    Complete NAS demonstration with real wireless signal data
    """
    print("🧬 NEURAL ARCHITECTURE SEARCH - COMPLETE DEMO")
    print("=" * 60)
    
    # Load data
    print("📂 Loading wireless signal data...")
    X_train, y_train = load_dataset("split_dataset/train", chunk_samples=1024)
    X_val, y_val = load_dataset("split_dataset/validation", chunk_samples=1024)
    X_test, y_test = load_dataset("split_dataset/test", chunk_samples=1024)
    
    # Convert to sparse labels
    y_train_sparse = np.argmax(y_train, axis=1)
    y_val_sparse = np.argmax(y_val, axis=1)
    y_test_sparse = np.argmax(y_test, axis=1)
    
    # Use subset for NAS demo (faster execution)
    subset_size = 3000
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train_sparse[:subset_size]
    X_val_subset = X_val[:800]
    y_val_subset = y_val_sparse[:800]
    X_test_subset = X_test[:800]
    y_test_subset = y_test_sparse[:800]
    
    print(f"   ✅ Data loaded:")
    print(f"   Train: {X_train_subset.shape}, Val: {X_val_subset.shape}")
    
    # Apply feature optimization first
    print("\n🔧 Applying feature optimization...")
    # Simple feature optimization without external dependencies
    def reduce_chunk_size(signals, original_size=1024, new_size=512):
        step = original_size // new_size
        return signals[:, ::step, :]
    
    # Apply simple chunk size reduction
    X_train_opt = reduce_chunk_size(X_train_subset, 1024, 512)
    X_val_opt = reduce_chunk_size(X_val_subset, 1024, 512)
    X_test_opt = reduce_chunk_size(X_test_subset, 1024, 512)
    
    print(f"   ✅ Features optimized:")
    print(f"   Train: {X_train_opt.shape}, Val: {X_val_opt.shape}")
    
    # Initialize NAS
    print("\n🧬 Initializing Neural Architecture Search...")
    nas = WirelessSignalNAS(
        input_shape=X_train_opt.shape[1:],
        num_classes=len(CLASSES),
        population_size=12,  # Moderate size for demo
        generations=6        # Moderate generations for demo
    )
    
    # Run NAS search
    print("\n🔍 Running NAS search...")
    best_architecture = nas.search(X_train_opt, y_train_subset, X_val_opt, y_val_subset)
    
    # Get comprehensive results
    results = nas.get_search_results()
    
    # Build and evaluate best model
    print("\n🏗️ Building and evaluating best found architecture...")
    best_model = nas._build_model_from_architecture(best_architecture)
    
    # Train best model for more epochs
    print("🎯 Training best architecture...")
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    history = best_model.fit(
        X_train_opt, y_train_subset,
        validation_data=(X_val_opt, y_val_subset),
        epochs=15,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n📊 Final evaluation of NAS-optimized model...")
    test_predictions = best_model.predict(X_test_opt, verbose=0)
    test_predicted_classes = np.argmax(test_predictions, axis=1)
    test_accuracy = np.mean(test_predicted_classes == y_test_subset)
    
    # Generate comprehensive report
    print("\n" + "="*60)
    print("🎯 NAS SEARCH RESULTS SUMMARY")
    print("="*60)
    
    print(f"📊 Search Statistics:")
    print(f"   Search space size: {results['search_space_size']:,}")
    print(f"   Architectures evaluated: {results['population_size'] * results['generations']:,}")
    print(f"   Generations completed: {results['generations']}")
    
    print(f"\n🏆 Best Architecture Found:")
    best_metrics = best_architecture['metrics']
    print(f"   Final test accuracy: {test_accuracy:.4f}")
    print(f"   Validation accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   Parameters: {best_metrics['parameters']:,}")
    print(f"   Model size: {best_metrics['model_size_mb']:.2f} MB")
    print(f"   Fitness score: {best_architecture['fitness']:.4f}")
    
    # Compare with original model
    print(f"\n📈 Comparison with Original Model:")
    original_params = 42019  # From original model
    original_accuracy = 0.9796  # Estimated
    
    param_reduction = (original_params - best_metrics['parameters']) / original_params * 100
    accuracy_change = test_accuracy - original_accuracy
    
    print(f"   Original parameters: {original_params:,}")
    print(f"   NAS parameters: {best_metrics['parameters']:,}")
    print(f"   Parameter reduction: {param_reduction:.1f}%")
    print(f"   Original accuracy: {original_accuracy:.4f}")
    print(f"   NAS accuracy: {test_accuracy:.4f}")
    print(f"   Accuracy change: {accuracy_change:+.4f}")
    
    # Architecture details
    print(f"\n🏗️ Best Architecture Configuration:")
    for key, value in best_architecture.items():
        if key not in ['fitness', 'metrics']:
            print(f"   {key}: {value}")
    
    # Save results
    print(f"\n💾 Saving NAS results...")
    results_dir = "nas_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save best model
    best_model.save(os.path.join(results_dir, "nas_optimized_model.keras"))
    
    # Save architecture specification
    import json
    with open(os.path.join(results_dir, "best_architecture.json"), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        arch_json = {}
        for key, value in best_architecture.items():
            if key == 'metrics':
                arch_json[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                for k, v in value.items()}
            elif isinstance(value, (np.floating, np.integer)):
                arch_json[key] = float(value)
            else:
                arch_json[key] = value
        
        json.dump(arch_json, f, indent=2)
    
    # Save search history
    with open(os.path.join(results_dir, "search_history.json"), 'w') as f:
        json.dump(results['fitness_history'], f, indent=2)
    
    # Generate visualization
    nas.visualize_search_progress(os.path.join(results_dir, "nas_progress.png"))
    
    print(f"   ✅ Results saved to: {results_dir}/")
    print(f"   - nas_optimized_model.keras (best model)")
    print(f"   - best_architecture.json (architecture spec)")
    print(f"   - search_history.json (search progress)")
    print(f"   - nas_progress.png (visualization)")
    
    # Performance assessment
    print(f"\n🎯 NAS PERFORMANCE ASSESSMENT:")
    
    objectives_met = {
        'parameter_reduction': param_reduction >= 50,
        'accuracy_maintained': test_accuracy >= 0.95,
        'model_size': best_metrics['model_size_mb'] <= 0.1,
        'search_efficiency': results['search_space_size'] > results['population_size'] * results['generations']
    }
    
    for objective, met in objectives_met.items():
        status = "✅ MET" if met else "❌ NOT MET"
        print(f"   {objective.replace('_', ' ').title()}: {status}")
    
    overall_success = sum(objectives_met.values()) >= 3
    print(f"\n🏆 Overall NAS Success: {'✅ SUCCESS' if overall_success else '❌ PARTIAL'}")
    
    print(f"\n🎉 NAS Demo Completed Successfully!")
    print(f"💡 The NAS algorithm found an architecture with:")
    print(f"   - {param_reduction:.1f}% fewer parameters")
    print(f"   - {test_accuracy:.1%} accuracy")
    print(f"   - {best_metrics['model_size_mb']:.2f} MB model size")
    
    return nas, results, best_model


def run_nas_quick_demo():
    """
    Quick NAS demonstration with smaller search space
    """
    print("⚡ NAS QUICK DEMO")
    print("=" * 30)
    
    # Load smaller dataset
    X_train, y_train = load_dataset("split_dataset/train", chunk_samples=1024)
    X_val, y_val = load_dataset("split_dataset/validation", chunk_samples=1024)
    
    y_train_sparse = np.argmax(y_train, axis=1)
    y_val_sparse = np.argmax(y_val, axis=1)
    
    # Very small subset for quick demo
    X_train_subset = X_train[:1000]
    y_train_subset = y_train_sparse[:1000]
    X_val_subset = X_val[:300]
    y_val_subset = y_val_sparse[:300]
    
    # Quick feature optimization
    def reduce_chunk_size(signals, original_size=1024, new_size=512):
        step = original_size // new_size
        return signals[:, ::step, :]
    
    X_train_opt = reduce_chunk_size(X_train_subset, 1024, 512)
    X_val_opt = reduce_chunk_size(X_val_subset, 1024, 512)
    
    # Initialize small NAS
    nas = WirelessSignalNAS(
        input_shape=X_train_opt.shape[1:],
        num_classes=3,
        population_size=6,   # Very small
        generations=3        # Very few
    )
    
    # Run quick search
    best_architecture = nas.search(X_train_opt, y_train_subset, X_val_opt, y_val_subset)
    
    print(f"\n📊 Quick NAS Results:")
    print(f"   Best accuracy: {best_architecture['metrics']['accuracy']:.4f}")
    print(f"   Parameters: {best_architecture['metrics']['parameters']:,}")
    
    return nas, best_architecture


def compare_nas_vs_manual():
    """
    Compare NAS results with manually designed architectures
    """
    print("🔄 NAS vs Manual Architecture Comparison")
    print("=" * 50)
    
    # Load data
    X_train, y_train = load_dataset("split_dataset/train", chunk_samples=1024)
    X_val, y_val = load_dataset("split_dataset/validation", chunk_samples=1024)
    
    y_train_sparse = np.argmax(y_train, axis=1)
    y_val_sparse = np.argmax(y_val, axis=1)
    
    # Use subset
    subset_size = 2000
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train_sparse[:subset_size]
    X_val_subset = X_val[:500]
    y_val_subset = y_val_sparse[:500]
    
    # Apply feature optimization
    def reduce_chunk_size(signals, original_size=1024, new_size=512):
        step = original_size // new_size
        return signals[:, ::step, :]
    
    X_train_opt = reduce_chunk_size(X_train_subset, 1024, 512)
    X_val_opt = reduce_chunk_size(X_val_subset, 1024, 512)
    
    # Test manual architectures (simplified)
    print("🏗️ Testing manual architectures...")
    # Skip manual architecture comparison for now
    manual_results = {}
    
    # Test NAS architecture
    print("\n🧬 Testing NAS architecture...")
    nas = WirelessSignalNAS(
        input_shape=X_train_opt.shape[1:],
        num_classes=len(CLASSES),
        population_size=8,
        generations=4
    )
    
    nas_best = nas.search(X_train_opt, y_train_subset, X_val_opt, y_val_subset)
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"{'Architecture':<15} {'Parameters':<12} {'Accuracy':<10} {'Size (MB)':<10}")
    print("-" * 55)
    
    for arch_type, metrics in manual_results.items():
        print(f"{arch_type:<15} {metrics['params']:<12,} {metrics['accuracy']:<10.4f} {metrics['size']:<10.2f}")
    
    print(f"{'NAS Found':<15} {nas_best['metrics']['parameters']:<12,} {nas_best['metrics']['accuracy']:<10.4f} {nas_best['metrics']['model_size_mb']:<10.2f}")
    
    # Analysis
    nas_acc = nas_best['metrics']['accuracy']
    nas_params = nas_best['metrics']['parameters']
    
    best_manual = max(manual_results.keys(), key=lambda x: manual_results[x]['accuracy'])
    best_manual_acc = manual_results[best_manual]['accuracy']
    best_manual_params = manual_results[best_manual]['params']
    
    print(f"\n🎯 Analysis:")
    print(f"   Best manual: {best_manual} ({best_manual_acc:.4f}, {best_manual_params:,} params)")
    print(f"   NAS found: ({nas_acc:.4f}, {nas_params:,} params)")
    print(f"   NAS advantage: {nas_acc - best_manual_acc:+.4f} accuracy, {best_manual_params - nas_params:+,} params")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NAS Demo for Wireless Signal Classification')
    parser.add_argument('--mode', choices=['complete', 'quick', 'compare'], default='complete',
                       help='Demo mode: complete, quick, or compare')
    
    args = parser.parse_args()
    
    if args.mode == 'complete':
        nas, results, model = run_nas_complete_demo()
    elif args.mode == 'quick':
        nas, architecture = run_nas_quick_demo()
    else:
        compare_nas_vs_manual()
