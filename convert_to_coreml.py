#!/usr/bin/env python3
"""
Convert NAS model to Core ML for iOS
"""

import os
import sys
import shutil
import argparse
import tempfile


def _path_size_mb(path: str) -> float:
    """Return file/dir size in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)

def convert_keras_to_coreml(model_path: str, output_path: str, deploy_to_app: str = ""):
    """Convert Keras model to Core ML"""
    print("🔄 Starting Core ML conversion...")
    
    try:
        import coremltools as ct
        import tensorflow as tf
        import numpy as np
        
        print("✅ Dependencies loaded successfully")
        
        # Load the NAS model
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            return False
            
        print("🔄 Loading NAS model...")
        model = tf.keras.models.load_model(model_path)
        
        print(f"✅ Model loaded: {model.count_params():,} parameters")
        
        # Convert to Core ML through SavedModel to avoid Keras 3 incompatibilities
        saved_model_dir = tempfile.mkdtemp(prefix="nas_savedmodel_")
        try:
            print("🔄 Exporting temporary SavedModel...")
            tf.saved_model.save(model, saved_model_dir)
            
            print("🔄 Converting SavedModel to Core ML...")
            # Define input specification for IQ data (512 samples, 2 channels: real and imaginary)
            input_spec = ct.TensorType(
                shape=(1, 512, 2),  # batch, time_steps, features (real, imaginary)
                dtype=np.float32
            )
            
            coreml_model = ct.convert(
                saved_model_dir,
                source="tensorflow",
                inputs=[input_spec],
                minimum_deployment_target=ct.target.iOS15
            )
        finally:
            shutil.rmtree(saved_model_dir, ignore_errors=True)
        
        # Add metadata
        coreml_model.short_description = "NAS-Optimized Wireless Signal Classifier"
        coreml_model.author = "Jaime Arevalo"
        coreml_model.license = "MIT"
        coreml_model.version = "1.0"
        
        # Add input/output descriptions with robust key handling
        try:
            input_keys = list(coreml_model.input_description._fd_spec.keys())
            output_keys = list(coreml_model.output_description._fd_spec.keys())
        except Exception:
            input_keys, output_keys = [], []
        if input_keys:
            coreml_model.input_description[input_keys[0]] = "IQ signal data (512 samples, 2 channels: real, imaginary)"
        if output_keys:
            coreml_model.output_description[output_keys[0]] = "Classification probabilities for LTE, DVB-T, WiFi"
        
        # Save model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        coreml_model.save(output_path)
        
        print(f"✅ Core ML model saved to: {output_path}")
        if os.path.exists(output_path):
            print(f"📊 Model size: {_path_size_mb(output_path):.2f} MB")

        # Optional deploy step: copy directly into app project
        if deploy_to_app:
            if os.path.exists(deploy_to_app):
                if os.path.isdir(deploy_to_app):
                    shutil.rmtree(deploy_to_app)
                else:
                    os.remove(deploy_to_app)
            shutil.copytree(output_path, deploy_to_app)
            print(f"📱 Deployed Core ML package to app: {deploy_to_app}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Installing required packages...")
        
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "coremltools", "tensorflow"])
            print("✅ Packages installed. Please run the script again.")
            return False
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras NAS model to Core ML.")
    parser.add_argument(
        "--model-path",
        default="results_nas_highacc_v1/nas_optimized_wireless_classifier.keras",
        help="Path to .keras model."
    )
    parser.add_argument(
        "--output-path",
        default="results_nas_highacc_v1/nas_model.mlpackage",
        help="Output path for Core ML package (.mlpackage)."
    )
    parser.add_argument(
        "--deploy-to-app",
        default="",
        help="Optional target app path for nas_model.mlpackage."
    )
    args = parser.parse_args()

    success = convert_keras_to_coreml(args.model_path, args.output_path, args.deploy_to_app)
    if success:
        print("\n🎉 Core ML conversion completed successfully!")
        print("📱 Model ready for iOS deployment!")
    else:
        print("\n⚠️  Core ML conversion failed. Please check the errors above.")