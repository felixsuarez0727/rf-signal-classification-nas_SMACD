#!/usr/bin/env python3
"""
Convert NAS model to Core ML for iOS
"""

import os
import sys

def convert_keras_to_coreml():
    """Convert Keras model to Core ML"""
    print("🔄 Starting Core ML conversion...")
    
    try:
        import coremltools as ct
        import tensorflow as tf
        import numpy as np
        
        print("✅ Dependencies loaded successfully")
        
        # Load the NAS model
        model_path = 'results_nas/nas_optimized_wireless_classifier.keras'
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            return False
            
        print("🔄 Loading NAS model...")
        model = tf.keras.models.load_model(model_path)
        
        print(f"✅ Model loaded: {model.count_params():,} parameters")
        
        # Convert to Core ML
        print("🔄 Converting to Core ML...")
        
        # Define input specification for IQ data (512 samples, 2 channels: real and imaginary)
        input_spec = ct.TensorType(
            shape=(1, 512, 2),  # batch, time_steps, features (real, imaginary)
            dtype=np.float32
        )
        
        # Convert model
        coreml_model = ct.convert(
            model,
            source="tensorflow",
            inputs=[input_spec],
            minimum_deployment_target=ct.target.iOS15
        )
        
        # Add metadata
        coreml_model.short_description = "NAS-Optimized Wireless Signal Classifier"
        coreml_model.author = "Jaime Arevalo"
        coreml_model.license = "MIT"
        coreml_model.version = "1.0"
        
        # Add input description
        coreml_model.input_description["input_1"] = "IQ signal data (512 samples, 2 channels: real, imaginary)"
        
        # Add output description
        coreml_model.output_description["Identity"] = "Classification probabilities for LTE, DVB-T, WiFi"
        
        # Save model
        output_path = "results_nas/nas_model.mlmodel"
        coreml_model.save(output_path)
        
        print(f"✅ Core ML model saved to: {output_path}")
        print(f"📊 Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
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
    success = convert_keras_to_coreml()
    if success:
        print("\n🎉 Core ML conversion completed successfully!")
        print("📱 Model ready for iOS deployment!")
    else:
        print("\n⚠️  Core ML conversion failed. Please check the errors above.")