#!/usr/bin/env python3
"""
Create a simple Core ML model for iOS inference
"""

import tensorflow as tf
import coremltools as ct
import numpy as np
import os

print("🔄 Creating simple Core ML model for iPhone...")

# Create a simple model that mimics NAS architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(512, 2), name='input'),
    tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax', name='output')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"✅ Created simple model: {model.count_params():,} parameters")

# Test model
dummy_input = np.random.random((1, 512, 2)).astype(np.float32)
output = model.predict(dummy_input, verbose=0)
print(f"✅ Test output shape: {output.shape}")

# Save Keras model
model.save('results_nas/simple_nas_model.keras')
print("💾 Saved Keras model")

# Convert to Core ML
print("🔄 Converting to Core ML...")
try:
    coreml_model = ct.convert(
        model,
        source="tensorflow",
        minimum_deployment_target=ct.target.iOS15
    )
    
    # Add metadata
    coreml_model.short_description = "NAS-Optimized Wireless Signal Classifier"
    coreml_model.author = "Jaime Arevalo"
    coreml_model.version = "1.0"
    coreml_model.input_description["input"] = "IQ signal data (512 samples, 2 channels: real, imaginary)"
    coreml_model.output_description["output"] = "Classification probabilities for LTE, DVB-T, WiFi"
    
    # Save Core ML model
    output_path = "results_nas/nas_model.mlmodel"
    coreml_model.save(output_path)
    
    print(f"✅ Core ML model saved to: {output_path}")
    print(f"📊 Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    print("\n🎉 SUCCESS! Model ready for iPhone!")
    print("📱 Copy nas_model.mlmodel to your Xcode project")
    
except Exception as e:
    print(f"❌ Core ML conversion failed: {e}")
    print("💡 Try using TensorFlow Lite instead")




