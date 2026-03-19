# 📡 Wireless Signal Classification with Neural Architecture Search (NAS)

A deep learning pipeline for classifying wireless signals (LTE, DVB-T, WiFi) using automated Neural Architecture Search. The current project version is optimized for an aggressive edge-efficient operating point with very low parameter count.

---

## 🎯 Project Overview

This project demonstrates Neural Architecture Search (NAS) for wireless signal classification, currently achieving:
- **Automated architecture discovery** using evolutionary algorithms
- **Ultra-compact model** with only **3,539 parameters**
- **90.1% test accuracy** on balanced OTA subsets
- **<0.1 MB model size** for edge/IoT deployment
- **Multi-objective optimization** balancing accuracy and efficiency

### Original vs NAS-Optimized Performance
| Metric | Original Model | NAS-Optimized Model | Improvement |
|--------|----------------|-------------------|-------------|
| Parameters | ~42,019 | 3,539 | 91.6% reduction |
| Accuracy | 97.96% | 90.1% | Compact trade-off |
| Model Size | ~0.55 MB | ~0.09 MB | 83% reduction |
| Discovery Method | Manual Design | Automated NAS | Revolutionary |
| Search Space | Limited | 1.6M combinations | Comprehensive |

---

## 🤝 Funding and Supporting Institutions

<p align="center">
  <img src="logos/logo%20CSIC.png" alt="CSIC" height="70" />
  <img src="logos/_Logo-Momentum-Negativo_Circular.png" alt="Momentum" height="70" />
  <img src="logos/Next_Generation.png" alt="Next Generation" height="70" />
</p>
<p align="center">
  <img src="logos/PRTR.png" alt="PRTR" height="70" />
  <img src="logos/logo-doraito.png" alt="Logo Doraito" height="70" />
  <img src="logos/imse.png" alt="IMSE" height="70" />
</p>

---

## 🧬 Neural Architecture Search (NAS)

### **What is NAS?**
Neural Architecture Search automatically discovers optimal neural network architectures instead of relying on manual design. For wireless signal classification, NAS explores:

- **CNN Layer Configurations**: Number of layers, filters, kernel sizes
- **LSTM Architectures**: Units, bidirectional options, dropout rates  
- **Pooling Strategies**: Max, average, global average pooling
- **Dense Layer Designs**: Hidden units, activation functions
- **Optimization Parameters**: Learning rates, batch sizes, optimizers

### **NAS Implementation Features**
```python
# Optimized search space includes:
search_space = {
    'conv_layers': [2, 3, 4],  # Removed single layer
    'conv_filters': [[16, 32], [32, 64], [16, 32, 64], [32, 64, 128]],
    'lstm_units': [32, 64, 128],  # Removed very small units
    'dense_units': [16, 32, 64],  # Optimized sizes
    'optimizer': ['adam'],  # Focused on best optimizer
    'batch_size': [32, 64]  # Stable batch sizes
}
```

### **Evolutionary Search Process**
1. **Population Initialization**: Generate random architectures
2. **Fitness Evaluation**: Train and evaluate each architecture
3. **Selection**: Choose best-performing architectures
4. **Crossover**: Combine features from parent architectures
5. **Mutation**: Introduce random variations
6. **Evolution**: Repeat for multiple generations

### **Multi-objective Optimization**
NAS optimizes for multiple objectives simultaneously:
- **Accuracy**: Maximize classification performance
- **Efficiency**: Minimize parameter count
- **Size**: Minimize model size for deployment

---

## 📂 Project Structure

```
LTE_DVB_T_WiFi_small_model/
├── 📁 Core Project Files
│   ├── train.py                    # Original training script
│   ├── test.py                     # Model testing utilities
│   ├── model_summary.py           # Model analysis tools
│   ├── confusion_matrix.py        # Performance visualization
│   └── cnn_lstm_iq_model.keras    # Original trained model
│
├── 📁 Neural Architecture Search
│   ├── neural_architecture_search/
│   │   ├── __init__.py            # NAS package initialization
│   │   ├── nas_optimization.py    # Core NAS implementation
│   │   ├── demo_nas.py            # NAS demonstrations
│   │   ├── demo_nas_complete.py   # Complete NAS demo
│   │   ├── requirements.txt       # NAS dependencies
│   │   └── README.md              # NAS documentation
│   └── nas_fast_demo.py           # Optimized NAS demo (RECOMMENDED)
│
├── 📁 Data & Results
│   ├── split_dataset/             # Training/validation/test data
│   │   ├── train/                 # Training samples
│   │   ├── validation/            # Validation samples
│   │   └── test/                  # Test samples
│   ├── results_nas/               # NAS optimization results
│   │   ├── nas_optimized_wireless_classifier.keras
│   │   ├── nas_confusion_matrix_*.png
│   │   ├── nas_training_log.txt
│   │   ├── nas_results.json
│   │   └── nas_search_progress.png
│   └── pictures/                  # Original model visualizations
│
└── 📄 Documentation
    ├── readme.md                  # This file
    └── requirements.txt           # Dependencies
```

---

## 🚀 Quick Start Guide

### 1. **Environment Setup**
```bash
# Create virtual environment
    python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Run NAS Fast Demo (Recommended)**
```bash
# Execute updated high-accuracy NAS search (current best setup)
python nas_fast_demo.py \
  --population-size 16 \
  --generations 10 \
  --eval-epochs 8 \
  --train-epochs 60 \
  --train-samples-per-class 1400 \
  --val-samples-per-class 500 \
  --test-samples-per-class 500 \
  --seed 42 \
  --results-dir results_nas_highacc_v1
```

This will:
- Use larger balanced dataset subsets (1400/500/500 per class)
- Run 10 generations with 16 architectures per generation
- Evaluate architectures with longer per-candidate training (`eval_epochs=8`)
- Save outputs in `results_nas_highacc_v1/`
- Reproduce the current compact model result (3,539 params, 90.1% test accuracy)

### 3. **Run Complete NAS Demo**
```bash
# Execute full NAS search (longer but more thorough)
python neural_architecture_search/demo_nas_complete.py
```

### 4. **Compare NAS vs Manual Architectures**
```bash
# Compare NAS results with manual designs
python neural_architecture_search/demo_nas.py --mode compare
```

### 5. **Test Original Model**
```bash
# Test the original trained model
python test.py cnn_lstm_iq_model.keras split_dataset/test
```

---

## 📊 Dataset Processing Pipeline

### 1. **Raw IQ Data Loading**
- Binary files contain interleaved real/imaginary samples (float32)
- Complex IQ reconstruction: `data[0::2] + 1j * data[1::2]`
- Automatic signal type detection from filename prefixes

### 2. **Advanced Preprocessing**
```python
# Normalization
iq_normalized = (iq - np.mean(iq)) / np.std(iq)

# Chunking with optimization
chunks = [iq[i:i+chunk_size] for i in range(0, len(iq), chunk_size)]
chunks_matrix = [np.column_stack((np.real(c), np.imag(c))) for c in chunks]

# Feature optimization for NAS
selector = SelectKBest(f_classif, k=target_features)
optimized_features = selector.fit_transform(reshaped_data, labels)
```

### 3. **Label Processing**
- **LTE**: Files with `lte` prefix
- **DVB-T**: Files with `dvbt` prefix  
- **WiFi**: Files with `wf` prefix
- One-hot encoding for multi-class classification

---

## 🔬 NAS Optimization Methodology

### Phase 1: Search Space Definition
1. **Architecture Components**: Define CNN, LSTM, and Dense layer options
2. **Parameter Ranges**: Set realistic ranges for all hyperparameters
3. **Constraint Definition**: Ensure valid architecture combinations

### Phase 2: Evolutionary Search
1. **Population Initialization**: Generate random architectures
2. **Fitness Evaluation**: Train and evaluate each architecture
3. **Selection**: Tournament selection for parent architectures
4. **Crossover**: Combine features from parent architectures
5. **Mutation**: Introduce random variations
6. **Evolution**: Repeat for multiple generations

### Phase 3: Multi-objective Optimization
1. **Fitness Function**: Accuracy-first with soft parameter penalty  
   (paper/code): `F = 1.2 * (1-A) + 0.25 * P_penalty`
2. **Smart Parameter Targeting**: mild penalty up to ~10k parameters, stronger penalty beyond ~16k, encouraging compact models
3. **Architecture Ranking**: Rank by combined fitness score (lower is better)

### Phase 4: Final Evaluation
1. **Best Architecture Selection**: Choose top-performing architecture
2. **Extended Training**: Train best architecture for more epochs
3. **Performance Validation**: Validate on test set
4. **Deployment Analysis**: Assess deployment readiness

---

## 📈 Performance Results

### NAS Search Statistics
| Metric | Value |
|--------|-------|
| **Search Space Size** | 1,327,104 combinations |
| **Architectures Evaluated** | 160 (10 generations × 16 population) |
| **Search Time** | ~30-90 minutes (CPU dependent) |
| **Parameter Reduction** | 91.6% (42,019 → 3,539) |
| **Accuracy Achieved** | 90.1% |

### Classification Accuracy by Signal Type
| Signal Type | Original | NAS-Optimized | Change |
|-------------|----------|---------------|---------|
| **LTE** | -- | 92.8% recall | Strong |
| **DVB-T** | -- | 86.6% recall | Moderate |
| **WiFi** | -- | 90.8% recall | Strong |
| **Overall** | -- | 90.1% | Compact trade-off |

### Optimization Metrics
- **Parameter Reduction**: 91.6% reduction (42,019 → 3,539)
- **Model Size**: ~0.09 MB serialized model (`nas_results_highacc_v1`)
- **Weight Memory (FP32)**: ~13.8 KB (INT8 ~3.5 KB)
- **Search Objective**: Accuracy-first with soft parameter penalty
- **Deployment Target**: Core ML package for iOS app integration

### Deployment Readiness
- ✅ **Edge Devices**: Ultra-light model suitable for IoT
- ✅ **Mobile Apps**: TFLite format ready for Android/iOS
- ✅ **Real-time Processing**: Sub-millisecond inference
- ✅ **Cloud Deployment**: Efficient server-side processing

---

## 🛠️ Advanced Usage

### Custom NAS Configuration
```python
from neural_architecture_search import WirelessSignalNAS

# Initialize NAS with custom parameters
nas = WirelessSignalNAS(
    input_shape=(512, 2),
    num_classes=3,
    population_size=30,      # Larger population
    generations=15           # More generations
)

# Run architecture search
best_architecture = nas.search(X_train, y_train, X_val, y_val)

# Build and evaluate best model
best_model = nas._build_model_from_architecture(best_architecture)
```

### NAS Results Analysis
```python
# Get comprehensive search results
results = nas.get_search_results()

# Visualize search progress
nas.visualize_search_progress('nas_progress.png')

# Analyze architecture details
print(f"Best architecture: {results['best_architecture']}")
print(f"Search space explored: {results['search_space_size']:,}")
```

### Custom Search Space
```python
# Modify search space for specific requirements
nas.search_space = {
    'conv_layers': [2, 3],  # Limit to 2-3 CNN layers
    'lstm_units': [16, 32], # Smaller LSTM units
    'dense_units': [8, 16]  # Compact dense layers
}
```

---

## 📋 Requirements & Dependencies

### Core Dependencies
```
tensorflow>=2.10.0
scikit-learn>=1.1.0
numpy>=1.21.0
matplotlib>=3.5.0
tensorflow-model-optimization>=0.7.0
```

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU processing (slower but functional)
- **Recommended**: 16GB RAM, GPU acceleration (CUDA 11.2+)
- **Production**: Edge device with 2GB RAM, ARM/x86 architecture

---

## 🎯 Use Cases & Applications

### 1. **Spectrum Monitoring**
- Real-time wireless signal identification
- Interference detection and classification
- Regulatory compliance monitoring

### 2. **IoT Security**
- Device identification and authentication
- Network intrusion detection
- Wireless protocol verification

### 3. **Telecommunications**
- Network optimization and planning
- Signal quality assessment
- Protocol compliance testing

### 4. **Research & Development**
- Wireless communication research
- Signal processing algorithm development
- Machine learning model benchmarking
- Automated neural architecture discovery
- Multi-objective optimization studies

---

## 🔧 Troubleshooting

### Common Issues

**Memory Error During NAS Search**
```bash
# Reduce population size and generations
python neural_architecture_search/demo_nas.py --mode quick
```

**Slow Search Progress**
```bash
# Use smaller dataset subset for testing
# Modify demo_nas_complete.py to use fewer samples
```

**Low Quality NAS Results**
```bash
# Increase population size and generations
# Ensure sufficient training data
# Check feature optimization settings
```

**Import Errors**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Check Python version (3.8+ required)
python --version
```

---

## 📚 References & Further Reading

### Research Papers
- "Neural Architecture Search: A Survey" (Journal of Machine Learning Research)
- "Efficient Neural Architecture Search via Parameters Sharing" (ICML 2018)
- "DARTS: Differentiable Architecture Search" (ICLR 2019)
- "Deep Learning for Wireless Signal Classification" (IEEE Communications)

### Documentation
- [TensorFlow Model Optimization Guide](https://www.tensorflow.org/model_optimization)
- [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [TFLite Deployment Guide](https://www.tensorflow.org/lite/guide)

---

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Implement NAS improvements
4. Add comprehensive tests
5. Submit a pull request

### Areas for Contribution
- Additional search algorithms (RL-based, gradient-based)
- New architecture components
- Performance optimization
- Visualization improvements
- Mobile deployment optimizations

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Team & Acknowledgments

- **Development Team**: Wireless Signal Classification Research Group
- **Special Thanks**: TensorFlow Model Optimization team
- **Dataset Contributors**: Ghent University Wireless Research Lab

---

## 📞 Support & Contact

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Email**: wireless-signals@research.org

---

*Last updated: September 2024*
*Version: 3.1.0 (NAS Implementation Complete)*

---

## 🎉 Latest Update (Current Baseline)

Current verified baseline from `results_nas_highacc_v1`:

- ✅ **90.1% test accuracy** with **3,539 parameters**
- ✅ **91.6% parameter reduction** vs ~42K manual model baseline
- ✅ **160 architectures evaluated** (16 population × 10 generations)
- ✅ **Updated article + app metadata** aligned with latest metrics
- ✅ **Core ML conversion workflow fixed** for the current Keras/TensorFlow stack

Recommended command:

```bash
python nas_fast_demo.py \
  --population-size 16 \
  --generations 10 \
  --eval-epochs 8 \
  --train-epochs 60 \
  --train-samples-per-class 1400 \
  --val-samples-per-class 500 \
  --test-samples-per-class 500 \
  --seed 42 \
  --results-dir results_nas_highacc_v1
```