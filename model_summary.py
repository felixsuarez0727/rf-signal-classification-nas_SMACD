import sys
from tensorflow.keras.models import load_model

if len(sys.argv) < 2:
    print("Usage: python model_summary.py <model_path.keras>")
    sys.exit(1)

model_path = sys.argv[1]

# Load model
model = load_model(model_path)

# Show summary
model.summary()