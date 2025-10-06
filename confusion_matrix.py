import os
import glob
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==========================
# 0️⃣ Global definitions
# ==========================
CLASSES = ["LTE", "DVB-T", "WiFi"]
PREFIXES = ["lte", "dvbt", "wf"]
CHUNK_SAMPLES = 1024    
MODEL_PATH = "cnn_lstm_iq_model.keras"

# ==========================
# 1️⃣ Utility functions
# ==========================
def read_iq_file(filename):
    data = np.fromfile(filename, dtype=np.float32)
    return data[0::2] + 1j * data[1::2]

def normalize_iq(iq):
    return (iq - np.mean(iq)) / np.std(iq)

def chunks_from_iq(iq, chunk_samples):
    chunks_list = [iq[i:i+chunk_samples] for i in range(0, len(iq), chunk_samples)]
    return [np.column_stack((np.real(c), np.imag(c))) for c in chunks_list if len(c) == chunk_samples]

def load_dataset(base_folder, chunk_samples):
    X_all, y_all = [], []

    for class_name, prefix in zip(CLASSES, PREFIXES):
        pattern = os.path.join(base_folder, f"{prefix}*.bin")
        files = glob.glob(pattern)

        if not files:
            print(f"⚠️ No files found for class {class_name} in {base_folder}")
            continue

        class_index = CLASSES.index(class_name)

        for file in files:
            iq = normalize_iq(read_iq_file(file))
            X_chunks = chunks_from_iq(iq, chunk_samples)
            X_all.extend(X_chunks)
            y_all.extend([class_index] * len(X_chunks))

    if not X_all:
        return None, None

    X_all = np.array(X_all)
    y_all = np.array(y_all)  

    return X_all, y_all

# ==========================
# 2️⃣ Main
# ==========================
if __name__ == "__main__":
    # Load test dataset
    X_test, y_test = load_dataset("split_dataset/test", CHUNK_SAMPLES)
    if X_test is None:
        print("❌ No test data found.")
        exit()

    # Load model
    model = load_model(MODEL_PATH)

    # Predict
    y_pred = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes, labels=range(len(CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix (Absolute Counts)")
    plt.show()

    # ==========================
    # 🔹 Extra: show percentages per row
    # ==========================
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    print("\n📊 Confusion Matrix (percentages per true class):\n")
    for i, class_name in enumerate(CLASSES):
        row = " | ".join([f"{cm_percent[i, j]:6.2f}%" for j in range(len(CLASSES))])
        print(f"{class_name:>5} -> {row}")

    # Optional: plot percentages instead of absolute counts
    disp_percent = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=CLASSES)
    disp_percent.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title("Confusion Matrix (Percentages)")
    plt.show()
