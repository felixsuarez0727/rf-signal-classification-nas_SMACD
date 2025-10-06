import os
import glob
import numpy as np
from tensorflow.keras.models import load_model

# ==========================
# 0️⃣ Definition of global classes
# ==========================
CLASSES = ["LTE", "DVB-T", "WiFi"]
PREFIX_TO_CLASS = {
    "lte": "LTE",
    "dvbt": "DVB-T",
    "wf": "WiFi"
}

# ==========================
# 1️⃣ Utility functions
# ==========================
def read_iq_file(filename):
    data = np.fromfile(filename, dtype=np.float32)
    return data[0::2] + 1j * data[1::2]

def normalize_iq(iq):
    return (iq - np.mean(iq)) / np.std(iq)

def chunks_from_iq(iq, chunk_samples):
    """Split IQ into fixed-size chunks and convert to [re, im] matrix."""
    chunks_list = [iq[i:i+chunk_samples] for i in range(0, len(iq), chunk_samples)]
    return [np.column_stack((np.real(c), np.imag(c))) for c in chunks_list if len(c) == chunk_samples]

# ==========================
# 2️⃣ File classification
# ==========================
def classify_file(model, file_path, chunk_samples=512):
    iq = normalize_iq(read_iq_file(file_path))
    X_chunks = chunks_from_iq(iq, chunk_samples)

    if not X_chunks:
        return None, None

    X_chunks = np.array(X_chunks)
    preds = model.predict(X_chunks, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    # Majority vote
    counts = np.bincount(pred_labels, minlength=len(CLASSES))
    global_class = CLASSES[np.argmax(counts)]
    global_confidence = (np.max(counts) / len(pred_labels)) * 100

    return global_class, global_confidence, pred_labels

# ==========================
# 3️⃣ Classification of all files in a folder (with global accuracy by file and by chunks)
# ==========================
def classify_folder(model_path, folder_path, chunk_samples=512):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    files = glob.glob(os.path.join(folder_path, "*.bin"))
    if not files:
        print("⚠️ No .bin files found in the specified folder.")
        return

    print(f"\nClassifying {len(files)} files in '{folder_path}'...\n")
    print(f"{'File':<50} | {'Class':<10} | {'Accuracy (%)':>12}")
    print("-"*80)

    correct_count_files = 0
    total_count_files = 0

    correct_count_chunks = 0
    total_count_chunks = 0

    for file in files:
        class_label, confidence, pred_labels = classify_file(model, file, chunk_samples)
        file_name = os.path.basename(file)

        if class_label is None:
            class_label, confidence = "Error", 0
            print(f"{file_name:<50} | {class_label:<10} | {confidence:12.2f}")
            continue

        print(f"{file_name:<50} | {class_label:<10} | {confidence:12.2f}")

        # Infer true class from filename using prefix dictionary
        true_class = None
        file_name_lower = file_name.lower()
        for prefix, cls in PREFIX_TO_CLASS.items():
            if file_name_lower.startswith(prefix):
                true_class = cls
                break

        if true_class is not None:
            total_count_files += 1
            if class_label == true_class:
                correct_count_files += 1

            # Count chunk-level accuracy
            true_index = CLASSES.index(true_class)
            total_count_chunks += len(pred_labels)
            correct_count_chunks += np.sum(pred_labels == true_index)

    # Report global accuracy
    if total_count_files > 0:
        file_global_accuracy = (correct_count_files / total_count_files) * 100
        chunk_global_accuracy = (correct_count_chunks / total_count_chunks) * 100
        print(f"\n🌐 File-level global accuracy: {file_global_accuracy:.2f}% ({correct_count_files}/{total_count_files} correct)")
        print(f"📦 Chunk-level global accuracy: {chunk_global_accuracy:.2f}% ({correct_count_chunks}/{total_count_chunks} correct)")
    else:
        print("\n⚠️ Could not determine true classes for global accuracy.")

# ==========================
# 4️⃣ CLI usage
# ==========================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python test_iq_classifier_batch.py <model.keras> <test_folder>")
        sys.exit(1)

    model_path = sys.argv[1]
    folder_path = sys.argv[2]
    classify_folder(model_path, folder_path, chunk_samples=1024)
