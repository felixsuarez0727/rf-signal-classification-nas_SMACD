import os
import glob
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

# ==========================
# 0️⃣ Definition of global classes
# ==========================
CLASSES = ["LTE", "DVB-T", "WiFi"]
PREFIXES = ["lte", "dvbt", "wf"]

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
# 2️⃣ Model construction
# ==========================
def build_model(input_len, num_classes):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(input_len, 2)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        LSTM(64),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# ==========================
# 3️⃣ Load dataset from folder (train/val/test)
# ==========================
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
    y_all = to_categorical(y_all, num_classes=len(CLASSES))

    return X_all, y_all

# ==========================
# 4️⃣ Incremental training per folder
# ==========================
def start_train(folder, model=None, model_path="cnn_lstm_iq_model.keras", chunk_samples=512, epochs=5, batch_size=32):
    base_path = os.path.join(folder)

    # Load datasets
    X_train, y_train = load_dataset(os.path.join(base_path, "train"), chunk_samples)
    X_val, y_val = load_dataset(os.path.join(base_path, "validation"), chunk_samples)
    X_test, y_test = load_dataset(os.path.join(base_path, "test"), chunk_samples)

    if X_train is None or X_val is None:
        print(f"⚠️ Not enough data to train for {folder}")
        return model

    # Shuffle train set
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    print(f"\n✅ Dataset for {folder} prepared:")
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape if X_test is not None else 'N/A'}")

    # Build model if not existing
    if model is None:
        if os.path.exists(model_path):
            print(f"📂 Loading existing model from {model_path}")
            model = load_model(model_path)
            model.summary()
        else:
            print("🆕 No saved model found, building a new one.")
            model = build_model(chunk_samples, len(CLASSES))
            model.summary()

    # Train on this city's data
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on test if available
    if X_test is not None:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"📊 Final evaluation on {folder} TEST: acc={test_acc:.4f}, loss={test_loss:.4f}")

    # Save model after training with this city
    model.save(model_path)
    print(f"💾 Updated model saved as {model_path}")

    return model

# ==========================
# 5️⃣ Example usage
# ==========================
if __name__ == "__main__":
    model = None    
    model = start_train("split_dataset", model=model, epochs=1, chunk_samples=1024)
