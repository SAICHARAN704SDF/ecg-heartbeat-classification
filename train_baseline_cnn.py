import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

from config import (
    TRAIN_FILE, TEST_FILE, RANDOM_SEED, NUM_CLASSES,
    SIGNAL_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    VAL_SIZE, MODEL_DIR, RESULTS_DIR
)

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("=" * 70)
print("MIT-BIH BASELINE 1D CNN")
print("=" * 70)

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv(TRAIN_FILE, header=None)
test_df = pd.read_csv(TEST_FILE, header=None)

X = train_df.iloc[:, :SIGNAL_LENGTH].values.astype("float32")
y = train_df.iloc[:, SIGNAL_LENGTH].values.astype("int64")

X_test = test_df.iloc[:, :SIGNAL_LENGTH].values.astype("float32")
y_test = test_df.iloc[:, SIGNAL_LENGTH].values.astype("int64")

print("Training samples:", len(X))
print("Test samples:", len(X_test))
print("Signal length:", X.shape[1])
print("Classes:", np.unique(y))

# -----------------------------
# Stratified validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

# CNN input: (samples, timesteps, channels)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("Train split:", X_train.shape)
print("Validation split:", X_val.shape)
print("Test split:", X_test.shape)

# -----------------------------
# Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SIGNAL_LENGTH, 1)),

    tf.keras.layers.Conv1D(32, 7, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "baseline_cnn.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Evaluation
# -----------------------------
probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
pred = np.argmax(probs, axis=1)

accuracy = accuracy_score(y_test, pred)
balanced_acc = balanced_accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
weighted_f1 = f1_score(y_test, pred, average="weighted")
cm = confusion_matrix(y_test, pred)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Macro-F1:          {macro_f1:.4f}")
print(f"Weighted-F1:       {weighted_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    pred,
    target_names=[
        "Normal",
        "Supraventricular",
        "Ventricular",
        "Fusion",
        "Unknown"
    ],
    digits=4,
    zero_division=0
))

print("Confusion Matrix:")
print(cm)

# -----------------------------
# Save results
# -----------------------------
results = {
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_acc),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        y_test, pred, output_dict=True, zero_division=0
    ),
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "epochs_requested": EPOCHS,
    "random_seed": RANDOM_SEED
}

with open(RESULTS_DIR / "baseline_cnn_results.json", "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame(history.history).to_csv(
    RESULTS_DIR / "baseline_cnn_history.csv",
    index=False
)

np.save(RESULTS_DIR / "baseline_cnn_predictions.npy", pred)
np.save(RESULTS_DIR / "baseline_cnn_probabilities.npy", probs)

print("\nSaved:")
print("  models/baseline_cnn.keras")
print("  results/baseline_cnn_results.json")
print("  results/baseline_cnn_history.csv")
print("  results/baseline_cnn_predictions.npy")
print("  results/baseline_cnn_probabilities.npy")
