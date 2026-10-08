"""
Experiment 2: Multi-Representation ECG Fusion
Raw ECG branch + Wavelet branch -> feature fusion -> 5-class classifier

Research hypothesis:
Raw ECG captures waveform morphology in the time domain.
Wavelet coefficients provide complementary multi-resolution information.
The fusion model tests whether combining both representations improves
minority-class performance over the locked Raw 1D CNN baseline.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt

from config import (
    TRAIN_FILE,
    TEST_FILE,
    RANDOM_SEED,
    NUM_CLASSES,
    SIGNAL_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    VAL_SIZE,
    MODEL_DIR,
    RESULTS_DIR,
)

# ============================================================
# EXPERIMENT CONFIG
# ============================================================

PROJECT_NAME = "ecg-mitbih-research"
RUN_NAME = "raw-wavelet-fusion-v1"

CLASS_NAMES = [
    "Normal",
    "Supraventricular",
    "Ventricular",
    "Fusion",
    "Unknown",
]

WAVELET = "db4"
WAVELET_LEVEL = 3

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# DIRECTORIES
# ============================================================

MODEL_DIR = Path(MODEL_DIR)
RESULTS_DIR = Path(RESULTS_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# W&B
# ============================================================

run = wandb.init(
    project=PROJECT_NAME,
    name=RUN_NAME,
    config={
        "experiment": "Raw + Wavelet Feature Fusion",
        "dataset": "MIT-BIH ECG Heartbeat Categorization",
        "raw_representation": "187-point ECG waveform",
        "wavelet": WAVELET,
        "wavelet_level": WAVELET_LEVEL,
        "architecture": "Dual-branch CNN feature fusion",
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "validation_size": VAL_SIZE,
        "optimizer": "Adam",
        "loss": "Sparse Categorical Crossentropy",
        "selection_metric": "Validation Macro-F1",
        "random_seed": RANDOM_SEED,
    },
)

print("=" * 78)
print("EXPERIMENT 2 — RAW + WAVELET ECG FUSION")
print("=" * 78)

# ============================================================
# WAVELET TRANSFORM
# ============================================================

def wavelet_features(signal):
    """
    Decompose one ECG beat using Daubechies-4 wavelet.

    Output order:
        approximation at level 3
        detail level 3
        detail level 2
        detail level 1

    Concatenation preserves the multi-resolution coefficient
    representation as a 1D sequence.
    """
    coeffs = pywt.wavedec(
        signal,
        WAVELET,
        level=WAVELET_LEVEL,
        mode="symmetric",
    )

    features = np.concatenate(coeffs).astype(np.float32)

    # Per-beat standardization of the wavelet representation.
    # Small epsilon prevents division by zero.
    mean = features.mean()
    std = features.std()

    if std > 1e-8:
        features = (features - mean) / std
    else:
        features = features - mean

    return features


def make_wavelet_dataset(X):
    """Convert [N, 187] raw ECG data to [N, wavelet_length]."""
    print("\nComputing wavelet representation...")
    features = np.stack(
        [wavelet_features(signal) for signal in X]
    )

    print(f"Wavelet shape: {features.shape}")
    return features


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE, header=None)
test_df = pd.read_csv(TEST_FILE, header=None)

X = train_df.iloc[:, :SIGNAL_LENGTH].values.astype("float32")
y = train_df.iloc[:, SIGNAL_LENGTH].values.astype("int64")

X_test = test_df.iloc[:, :SIGNAL_LENGTH].values.astype("float32")
y_test = test_df.iloc[:, SIGNAL_LENGTH].values.astype("int64")

print(f"Training samples: {len(X)}")
print(f"Test samples:     {len(X_test)}")
print(f"Raw signal:       {X.shape}")

# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,
)

print(f"Train split:      {X_train.shape}")
print(f"Validation split: {X_val.shape}")
print(f"Test split:       {X_test.shape}")

# ============================================================
# CREATE WAVELET REPRESENTATIONS
# ============================================================

W_train = make_wavelet_dataset(X_train)
W_val = make_wavelet_dataset(X_val)
W_test = make_wavelet_dataset(X_test)

# Add channel dimension for Conv1D
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

W_train = W_train[..., np.newaxis]
W_val = W_val[..., np.newaxis]
W_test = W_test[..., np.newaxis]

print(f"\nRaw input shape:      {X_train.shape}")
print(f"Wavelet input shape:  {W_train.shape}")

# ============================================================
# MODEL BRANCH
# ============================================================

def cnn_branch(input_shape, prefix):
    """
    Same feature extractor structure for both representations.
    Keeping the branches structurally comparable makes the
    fusion experiment easier to interpret.
    """
    inputs = tf.keras.Input(shape=input_shape, name=f"{prefix}_input")

    x = tf.keras.layers.Conv1D(
        32, 7, padding="same", activation="relu",
        name=f"{prefix}_conv1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(
        name=f"{prefix}_bn1"
    )(x)
    x = tf.keras.layers.MaxPooling1D(
        2, name=f"{prefix}_pool1"
    )(x)

    x = tf.keras.layers.Conv1D(
        64, 5, padding="same", activation="relu",
        name=f"{prefix}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name=f"{prefix}_bn2"
    )(x)
    x = tf.keras.layers.MaxPooling1D(
        2, name=f"{prefix}_pool2"
    )(x)

    x = tf.keras.layers.Conv1D(
        128, 3, padding="same", activation="relu",
        name=f"{prefix}_conv3",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name=f"{prefix}_bn3"
    )(x)

    x = tf.keras.layers.GlobalAveragePooling1D(
        name=f"{prefix}_gap"
    )(x)

    return inputs, x


raw_input, raw_features = cnn_branch(
    (SIGNAL_LENGTH, 1),
    "raw",
)

wavelet_input, wavelet_features_out = cnn_branch(
    (W_train.shape[1], 1),
    "wavelet",
)

# ============================================================
# FEATURE FUSION
# ============================================================

fused = tf.keras.layers.Concatenate(
    name="feature_fusion"
)([
    raw_features,
    wavelet_features_out,
])

fused = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="fusion_dense",
)(fused)

fused = tf.keras.layers.Dropout(
    0.4,
    name="fusion_dropout",
)(fused)

output = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="classifier",
)(fused)

model = tf.keras.Model(
    inputs=[raw_input, wavelet_input],
    outputs=output,
    name="RawWaveletFusionCNN",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print(f"\nTotal parameters: {model.count_params():,}")

# ============================================================
# VALIDATION METRICS CALLBACK
# ============================================================

class ValidationMetricsCallback(tf.keras.callbacks.Callback):

    def __init__(self, raw_val, wavelet_val, y_val):
        super().__init__()
        self.raw_val = raw_val
        self.wavelet_val = wavelet_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        probabilities = self.model.predict(
            [self.raw_val, self.wavelet_val],
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        predictions = np.argmax(probabilities, axis=1)

        macro_f1 = f1_score(
            self.y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            self.y_val,
            predictions,
            average="weighted",
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            self.y_val,
            predictions,
        )

        macro_precision = precision_score(
            self.y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            self.y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        logs["val_macro_f1"] = float(macro_f1)
        logs["val_weighted_f1"] = float(weighted_f1)
        logs["val_balanced_accuracy"] = float(balanced_acc)
        logs["val_macro_precision"] = float(macro_precision)
        logs["val_macro_recall"] = float(macro_recall)

        per_class = {}

        for class_id, class_name in enumerate(CLASS_NAMES):
            class_report = classification_report(
                self.y_val,
                predictions,
                labels=[class_id],
                target_names=[class_name],
                output_dict=True,
                zero_division=0,
            )[class_name]

            class_f1 = class_report["f1-score"]
            per_class[f"val_f1/{class_name}"] = float(class_f1)
            logs[f"val_f1_class_{class_id}"] = float(class_f1)

        wandb.log({
            "epoch": epoch + 1,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1,
            "val_balanced_accuracy": balanced_acc,
            "val_macro_precision": macro_precision,
            "val_macro_recall": macro_recall,
            **per_class,
        })

        print(
            f"\nEpoch {epoch + 1:02d} | "
            f"Val Macro-F1: {macro_f1:.4f} | "
            f"Val Balanced Acc: {balanced_acc:.4f}"
        )


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_path = (
    MODEL_DIR /
    "raw_wavelet_fusion_best_macro_f1.keras"
)

callbacks = [
    ValidationMetricsCallback(
        X_val,
        W_val,
        y_val,
    ),

    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_macro_f1",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_macro_f1",
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    [X_train, W_train],
    y_train,
    validation_data=(
        [X_val, W_val],
        y_val,
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\nLoading best validation Macro-F1 checkpoint...")
model = tf.keras.models.load_model(checkpoint_path)

# ============================================================
# FINAL TEST
# ============================================================

probabilities = model.predict(
    [X_test, W_test],
    batch_size=BATCH_SIZE,
    verbose=1,
)

predictions = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, predictions)
balanced_acc = balanced_accuracy_score(y_test, predictions)

macro_precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

cm = confusion_matrix(y_test, predictions)

report_text = classification_report(
    y_test,
    predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

report_dict = classification_report(
    y_test,
    predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

print("\n" + "=" * 78)
print("FINAL TEST RESULTS — RAW + WAVELET FUSION")
print("=" * 78)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")
print(f"Macro-F1:          {macro_f1:.4f}")
print(f"Weighted-F1:       {weighted_f1:.4f}")

print("\nClassification Report:")
print(report_text)

print("Confusion Matrix:")
print(cm)

# ============================================================
# W&B CONFUSION MATRIX
# ============================================================

wandb_cm = wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_test,
    preds=predictions,
    class_names=CLASS_NAMES,
    title="Raw + Wavelet Fusion — Test Confusion Matrix",
)

wandb.log({
    "test/confusion_matrix": wandb_cm,
})

# ============================================================
# W&B PER-CLASS TABLE
# ============================================================

metric_table = wandb.Table(
    columns=[
        "class",
        "precision",
        "recall",
        "f1",
        "support",
    ]
)

for class_name in CLASS_NAMES:
    row = report_dict[class_name]

    metric_table.add_data(
        class_name,
        row["precision"],
        row["recall"],
        row["f1-score"],
        int(row["support"]),
    )

wandb.log({
    "test/per_class_metrics": metric_table,
})

# ============================================================
# LOCAL TRAINING CURVE
# ============================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    RESULTS_DIR /
    "raw_wavelet_fusion_history.csv",
    index=False,
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 9),
)

axes[0].plot(
    history_df["loss"],
    label="Training Loss",
)

axes[0].plot(
    history_df["val_loss"],
    label="Validation Loss",
)

axes[0].set_title(
    "Raw + Wavelet Fusion — Loss"
)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    history_df["accuracy"],
    label="Training Accuracy",
)

axes[1].plot(
    history_df["val_accuracy"],
    label="Validation Accuracy",
)

if "val_macro_f1" in history_df:
    axes[1].plot(
        history_df["val_macro_f1"],
        label="Validation Macro-F1",
    )

axes[1].set_title(
    "Raw + Wavelet Fusion — Accuracy / Macro-F1"
)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

training_plot = (
    RESULTS_DIR /
    "raw_wavelet_fusion_training_curves.png"
)

plt.savefig(
    training_plot,
    dpi=180,
)

wandb.log({
    "plots/training_curves": wandb.Image(
        str(training_plot)
    )
})

plt.close()

# ============================================================
# PER-CLASS F1 PLOT
# ============================================================

class_f1 = [
    report_dict[name]["f1-score"]
    for name in CLASS_NAMES
]

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    CLASS_NAMES,
    class_f1,
)

ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.set_title(
    "Raw + Wavelet Fusion — Test Per-Class F1"
)
ax.grid(
    axis="y",
    alpha=0.3,
)

for i, value in enumerate(class_f1):
    ax.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
    )

plt.tight_layout()

f1_plot = (
    RESULTS_DIR /
    "raw_wavelet_fusion_per_class_f1.png"
)

plt.savefig(
    f1_plot,
    dpi=180,
)

wandb.log({
    "plots/per_class_f1": wandb.Image(
        str(f1_plot)
    )
})

plt.close()

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "experiment": RUN_NAME,
    "representation": {
        "raw": True,
        "wavelet": WAVELET,
        "wavelet_level": WAVELET_LEVEL,
    },
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_acc),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "parameters": int(model.count_params()),
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "epochs_requested": int(EPOCHS),
    "epochs_completed": int(len(history.history["loss"])),
    "confusion_matrix": cm.tolist(),
    "classification_report": report_dict,
    "random_seed": RANDOM_SEED,
}

with open(
    RESULTS_DIR /
    "raw_wavelet_fusion_results.json",
    "w",
) as f:
    json.dump(
        results,
        f,
        indent=2,
    )

np.save(
    RESULTS_DIR /
    "raw_wavelet_fusion_predictions.npy",
    predictions,
)

np.save(
    RESULTS_DIR /
    "raw_wavelet_fusion_probabilities.npy",
    probabilities,
)

# ============================================================
# W&B SUMMARY
# ============================================================

wandb.summary["test_accuracy"] = accuracy
wandb.summary["test_balanced_accuracy"] = balanced_acc
wandb.summary["test_macro_precision"] = macro_precision
wandb.summary["test_macro_recall"] = macro_recall
wandb.summary["test_macro_f1"] = macro_f1
wandb.summary["test_weighted_f1"] = weighted_f1
wandb.summary["parameters"] = model.count_params()

wandb.finish()

print("\n" + "=" * 78)
print("RAW + WAVELET FUSION EXPERIMENT COMPLETE")
print("=" * 78)
print(f"Best checkpoint: {checkpoint_path}")
print(
    f"Completed epochs: "
    f"{len(history.history['loss'])}"
)
print("Results saved locally and logged to W&B.")
