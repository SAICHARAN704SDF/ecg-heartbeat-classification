import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
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
    TRAIN_FILE, TEST_FILE, RANDOM_SEED, NUM_CLASSES,
    SIGNAL_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    VAL_SIZE, MODEL_DIR, RESULTS_DIR
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_NAME = "ecg-mitbih-research"
RUN_NAME = "baseline-cnn-v2-wandb"

CLASS_NAMES = [
    "Normal",
    "Supraventricular",
    "Ventricular",
    "Fusion",
    "Unknown",
]

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# W&B INITIALIZATION
# ============================================================

run = wandb.init(
    project=PROJECT_NAME,
    name=RUN_NAME,
    config={
        "experiment": "Baseline 1D CNN",
        "dataset": "MIT-BIH ECG Heartbeat Categorization",
        "architecture": "3-layer 1D CNN",
        "signal_length": SIGNAL_LENGTH,
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

print("=" * 75)
print("MIT-BIH BASELINE 1D CNN — WEIGHTS & BIASES")
print("=" * 75)

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
print(f"Signal length:    {X.shape[1]}")

# ============================================================
# DATASET CLASS DISTRIBUTION
# ============================================================

class_counts = np.bincount(y, minlength=NUM_CLASSES)

distribution_table = wandb.Table(
    columns=["class_id", "class_name", "count", "percentage"]
)

for class_id, count in enumerate(class_counts):
    distribution_table.add_data(
        class_id,
        CLASS_NAMES[class_id],
        int(count),
        float(count / len(y) * 100),
    )

wandb.log({"dataset/class_distribution": distribution_table})

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,
)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"Train split:      {X_train.shape}")
print(f"Validation split: {X_val.shape}")
print(f"Test split:       {X_test.shape}")

# ============================================================
# MODEL
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SIGNAL_LENGTH, 1)),

    tf.keras.layers.Conv1D(
        32, 7, padding="same", activation="relu", name="conv_block_1"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(
        64, 5, padding="same", activation="relu", name="conv_block_2"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(
        128, 3, padding="same", activation="relu", name="conv_block_3"
    ),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ============================================================
# CUSTOM MACRO-F1 CALLBACK
# ============================================================

class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """
    Calculates validation Macro-F1, Balanced Accuracy,
    per-class F1, precision and recall after every epoch.
    """

    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        probabilities = self.model.predict(
            self.x_val,
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
            predictions
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

        for class_id in range(NUM_CLASSES):
            class_f1 = f1_score(
                self.y_val,
                predictions,
                labels=[class_id],
                average=None,
                zero_division=0,
            )[0]

            logs[f"val_f1_class_{class_id}"] = float(class_f1)

        wandb.log({
            "epoch": epoch + 1,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1,
            "val_balanced_accuracy": balanced_acc,
            "val_macro_precision": macro_precision,
            "val_macro_recall": macro_recall,
            **{
                f"val_f1/{CLASS_NAMES[class_id]}":
                logs[f"val_f1_class_{class_id}"]
                for class_id in range(NUM_CLASSES)
            },
        })

        print(
            f"\nEpoch {epoch + 1:02d} | "
            f"Val Macro-F1: {macro_f1:.4f} | "
            f"Val Balanced Acc: {balanced_acc:.4f}"
        )


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = MODEL_DIR / "baseline_cnn_v2_best_macro_f1.keras"

callbacks = [
    ValidationMetricsCallback(X_val, y_val),

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
# TRAIN — FULL 30 EPOCH BUDGET
# ============================================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# ============================================================
# LOAD BEST MACRO-F1 CHECKPOINT
# ============================================================

print("\nLoading best validation Macro-F1 checkpoint...")

model = tf.keras.models.load_model(checkpoint_path)

# ============================================================
# FINAL TEST EVALUATION
# ============================================================

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1,
)

predictions = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, predictions)
balanced_acc = balanced_accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")
weighted_f1 = f1_score(y_test, predictions, average="weighted")

macro_precision = precision_score(
    y_test, predictions, average="macro", zero_division=0
)

macro_recall = recall_score(
    y_test, predictions, average="macro", zero_division=0
)

cm = confusion_matrix(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print("\n" + "=" * 75)
print("FINAL TEST RESULTS — BEST VALIDATION MACRO-F1 CHECKPOINT")
print("=" * 75)
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")
print(f"Macro-F1:          {macro_f1:.4f}")
print(f"Weighted-F1:       {weighted_f1:.4f}")

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(cm)

# ============================================================
# CONFUSION MATRIX — W&B
# ============================================================

wandb_cm = wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_test,
    preds=predictions,
    class_names=CLASS_NAMES,
    title="MIT-BIH Test Confusion Matrix",
)

wandb.log({
    "test/confusion_matrix": wandb_cm
})

# ============================================================
# PER-CLASS METRICS — W&B
# ============================================================

report_dict = classification_report(
    y_test,
    predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

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
    "test/per_class_metrics": metric_table
})

# ============================================================
# LOCAL VISUALIZATIONS
# ============================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    RESULTS_DIR / "baseline_cnn_v2_wandb_history.csv",
    index=False,
)

# Training curves
fig, axes = plt.subplots(2, 1, figsize=(10, 9))

axes[0].plot(
    history_df["loss"],
    label="Training Loss",
)
axes[0].plot(
    history_df["val_loss"],
    label="Validation Loss",
)
axes[0].set_title("Baseline CNN — Loss")
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
axes[1].set_title("Baseline CNN — Accuracy / Macro-F1")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

training_plot_path = RESULTS_DIR / "baseline_cnn_v2_training_curves.png"
plt.savefig(training_plot_path, dpi=180)
wandb.log({
    "plots/training_curves": wandb.Image(
        str(training_plot_path)
    )
})
plt.close()

# Per-class F1
class_f1_values = [
    report_dict[name]["f1-score"]
    for name in CLASS_NAMES
]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(CLASS_NAMES, class_f1_values)
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.set_title("Baseline CNN — Test Per-Class F1")
ax.grid(axis="y", alpha=0.3)

for i, value in enumerate(class_f1_values):
    ax.text(i, value + 0.02, f"{value:.3f}", ha="center")

plt.tight_layout()

f1_plot_path = RESULTS_DIR / "baseline_cnn_v2_per_class_f1.png"
plt.savefig(f1_plot_path, dpi=180)
wandb.log({
    "plots/per_class_f1": wandb.Image(
        str(f1_plot_path)
    )
})
plt.close()

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "experiment": RUN_NAME,
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_acc),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "confusion_matrix": cm.tolist(),
    "classification_report": report_dict,
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "epochs_requested": int(EPOCHS),
    "epochs_completed": int(len(history.history["loss"])),
    "random_seed": RANDOM_SEED,
}

with open(
    RESULTS_DIR / "baseline_cnn_v2_wandb_results.json",
    "w"
) as f:
    json.dump(results, f, indent=2)

np.save(
    RESULTS_DIR / "baseline_cnn_v2_predictions.npy",
    predictions,
)

np.save(
    RESULTS_DIR / "baseline_cnn_v2_probabilities.npy",
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

print("\n" + "=" * 75)
print("EXPERIMENT COMPLETE")
print("=" * 75)
print(f"Best checkpoint: {checkpoint_path}")
print(f"Completed epochs: {len(history.history['loss'])}")
print("W&B run has been logged.")
