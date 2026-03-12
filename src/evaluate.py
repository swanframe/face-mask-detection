# =============================================================================
# evaluate.py — Model evaluation: metrics, confusion matrix, training curves
# =============================================================================

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    TEST_DIR, IMG_SIZE, BATCH_SIZE,
    MODEL_BEST_PATH, OUTPUT_DIR, CLASSES
)


def load_test_generator():
    """Create test data generator (no augmentation, no shuffle)."""
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=False,
    )
    return test_gen


def plot_training_curves() -> None:
    """Plot accuracy & loss curves for both training phases."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    phases = [
        ("frozen",   "Phase 1 — Frozen Backbone"),
        ("finetune", "Phase 2 — Fine-tuning"),
    ]

    for col, (phase, title) in enumerate(phases):
        history_path = os.path.join(OUTPUT_DIR, f"history_{phase}.json")
        if not os.path.exists(history_path):
            print(f"  ⚠️  History not found: {history_path}")
            continue

        with open(history_path) as f:
            h = json.load(f)

        epochs = range(1, len(h["accuracy"]) + 1)

        # Accuracy
        axes[0][col].plot(epochs, h["accuracy"],     label="Train", marker="o")
        axes[0][col].plot(epochs, h["val_accuracy"], label="Val",   marker="s")
        axes[0][col].set_title(f"{title} — Accuracy")
        axes[0][col].set_xlabel("Epoch")
        axes[0][col].set_ylabel("Accuracy")
        axes[0][col].legend()
        axes[0][col].grid(True, alpha=0.3)
        axes[0][col].set_ylim(0.8, 1.01)

        # Loss
        axes[1][col].plot(epochs, h["loss"],     label="Train", marker="o")
        axes[1][col].plot(epochs, h["val_loss"], label="Val",   marker="s")
        axes[1][col].set_title(f"{title} — Loss")
        axes[1][col].set_xlabel("Epoch")
        axes[1][col].set_ylabel("Loss")
        axes[1][col].legend()
        axes[1][col].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 Training curve saved: {out_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Confusion Matrix (Count)", "Confusion Matrix (%)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 Confusion matrix saved: {out_path}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score   = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 ROC curve saved: {out_path}")


def save_classification_report(report: str) -> None:
    """Save text classification report to outputs/."""
    out_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"  💾 Classification report saved: {out_path}")


def evaluate() -> None:
    print("=" * 60)
    print("  FACE MASK DETECTION — EVALUATION PIPELINE")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model & test data
    # ------------------------------------------------------------------
    print("\n📂 Loading the best model...")
    model = tf.keras.models.load_model(MODEL_BEST_PATH)
    print(f"  ✅ Model loaded from: {MODEL_BEST_PATH}")

    print("\n📂 Loading test generator...")
    test_gen = load_test_generator()
    print(f"  ✅ Test : {test_gen.samples} images")

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    print("\n🔍 Making predictions on the test set...")
    y_prob = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_gen.classes

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("\n📊 Evaluation Results:")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(report)
    save_classification_report(report)

    auc = roc_auc_score(y_true, y_prob)
    print(f"  ROC-AUC Score : {auc:.4f}")

    acc = np.mean(y_true == y_pred)
    print(f"  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    print("\n📈 Creating visualizations...")
    plot_training_curves()
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_prob)

    print("\n🎉 Evaluation complete! All output is saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    evaluate()
