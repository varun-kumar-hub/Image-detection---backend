"""
Model Evaluation Script
=======================
Evaluates the trained ImageGuard model strictly on the untouched test dataset:
- data/processed/test/

Outputs:
- models/evaluation_results.json
- models/confusion_matrix.png
- models/roc_curve.png
- Terminal diagnostic report

Metrics computed:
- Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Score
- AUC-ROC
- Confusion Matrix: TN, FP, FN, TP
- False Positive Rate (FPR) & False Negative Rate (FNR)
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import tensorflow as tf
from sklearn.metrics import roc_curve, auc

# Headless matplotlib configuration
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def evaluate_model(
    model_path: str = "models/image_detection_v1.keras",
    test_dir: str = "data/data/processed/test",
    output_dir: str = "models"
):
    model_file = Path(model_path).resolve()
    test_path = Path(test_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not model_file.exists():
        # Fallback to checkpoint if available
        checkpoint_file = model_file.parent / "checkpoints" / "best_model.keras"
        if checkpoint_file.exists():
            print(f"[Evaluation] {model_file} not found. Using best checkpoint at {checkpoint_file}")
            model_file = checkpoint_file
        else:
            print(f"Error: Model file not found at {model_file}")
            sys.exit(1)

    print("=" * 65)
    print("IMAGEGUARD TEST SET EVALUATION")
    print("=" * 65)
    print(f"Model file:  {model_file}")
    print(f"Test data:   {test_path}")

    # Load model
    print("\n[Evaluation] Loading model...")
    model = tf.keras.models.load_model(str(model_file))

    # Load test dataset (unshuffled, unaugmented)
    print("\n[Evaluation] Ingesting test dataset batches...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "fake"], # 0: real, 1: fake
        color_mode="rgb",
        batch_size=32,
        image_size=(224, 224),
        shuffle=False
    )

    # Cast to float32 for EfficientNetB0 internal rescaling
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))

    y_true = []
    y_pred_probs = []

    print("[Evaluation] Running batch inference across test images...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.array(y_pred_probs, dtype=float)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # Confusion matrix calculations
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total_samples = len(y_true)
    accuracy = float((tp + tn) / total_samples) if total_samples > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC calculation
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = float(auc(fpr_curve, tpr_curve))

    results = {
        "dataset": "Untouched Hold-Out Test Set",
        "total_test_samples": total_samples,
        "test_real_count": int(np.sum(y_true == 0)),
        "test_fake_count": int(np.sum(y_true == 1)),
        "metrics": {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "specificity": round(specificity * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "auc_roc": round(roc_auc, 4),
            "false_positive_rate": round(fpr * 100, 2),
            "false_negative_rate": round(fnr * 100, 2)
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        }
    }

    # Save results JSON
    results_json_path = out_path / "evaluation_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Artifacts] Evaluation metrics saved to {results_json_path}")

    # Plot 1: Minimalist Confusion Matrix
    cm_path = out_path / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    cm_data = np.array([[tn, fp], [fn, tp]])
    cax = ax.imshow(cm_data, cmap="Blues", interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "AI-Generated"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["Real", "AI-Generated"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Label", fontsize=11, labelpad=8)
    ax.set_title("ImageGuard Test Confusion Matrix", fontsize=12, fontweight="bold", pad=12)

    # Annotate numbers
    for i in range(2):
        for j in range(2):
            val = cm_data[i, j]
            color = "white" if val > cm_data.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(cm_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[Artifacts] Confusion matrix visualization saved to {cm_path}")

    # Plot 2: ROC Curve
    roc_path = out_path / "roc_curve.png"
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.plot(fpr_curve, tpr_curve, color="#0F172A", lw=2, label=f"EfficientNet-B0 (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(roc_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[Artifacts] ROC curve visualization saved to {roc_path}")

    # Terminal report
    print("\n" + "=" * 65)
    print("           MODEL TEST EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Test Samples:    {total_samples} ({results['test_real_count']} Real, {results['test_fake_count']} Fake)")
    print(f"Accuracy:              {results['metrics']['accuracy']}%")
    print(f"Precision:             {results['metrics']['precision']}%")
    print(f"Recall (Sensitivity):  {results['metrics']['recall']}%")
    print(f"Specificity:           {results['metrics']['specificity']}%")
    print(f"F1 Score:              {results['metrics']['f1_score']}%")
    print(f"AUC-ROC:               {results['metrics']['auc_roc']}")
    print(f"False Positive Rate:   {results['metrics']['false_positive_rate']}%")
    print(f"False Negative Rate:   {results['metrics']['false_negative_rate']}%")
    print("-" * 65)
    print("CONFUSION MATRIX:")
    print(f"  True Negatives  (Real -> Real): {tn:,}")
    print(f"  False Positives (Real -> AI):   {fp:,}")
    print(f"  False Negatives (AI -> Real):   {fn:,}")
    print(f"  True Positives  (AI -> AI):     {tp:,}")
    print("=" * 65 + "\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ImageGuard on hold-out test set.")
    parser.add_argument("--model", type=str, default="models/image_detection_v1.keras")
    parser.add_argument("--test-dir", type=str, default="data/data/processed/test")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    evaluate_model(args.model, args.test_dir, args.output_dir)
