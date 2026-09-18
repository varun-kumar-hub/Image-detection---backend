"""Evaluate a CIFAKE model only against the untouched official test manifest."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from ml_pipeline.data_pipeline import dataset_from_manifest, load_manifest, prepare_cifake


def evaluate(model: str = "models/ai_image_detector.keras", dataset_root: str = "data/datasets/cifake/raw", output: str = "models/evaluation_results.json", batch_size: int = 64) -> None:
    root = Path(dataset_root).resolve()
    manifest = root / "manifests" / "official_test.jsonl"
    if not manifest.exists():
        prepare_cifake(root)
    records = load_manifest(manifest)
    model_instance = tf.keras.models.load_model(model)
    probabilities = model_instance.predict(dataset_from_manifest(manifest, batch_size, training=False), verbose=1).reshape(-1)
    labels = np.array([record.label for record in records])
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    results = {"dataset": "CIFAKE official untouched test set", "class_mapping": {"REAL": 0, "AI GENERATED": 1}, "total_test_samples": int(len(labels)),
        "metrics": {"accuracy": float((tp + tn) / len(labels)), "precision": float(precision), "recall": float(recall), "f1_score": float(f1), "roc_auc": float(roc_auc_score(labels, probabilities)),
                    "real_accuracy": float(tn / (tn + fp)), "ai_accuracy": float(tp / (tp + fn)), "false_positive_rate": float(fp / (fp + tn)), "false_negative_rate": float(fn / (fn + tp))},
        "confusion_matrix": {"true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp)}}
    output_path = Path(output).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ai_image_detector.keras")
    parser.add_argument("--dataset-root", default="data/datasets/cifake/raw")
    parser.add_argument("--output", default="models/evaluation_results.json")
    parser.add_argument("--batch-size", type=int, default=64)
    evaluate(**vars(parser.parse_args()))
