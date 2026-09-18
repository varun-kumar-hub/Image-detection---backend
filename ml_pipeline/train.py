"""Fine-tune EfficientNet-B0 for CIFAKE REAL (0) versus AI GENERATED (1)."""
import argparse
import csv
import json
from pathlib import Path

import tensorflow as tf

from ml_pipeline.data_pipeline import CLASS_MAPPING, dataset_from_manifest, prepare_cifake
from ml_pipeline.model_builder import build_efficientnet_model


def train(dataset_root: str = "data/datasets/cifake/raw", output: str = "models/ai_image_detector.keras", epochs: int = 15, batch_size: int = 64, learning_rate: float = 1e-4, seed: int = 42) -> None:
    tf.keras.utils.set_random_seed(seed)
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("CUDA GPU detected: mixed-precision training enabled.")
    root = Path(dataset_root).resolve()
    manifests = root / "manifests"
    if not (manifests / "train.jsonl").exists():
        prepare_cifake(root, seed=seed)
    train_ds = dataset_from_manifest(manifests / "train.jsonl", batch_size, training=True, seed=seed)
    validation_ds = dataset_from_manifest(manifests / "validation.jsonl", batch_size, training=False)

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.parent / "checkpoints" / "best_ai_image_detector.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_efficientnet_model(variant="B0", trainable_base=False, learning_rate=learning_rate)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]
    history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=callbacks)
    # Reload the monitored best checkpoint, never the final epoch, then export it for the app.
    best_model = tf.keras.models.load_model(checkpoint_path)
    best_model.save(output_path)
    with (output_path.parent / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *history.history])
        writer.writerows([index + 1, *(history.history[key][index] for key in history.history)] for index in range(len(history.epoch)))
    metadata = {
        "model_name": "CIFAKE AI-Generated Image Detector", "architecture": "EfficientNet-B0", "class_mapping": CLASS_MAPPING,
        "classes": ["REAL", "AI GENERATED"], "input_size": [224, 224, 3], "color_mode": "RGB",
        "normalization": "EfficientNetB0 built-in ImageNet preprocessing; RGB float32 pixels in [0, 255]", "best_checkpoint": str(checkpoint_path),
    }
    (output_path.parent / "model_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved best CIFAKE model: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/datasets/cifake/raw")
    parser.add_argument("--output", default="models/ai_image_detector.keras")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    train(**vars(parser.parse_args()))
