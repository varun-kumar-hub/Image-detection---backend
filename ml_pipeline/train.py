"""
Model Training Script
=====================
Trains EfficientNet-B0 on the balanced and stratified dataset:
- data/processed/train/
- data/processed/validation/

Features:
- Rigorous data isolation: Validation and test sets are never augmented.
- Preprocessing parity: Inputs in [0, 255] range for EfficientNetB0 internal rescaling.
- Callbacks: ModelCheckpoint (monitoring val_auc), EarlyStopping, ReduceLROnPlateau.
- Artifacts:
  - models/image_detection_v1.keras
  - models/checkpoints/best_model.keras
  - models/model_config.json
  - models/training_history.csv
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from ml_pipeline.model_builder import build_efficientnet_model


def get_datasets(
    train_dir: Path,
    val_dir: Path,
    image_size=(224, 224),
    batch_size=32,
    seed=42
):
    """
    Loads train and validation datasets from pre-partitioned directories.
    Applies data augmentation ONLY to the training dataset.
    """
    print(f"[DataLoader] Loading training data from: {train_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "fake"], # 0: real, 1: fake
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed
    )

    print(f"[DataLoader] Loading validation data from: {val_dir}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "fake"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    # Augmentation pipeline applied exclusively to training data
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
    ], name="training_augmentation")

    # Important: EfficientNetB0 includes an internal Rescaling(1./255.0) layer.
    # We keep values in [0, 255] float32 so the internal layer normalizes correctly.
    train_ds = train_ds.map(lambda x, y: (data_augmentation(tf.cast(x, tf.float32)), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def train(
    train_dir: str = "data/processed/train",
    val_dir: str = "data/processed/validation",
    output_model_path: str = "models/image_detection_v1.keras",
    variant: str = "B0",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4
):
    train_path = Path(train_dir).resolve()
    val_path = Path(val_dir).resolve()
    output_path = Path(output_model_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.keras"

    print("=" * 65)
    print(f"IMAGEGUARD TRAINING PIPELINE (EfficientNet-{variant})")
    print("=" * 65)
    print(f"Train path:      {train_path}")
    print(f"Val path:        {val_path}")
    print(f"Model output:    {output_path}")
    print(f"Epochs:          {epochs}")
    print(f"Batch size:      {batch_size}")
    print(f"Learning rate:   {learning_rate}")

    train_ds, val_ds = get_datasets(train_path, val_path, batch_size=batch_size)

    # Initialize transfer learning architecture
    print("\n[Model] Building EfficientNet-B0 architecture...")
    model = build_efficientnet_model(
        input_shape=(224, 224, 3),
        variant=variant,
        trainable_base=False,
        learning_rate=learning_rate
    )
    model.summary()

    # Callbacks
    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=str(best_checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n[Training] Commencing training across {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callback_list
    )

    # Save final model
    print(f"\n[Artifacts] Saving final trained model to {output_path}...")
    model.save(str(output_path))

    # Save training history CSV
    history_csv_path = output_path.parent / "training_history.csv"
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        keys = list(history.history.keys())
        writer.writerow(["epoch"] + keys)
        num_recorded_epochs = len(history.history[keys[0]])
        for ep in range(num_recorded_epochs):
            row = [ep + 1] + [history.history[k][ep] for k in keys]
            writer.writerow(row)
    print(f"[Artifacts] Training history exported to {history_csv_path}")

    # Best validation metrics
    best_val_auc = max(history.history.get("val_auc", [0.0]))
    best_val_acc = max(history.history.get("val_accuracy", [0.0]))

    # Model configuration metadata
    config_path = output_path.parent / "model_config.json"
    config_data = {
        "model_name": "ImageGuard",
        "model_version": "1.0",
        "architecture": f"EfficientNet-{variant}",
        "input_size": [224, 224, 3],
        "classes": ["real", "ai_generated"],
        "class_mapping": {"0": "real", "1": "ai_generated"},
        "threshold_ai": 0.55,
        "threshold_real": 0.45,
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": len(history.epoch),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_val_auc": float(best_val_auc),
            "best_val_accuracy": float(best_val_acc)
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"[Artifacts] Model config written to {config_path}")

    print("\n" + "=" * 65)
    print("TRAINING PHASE COMPLETE")
    print(f"Best Validation AUC:      {best_val_auc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet on balanced Real vs AI dataset.")
    parser.add_argument("--train-dir", type=str, default="data/processed/train")
    parser.add_argument("--val-dir", type=str, default="data/processed/validation")
    parser.add_argument("--output", type=str, default="models/image_detection_v1.keras")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--variant", type=str, default="B0", help="EfficientNet variant")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_model_path=args.output,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
