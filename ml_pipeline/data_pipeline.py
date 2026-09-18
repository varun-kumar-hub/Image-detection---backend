"""CIFAKE dataset discovery, verification, splitting, and TensorFlow loading."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import tensorflow as tf
from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ("REAL", "AI GENERATED")
CLASS_MAPPING = {"REAL": 0, "AI GENERATED": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FOLDER_ALIASES = {
    "real": "REAL", "real_images": "REAL", "realimage": "REAL",
    "fake": "AI GENERATED", "ai": "AI GENERATED", "ai_generated": "AI GENERATED",
    "ai-generated": "AI GENERATED", "synthetic": "AI GENERATED",
}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    class_name: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_from_folder(folder: Path) -> str | None:
    normalized = folder.name.lower().replace(" ", "_")
    return FOLDER_ALIASES.get(normalized)


def _find_split_root(dataset_root: Path, split: str) -> Path:
    candidates = [entry for entry in dataset_root.iterdir() if entry.is_dir() and entry.name.lower() == split]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one '{split}' directory under {dataset_root}; found {len(candidates)}.")
    return candidates[0]


def scan_split(split_root: Path) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    """Read CIFAKE's folder labels, validating each image before it is used."""
    records: list[ImageRecord] = []
    invalid: list[dict[str, str]] = []
    for class_dir in sorted(entry for entry in split_root.iterdir() if entry.is_dir()):
        class_name = _class_from_folder(class_dir)
        if class_name is None:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                if image_path.stat().st_size == 0:
                    raise ValueError("empty file")
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.convert("RGB").load()
                records.append(ImageRecord(str(image_path.resolve()), CLASS_MAPPING[class_name], class_name, _sha256(image_path)))
            except (OSError, ValueError, UnidentifiedImageError) as error:
                invalid.append({"path": str(image_path.resolve()), "reason": str(error)})
    if not records:
        raise ValueError(f"No readable CIFAKE images were found in {split_root}.")
    return records, invalid


def _partition_training_records(records: list[ImageRecord], validation_fraction: float, seed: int) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split by content hash so exact duplicates cannot cross train/validation."""
    grouped: dict[int, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.label][record.sha256].append(record)

    train_records: list[ImageRecord] = []
    validation_records: list[ImageRecord] = []
    for label, hash_groups in grouped.items():
        groups = list(hash_groups.values())
        random.Random(seed + label).shuffle(groups)
        target_validation = round(sum(len(group) for group in groups) * validation_fraction)
        validation_count = 0
        for group in groups:
            if validation_count < target_validation:
                validation_records.extend(group)
                validation_count += len(group)
            else:
                train_records.extend(group)
    return train_records, validation_records


def _write_manifest(path: Path, records: Iterable[ImageRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")


def load_manifest(path: str | Path) -> list[ImageRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        return [ImageRecord(**json.loads(line)) for line in handle if line.strip()]


def prepare_cifake(dataset_root: str | Path, validation_fraction: float = 0.1, seed: int = 42) -> dict:
    """Create manifests without moving, augmenting, or modifying official CIFAKE files."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    root = Path(dataset_root).resolve()
    train_root, test_root = _find_split_root(root, "train"), _find_split_root(root, "test")
    source_train, invalid_train = scan_split(train_root)
    official_test, invalid_test = scan_split(test_root)
    train_records, validation_records = _partition_training_records(source_train, validation_fraction, seed)

    train_hashes = {record.sha256 for record in train_records}
    validation_hashes = {record.sha256 for record in validation_records}
    test_hashes = {record.sha256 for record in official_test}
    overlap_with_test = (train_hashes | validation_hashes) & test_hashes
    if overlap_with_test:
        raise ValueError(f"Detected {len(overlap_with_test)} exact duplicate hashes between CIFAKE training and official test data.")
    if train_hashes & validation_hashes:
        raise AssertionError("Duplicate hashes crossed the train/validation split.")

    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifests / "train.jsonl", train_records)
    _write_manifest(manifests / "validation.jsonl", validation_records)
    _write_manifest(manifests / "official_test.jsonl", official_test)
    (manifests / "invalid_files.json").write_text(json.dumps(invalid_train + invalid_test, indent=2), encoding="utf-8")

    def distribution(records: list[ImageRecord]) -> dict[str, int]:
        counts = Counter(record.class_name for record in records)
        return {class_name: counts[class_name] for class_name in CLASS_NAMES}

    summary = {
        "dataset": "CIFAKE Real and AI-Generated Synthetic Images",
        "dataset_root": str(root),
        "class_mapping": {"REAL": 0, "AI GENERATED": 1},
        "seed": seed,
        "validation_fraction": validation_fraction,
        "splits": {"train": distribution(train_records), "validation": distribution(validation_records), "official_test": distribution(official_test)},
        "invalid_files": len(invalid_train) + len(invalid_test),
        "exact_duplicate_hashes_within_source_train": len(source_train) - len({record.sha256 for record in source_train}),
        "official_test_modified": False,
    }
    (manifests / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for split_name in ("train", "validation", "official_test"):
        counts = summary["splits"][split_name]
        print(f"{split_name.replace('_', ' ').title()}: REAL {counts['REAL']}, AI GENERATED {counts['AI GENERATED']}")
    return summary


def dataset_from_manifest(manifest: str | Path, batch_size: int, training: bool, seed: int = 42) -> tf.data.Dataset:
    records = load_manifest(manifest)
    paths = [record.path for record in records]
    labels = [record.label for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        data = tf.io.read_file(path)
        image = tf.io.decode_image(data, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(tf.cast(image, tf.float32), (224, 224), method="bilinear")
        return image, tf.cast(label, tf.float32)

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.05), tf.keras.layers.RandomZoom(0.08)], name="training_augmentation")
        dataset = dataset.map(lambda image, label: (augmentation(image, training=True), label), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
