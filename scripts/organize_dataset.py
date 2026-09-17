"""
ImageGuard Dataset Organizer
============================
Classifies and organizes raw image datasets into `data/real/` and `data/fake/`.

Features:
- Deterministic label assignment using folder hierarchy and file names
- Non-destructive: Copies images while strictly preserving original files
- Validates formats: JPG, JPEG, PNG, WEBP (ignores all non-image files)
- Thorough corruption detection using PIL (header verify + data load)
- Collision-resistant unique naming to prevent file overwrites
- Comprehensive error logging and final summary generation
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from PIL import Image

# Supported image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def setup_logger(log_file_path: Path):
    """Configures file and console logging for corruption tracking."""
    logger = logging.getLogger("DatasetOrganizer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler for corrupted files
    fh = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler for progress
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def verify_image(file_path: Path) -> bool:
    """
    Checks whether an image is valid and uncorrupted.
    Runs both verify() for format integrity and load() for pixel data integrity.
    """
    try:
        if file_path.stat().st_size == 0:
            return False
        with Image.open(file_path) as img:
            img.verify()
        # verify() closes the file or marks it unreadable for subsequent operations,
        # so reopen to test pixel decoding
        with Image.open(file_path) as img:
            img.load()
        return True
    except Exception:
        return False

def determine_class(file_path: Path, root_dir: Path) -> str | None:
    """
    Determines whether a file belongs to 'real' or 'fake' based on:
    1. Parent directory names relative to root_dir
    2. File name tokens
    Returns 'real', 'fake', or None.
    """
    rel_parts = [p.lower() for p in file_path.relative_to(root_dir).parts]
    full_str = "/".join(rel_parts)
    filename_lower = file_path.name.lower()

    # Priority check: explicit folder markers
    for part in rel_parts[:-1]:  # check parent folders
        if "real" in part or "authentic" in part or "original" in part:
            return "real"
        if "fake" in part or "ai" in part or "synthetic" in part or "generated" in part:
            return "fake"

    # Fallback check: filename markers
    if "real" in filename_lower or "authentic" in filename_lower:
        return "real"
    if "fake" in filename_lower or "ai" in filename_lower or "synth" in filename_lower:
        return "fake"

    return None

def get_unique_dest_path(dest_dir: Path, original_filename: str) -> Path:
    """Generates a non-colliding destination path if filename already exists."""
    dest_path = dest_dir / original_filename
    if not dest_path.exists():
        return dest_path

    stem = dest_path.stem
    suffix = dest_path.suffix
    counter = 1
    while dest_path.exists():
        dest_path = dest_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return dest_path

def organize_dataset(source_dir: str | Path, target_dir: str | Path):
    source_path = Path(source_dir).resolve()
    target_path = Path(target_dir).resolve()

    real_dir = target_path / "real"
    fake_dir = target_path / "fake"
    target_path.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    log_file = target_path / "corrupted_files.log"
    logger = setup_logger(log_file)

    logger.info("=" * 70)
    logger.info("IMAGEGUARD: DATASET CLASSIFICATION & ORGANIZATION")
    logger.info("=" * 70)
    logger.info(f"Source Directory: {source_path}")
    logger.info(f"Target Directory: {target_path}")
    logger.info(f"Output Folders  : {real_dir} | {fake_dir}")
    logger.info("Scanning for images (JPG, JPEG, PNG, WEBP)...")

    counts = {
        "total_scanned": 0,
        "real_copied": 0,
        "fake_copied": 0,
        "corrupted": 0,
        "ignored_non_image": 0,
        "unclassified": 0,
    }

    corrupted_files_list = []

    # Traverse all files in source
    for root, _, files in os.walk(source_path):
        for file in files:
            counts["total_scanned"] += 1
            file_path = Path(root) / file
            suffix = file_path.suffix.lower()

            # Filter non-image extensions
            if suffix not in ALLOWED_EXTENSIONS:
                counts["ignored_non_image"] += 1
                continue

            # Classify based on dataset labels
            cls = determine_class(file_path, source_path)
            if cls is None:
                counts["unclassified"] += 1
                continue

            # Integrity check
            if not verify_image(file_path):
                counts["corrupted"] += 1
                corrupted_files_list.append(str(file_path))
                logger.warning(f"Corrupted or unreadable image skipped: {file_path}")
                continue

            # Copy to target directory with collision avoidance
            target_dest_dir = real_dir if cls == "real" else fake_dir
            dest_file_path = get_unique_dest_path(target_dest_dir, file_path.name)

            shutil.copy2(file_path, dest_file_path)

            if cls == "real":
                counts["real_copied"] += 1
            else:
                counts["fake_copied"] += 1

            # Progress log every 500 images
            total_processed = counts["real_copied"] + counts["fake_copied"]
            if total_processed % 500 == 0:
                logger.info(f"Progress: {total_processed} valid images organized (Real: {counts['real_copied']}, Fake: {counts['fake_copied']})...")

    # Save summary json
    summary_path = target_path / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_dir": str(source_path),
            "target_dir": str(target_path),
            "metrics": counts,
            "corrupted_files_count": len(corrupted_files_list)
        }, f, indent=2)

    # Print clean final summary
    print("\n" + "=" * 70)
    print("                DATASET ORGANIZATION COMPLETE")
    print("=" * 70)
    print(f"Total Files Scanned      : {counts['total_scanned']}")
    print(f"Total REAL Images Copied : {counts['real_copied']}")
    print(f"Total FAKE Images Copied : {counts['fake_copied']}")
    print(f"Total Valid Images       : {counts['real_copied'] + counts['fake_copied']}")
    print(f"Corrupted Images Skipped : {counts['corrupted']}")
    print(f"Non-Image Files Ignored  : {counts['ignored_non_image']}")
    print(f"Unclassified Files       : {counts['unclassified']}")
    print("-" * 70)
    print(f"Corrupted Files Log      : {log_file}")
    print(f"Dataset Summary JSON     : {summary_path}")
    print("=" * 70)

    # Final Directory structure check
    print("\nFinal Folder Structure:")
    print("data/")
    print(f"|-- real/  ({counts['real_copied']} images)")
    print(f"|-- fake/  ({counts['fake_copied']} images)")
    print(f"|-- corrupted_files.log  ({counts['corrupted']} entries)")
    print(r"\-- dataset_summary.json")
    print("=" * 70 + "\n")

    return counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organize dataset into REAL and FAKE folders.")
    parser.add_argument("--source", type=str, required=False, help="Source path containing raw dataset")
    parser.add_argument("--target", type=str, default="d:/Image detection/data", help="Destination data folder")
    args = parser.parse_args()

    # If source is not provided, try to auto-detect Kaggle download or raw folder
    source = args.source
    if not source:
        candidates = [
            Path("D:/kaggle_cache/datasets/shivamardeshna/real-and-fake-images-dataset-for-image-forensics"),
            Path("d:/Image detection/data/raw"),
        ]
        for c in candidates:
            if c.exists():
                source = str(c)
                break

    if not source or not Path(source).exists():
        # Check subdirectories in D:/kaggle_cache
        base_cache = Path("D:/kaggle_cache")
        if base_cache.exists():
            for p in base_cache.glob("**/real-and-fake*"):
                if p.is_dir():
                    source = str(p)
                    break

    if not source or not Path(source).exists():
        print(f"Error: Could not locate source dataset directory. Please pass --source <path>", file=sys.stderr)
        sys.exit(1)

    organize_dataset(source, args.target)
