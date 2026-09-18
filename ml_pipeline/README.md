# CIFAKE Training Pipeline

This pipeline trains an EfficientNet-B0 classifier for **REAL IMAGE (0)** versus **AI GENERATED (1)** using CIFAKE. It keeps CIFAKE's official test split untouched and creates a deterministic validation split only from the official training split.

## Layout

```
data/datasets/
├── cifake/raw/{train,test}/{REAL,FAKE}/
├── cifake/manifests/{train,validation,official_test}.jsonl
├── additional_real/
└── additional_ai/
```

`prepare_dataset.py` verifies images, records unreadable files, detects exact content duplicates, and fails if training data overlaps the official test set. It does not alter CIFAKE source files.

## Commands

```powershell
python scripts/download_dataset.py
python scripts/prepare_dataset.py --dataset-root data/datasets/cifake/raw
python -m ml_pipeline.train --dataset-root data/datasets/cifake/raw
python -m ml_pipeline.evaluate --model models/ai_image_detector.keras --dataset-root data/datasets/cifake/raw
```

The application loads `models/ai_image_detector.keras` by default. Its Grad-CAM implementation targets EfficientNet-B0's final convolutional layer and the predicted binary class.

## Limitation

CIFAKE synthetic examples are Stable-Diffusion-generated, CIFAR-10-style low-resolution images. Results from CIFAKE alone do not establish performance for every modern image generator. `additional_real` and `additional_ai` are reserved for future datasets after separate provenance and split controls are added.
