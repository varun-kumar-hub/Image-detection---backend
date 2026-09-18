"""Verify CIFAKE and create deterministic train/validation/test manifests."""
import argparse
from ml_pipeline.data_pipeline import prepare_cifake

parser = argparse.ArgumentParser(description="Prepare CIFAKE without altering its official test split.")
parser.add_argument("--dataset-root", default="data/datasets/cifake/raw")
parser.add_argument("--validation-fraction", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
prepare_cifake(args.dataset_root, args.validation_fraction, args.seed)
