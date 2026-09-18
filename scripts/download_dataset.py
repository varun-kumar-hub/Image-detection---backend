"""Download CIFAKE to the configurable local dataset root using KaggleHub."""
from pathlib import Path
import argparse
import shutil
import kagglehub

parser = argparse.ArgumentParser()
parser.add_argument("--destination", default="data/datasets/cifake/raw")
args = parser.parse_args()
source = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))
destination = Path(args.destination).resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists() and any(destination.iterdir()):
    raise SystemExit(f"Destination already contains files: {destination}. Choose an empty destination.")
shutil.copytree(source, destination, dirs_exist_ok=True)
print(f"CIFAKE downloaded to {destination}")
