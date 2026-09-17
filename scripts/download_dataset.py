import os
import sys

# Ensure downloads go to D: drive where there is 120GB free space
os.environ["KAGGLEHUB_CACHE"] = r"D:\kaggle_cache"

import kagglehub

print("Connecting to Kaggle to download dataset: shivamardeshna/real-and-fake-images-dataset-for-image-forensics ...")
try:
    path = kagglehub.dataset_download("shivamardeshna/real-and-fake-images-dataset-for-image-forensics")
    print(f"SUCCESS: Dataset downloaded to: {path}")
except Exception as e:
    print(f"ERROR downloading dataset: {e}", file=sys.stderr)
    sys.exit(1)
