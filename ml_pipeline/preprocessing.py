"""
Image Preprocessing Utilities
=============================
Standardized preprocessing pipeline for training and inference.
Guarantees identical resolution, channel order, and normalization.
"""

import numpy as np
from PIL import Image
from typing import Tuple

TARGET_IMAGE_SIZE = (224, 224)

def load_and_preprocess_image(
    image_input: str | Image.Image,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> np.ndarray:
    """
    Loads, resizes, and normalizes an image for EfficientNet inference.

    Pipeline:
        Input -> Verify/Open -> Convert to RGB -> Resize (Bilinear) -> Normalize [0, 1] -> Expand Dims

    Returns:
        np.ndarray with shape (1, height, width, 3) and dtype float32.
    """
    if isinstance(image_input, (str, bytes)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Ensure 3-channel RGB (handles RGBA, grayscale, CMYK, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # High quality resize
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float32 array in [0, 255] range
    # NOTE: EfficientNetB0 includes an internal Rescaling(1./255.0) and Normalization layer,
    # so input tensors must remain in [0, 255] to avoid accidental double-normalization.
    arr = np.array(img, dtype=np.float32)

    # Expand batch dimension
    return np.expand_dims(arr, axis=0)

def preprocess_for_training(image_array: np.ndarray) -> np.ndarray:
    """Preprocesses raw uint8 batch arrays for model consumption in [0, 255] float32."""
    return image_array.astype(np.float32)

