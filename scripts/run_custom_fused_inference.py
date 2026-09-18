"""Run inference with the 7-channel HolisticParameterCurveFittingCNN checkpoint.

Usage:
    python run_custom_fused_inference.py path/to/image.jpg
    python run_custom_fused_inference.py path/to/image.jpg --weights custom_fused_parameter_model.pth
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional


IMAGE_SIZE = 32
CLASS_NAMES = ("AI-Generated (FAKE)", "Real Photo (REAL)")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "custom_fused_parameter_model.pth"


class NonLinearCurveResidualBlock(nn.Module):
    """The residual curve-fitting block represented by the checkpoint tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


class HolisticParameterCurveFittingCNN(nn.Module):
    """Exact module/key layout inferred from `custom_fused_parameter_model.pth`."""

    def __init__(self) -> None:
        super().__init__()
        self.init_block = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = NonLinearCurveResidualBlock(64)
        self.transition = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.layer2 = NonLinearCurveResidualBlock(128)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.init_block(inputs)
        outputs = self.layer1(outputs)
        outputs = self.transition(outputs)
        outputs = self.layer2(outputs)
        outputs = functional.adaptive_avg_pool2d(outputs, output_size=(4, 4))
        return self.classifier(outputs)


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize one feature plane independently and keep constant planes finite."""
    minimum, maximum = float(values.min()), float(values.max())
    span = maximum - minimum
    if span <= np.finfo(np.float32).eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / span).astype(np.float32)


def build_seven_channel_tensor(image_path: str | Path) -> Tensor:
    """Create a [1, 7, 32, 32] tensor without mixing spatial, texture, and FFT domains."""
    path = str(image_path)
    source_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise ValueError(f"Unable to decode image: {path}")
    rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    spatial_rgb = np.moveaxis(rgb, -1, 0)
    laplacian_rgb = np.stack(
        [_min_max_normalize(np.abs(cv2.Laplacian(rgb[:, :, channel], cv2.CV_32F, ksize=3))) for channel in range(3)],
        axis=0,
    )
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fft_magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(grayscale))))
    fft_channel = _min_max_normalize(fft_magnitude)[None, :, :]

    features = np.concatenate((spatial_rgb, laplacian_rgb, fft_channel), axis=0).astype(np.float32, copy=False)
    if features.shape != (7, IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(features).all():
        raise RuntimeError(f"Invalid feature tensor produced: shape={features.shape}")
    return torch.from_numpy(features).unsqueeze(0)


def _extract_state_dict(checkpoint: object) -> Mapping[str, Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                return checkpoint[key]
        if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint must be a PyTorch state_dict or contain model_state_dict/state_dict.")


def load_model(weights_path: str | Path, device: torch.device) -> HolisticParameterCurveFittingCNN:
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        checkpoint = torch.load(weights_path, map_location=device)
    model = HolisticParameterCurveFittingCNN().to(device)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    return model.eval()


def predict(image_path: str | Path, weights_path: str | Path, device: torch.device) -> tuple[str, float]:
    model = load_model(weights_path, device)
    inputs = build_seven_channel_tensor(image_path).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    class_index = int(torch.argmax(probabilities).item())
    return CLASS_NAMES[class_index], float(probabilities[class_index].item() * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="7-channel AI-image inference with a fused parameter CNN.")
    parser.add_argument("image", type=Path, help="Image to classify")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    label, confidence = predict(args.image, args.weights, device)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
