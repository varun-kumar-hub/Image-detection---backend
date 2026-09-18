"""PyTorch inference for the custom 7-channel fused-parameter classifier."""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from backend.app.core.config import settings

logger = logging.getLogger(__name__)
IMAGE_SIZE = 32
CLASS_LABELS = ("AI-Generated (FAKE)", "Real Photo (REAL)")


class NonLinearCurveResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + inputs)


class HolisticParameterCurveFittingCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.init_block = nn.Sequential(nn.Conv2d(7, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = NonLinearCurveResidualBlock(64)
        self.transition = nn.Conv2d(64, 128, 1, bias=False)
        self.layer2 = NonLinearCurveResidualBlock(128)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 2048), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(2048, 2))

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.layer2(self.transition(self.layer1(self.init_block(inputs))))
        return self.classifier(functional.adaptive_avg_pool2d(outputs, (4, 4)))


def _normalize_feature_plane(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    cv2.normalize(values, result, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return result


def extract_seven_channel_features(raw_image_bytes: bytes) -> np.ndarray:
    """Decode upload bytes into independent RGB, Laplacian, and FFT domains."""
    if not raw_image_bytes:
        raise ValueError("The uploaded image is empty.")
    image = cv2.imdecode(np.frombuffer(raw_image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError("The uploaded file is not a decodable image.")
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported decoded image shape: {image.shape}")

    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    scale = np.iinfo(rgb.dtype).max if np.issubdtype(rgb.dtype, np.integer) else 1.0
    rgb = np.clip(rgb.astype(np.float32) / scale, 0.0, 1.0)
    texture = np.stack([_normalize_feature_plane(cv2.Laplacian(rgb[:, :, index], cv2.CV_32F, ksize=3)) for index in range(3)], axis=-1)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(grayscale)))).astype(np.float32)
    frequency = _normalize_feature_plane(spectrum)[..., np.newaxis]
    features = np.concatenate((rgb, texture, frequency), axis=-1)
    if features.shape != (IMAGE_SIZE, IMAGE_SIZE, 7) or not np.isfinite(features).all():
        raise ValueError("Unable to construct a finite 7-channel feature matrix.")
    return np.ascontiguousarray(features.transpose(2, 0, 1), dtype=np.float32)


class PredictorService:
    def __init__(self, model_path: str | Path = "models/custom_fused_parameter_model.pth") -> None:
        backend_root = Path(__file__).resolve().parents[2]
        path = Path(model_path)
        self.model_path = path if path.is_absolute() else backend_root / path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: HolisticParameterCurveFittingCNN | None = None
        self.model_name = "HolisticParameterCurveFittingCNN"
        self.model_version = "custom-fused-parameter-v1"
        self._inference_lock = threading.Lock()
        self.load_model()

    def load_model(self) -> None:
        if not self.model_path.is_file():
            logger.error("Custom model weights were not found at %s", self.model_path)
            self.model = None
            return
        try:
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(self.model_path, map_location=self.device)
            model = HolisticParameterCurveFittingCNN().to(self.device)
            model.load_state_dict(self._extract_state_dict(checkpoint), strict=True)
            self.model = model.eval()
            logger.info("Loaded custom 7-channel model on %s", self.device.type)
        except (OSError, RuntimeError, ValueError, TypeError, EOFError) as error:
            logger.exception("Unable to load custom model weights: %s", error)
            self.model = None

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> Mapping[str, Tensor]:
        if isinstance(checkpoint, Mapping):
            for key in ("model_state_dict", "state_dict"):
                if isinstance(checkpoint.get(key), Mapping):
                    return checkpoint[key]
            if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
                return checkpoint
        raise ValueError("Checkpoint does not contain a valid PyTorch state dictionary.")

    def is_loaded(self) -> bool:
        return self.model is not None

    async def predict(self, raw_image_bytes: bytes) -> dict[str, Any]:
        return await asyncio.to_thread(self._predict_sync, raw_image_bytes)

    def _predict_sync(self, raw_image_bytes: bytes) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("The custom PyTorch model is unavailable.")
        inputs = torch.from_numpy(extract_seven_channel_features(raw_image_bytes)).unsqueeze(0).to(self.device)
        with self._inference_lock, torch.no_grad():
            probabilities = torch.softmax(self.model(inputs), dim=1)[0].detach().cpu().numpy()
        class_index = int(np.argmax(probabilities))
        fake_probability, real_probability = float(probabilities[0]), float(probabilities[1])
        confidence_percentage = float(probabilities[class_index] * 100.0)
        return {
            "success": True,
            "classification": "ai_generated" if class_index == 0 else "real",
            "prediction_label": CLASS_LABELS[class_index],
            "class_index": class_index,
            "confidence": "high" if confidence_percentage >= 90 else "medium" if confidence_percentage >= 70 else "low",
            "confidence_percentage": round(confidence_percentage, 2),
            "ai_probability": round(fake_probability * 100.0, 2),
            "real_probability": round(real_probability * 100.0, 2),
            "confidence_explanation": "Softmax confidence from the custom 7-channel PyTorch classifier.",
        }

    async def feature_analysis(self, raw_image_bytes: bytes) -> dict[str, Any]:
        features = await asyncio.to_thread(extract_seven_channel_features, raw_image_bytes)
        return {"embedding": {"available": False, "note": "No persisted embedding API is available."}, "preprocessing": {"status": "MATCHED", "input_size": "32 × 32 × 7", "domains": ["RGB", "Laplacian", "FFT magnitude"]}, "image_statistics": {"spatial_mean": round(float(features[:3].mean()), 4), "texture_mean": round(float(features[3:6].mean()), 4), "frequency_mean": round(float(features[6].mean()), 4)}, "similarity": {"available": False, "note": "Reference embeddings are not configured."}}

    async def explain(self, raw_image_bytes: bytes) -> dict[str, Any]:
        return {"available": False, "description": "Grad-CAM is not configured for the custom PyTorch model."}


predictor_service = PredictorService(settings.MODEL_PATH)
