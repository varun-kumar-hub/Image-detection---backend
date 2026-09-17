"""
Predictor Service
=================
Loads the EfficientNet model, manages inference lifecycle, and formats classification,
probability, and calibrated confidence estimates.
"""

import os
import time
import threading
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional
import tensorflow as tf

# Render's free instances have limited RAM and CPU. Avoid TensorFlow creating
# large thread pools that compete with the model and request processing.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from ml_pipeline.preprocessing import load_and_preprocess_image
from ml_pipeline.model_builder import build_efficientnet_model
from backend.app.schemas.analysis import PredictorResult

class PredictorService:
    def __init__(self, model_path: str = "models/image_detection_v1.keras"):
        self.model_path = Path(model_path)
        self.model: Optional[tf.keras.Model] = None
        self.model_version = "v1.0"
        self.model_name = "EfficientNet-B0"
        self.inference_lock = threading.Lock()
        self.load_model()

    def load_model(self):
        """Loads trained weights if available, or initializes backbone architecture."""
        try:
            if self.model_path.exists():
                print(f"[PredictorService] Loading model from {self.model_path}...")
                self.model = tf.keras.models.load_model(str(self.model_path))
                print("[PredictorService] Successfully loaded trained weights.")
            else:
                print(f"[PredictorService] Notice: {self.model_path} not found. Initializing pre-trained EfficientNet backbone.")
                self.model = build_efficientnet_model(variant="B0")
                print("[PredictorService] Initialized EfficientNet baseline architecture.")
        except Exception as e:
            print(f"[PredictorService] Error during model initialization: {e}")
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> PredictorResult:
        """
        Runs model inference on an image and returns structured probability and confidence metrics.
        """
        if self.model is None:
            raise RuntimeError("ML model is not loaded.")

        # Preprocess to (1, 224, 224, 3) normalized float32
        tensor = load_and_preprocess_image(image)

        # Inference (0 -> Real, 1 -> AI-Generated)
        with self.inference_lock:
            raw_pred = self.model.predict(tensor, verbose=0)
        prob_ai = float(raw_pred[0][0])
        prob_ai = min(max(prob_ai, 0.0), 1.0)

        ai_percentage = round(prob_ai * 100.0, 2)
        real_percentage = round((1.0 - prob_ai) * 100.0, 2)

        # Classification rule with borderline tolerance
        if 45.0 <= ai_percentage <= 55.0:
            classification = "needs_review"
        elif ai_percentage > 50.0:
            classification = "ai_generated"
        else:
            classification = "real"

        # Confidence Scoring
        dominant_prob = max(ai_percentage, real_percentage)
        if dominant_prob >= 90.0:
            confidence = "high"
            explanation = "The model produced a strong, decisive signal toward the detected class."
        elif dominant_prob >= 70.0:
            confidence = "medium"
            explanation = "The model found moderate evidence, though some subtle visual ambiguity exists."
        else:
            confidence = "low"
            explanation = "The model found conflicting visual indicators. The assessment should be interpreted cautiously."

        return PredictorResult(
            classification=classification,
            ai_probability=ai_percentage,
            real_probability=real_percentage,
            confidence=confidence,
            confidence_explanation=explanation
        )

    def explain(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Generates Grad-CAM visual explanation overlay highlighting prediction focus areas.
        """
        if self.model is None:
            return None

        try:
            import io
            import base64
            from ml_pipeline.explainability import get_gradcam_heatmap, generate_gradcam_overlay

            tensor = load_and_preprocess_image(image)
            heatmap = get_gradcam_heatmap(self.model, tensor)
            overlay = generate_gradcam_overlay(image, heatmap)

            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

            return {
                "available": True,
                "overlay_base64": f"data:image/png;base64,{b64_str}",
                "description": (
                    "Highlighted regions represent visual textures and spatial patterns that "
                    "contributed most strongly to the model's prediction. These indicators provide "
                    "supporting visual evidence, not standalone proof of AI generation."
                )
            }
        except Exception as e:
            print(f"[PredictorService] Explainability generation error: {e}")
            return {
                "available": False,
                "description": "Grad-CAM explanation is not available for this image."
            }

# Global singleton
predictor_service = PredictorService()
