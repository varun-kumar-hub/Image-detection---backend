"""
Image Analysis Service
======================
Calculates supporting image characteristics:
- EXIF metadata presence
- Compression & error-level attributes (ELA)
- Sensor noise variance & texture distribution
- Resolution & aspect ratio integrity

These observations are supporting indicators and are not treated as independent proof of AI generation.
"""

from PIL import Image
from typing import Dict, Any
from ml_pipeline.manipulation_detector import ManipulationDetector
from backend.app.schemas.analysis import ManipulationAnalysisResult

class ImageAnalysisService:
    def __init__(self):
        self.detector = ManipulationDetector()

    def analyze_image(self, image: Image.Image) -> ManipulationAnalysisResult:
        raw_result = self.detector.analyze(image)
        return ManipulationAnalysisResult(
            compression_status=raw_result["compression_status"],
            resize_status=raw_result["resize_status"],
            filter_status=raw_result["filter_status"],
            metadata_status=raw_result["metadata_status"],
            details={
                "exif": raw_result.get("exif", {}),
                "ela": raw_result.get("ela", {}),
                "noise": raw_result.get("noise", {}),
                "dimensions": raw_result.get("dimensions", {})
            }
        )

image_analysis_service = ImageAnalysisService()
# Alias for compatibility
forensics_service = image_analysis_service
