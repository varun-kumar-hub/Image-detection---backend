"""
Forensics Service
=================
Integrates manipulation detection, EXIF parsing, ELA, and noise analysis into the API pipeline.
"""

from PIL import Image
from typing import Dict, Any
from ml_pipeline.manipulation_detector import ManipulationDetector
from backend.app.schemas.analysis import ManipulationAnalysisResult

class ForensicsService:
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

forensics_service = ForensicsService()
