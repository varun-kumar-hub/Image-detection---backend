"""
Tests for Image Analysis and Manipulation Detector
"""

import numpy as np
from PIL import Image
from ml_pipeline.manipulation_detector import ManipulationDetector

def test_manipulation_detector_initialization():
    detector = ManipulationDetector()
    assert detector.ela_quality == 90
    assert detector.ela_scale == 15

def test_manipulation_detector_synthetic_image():
    # Create clean synthetic image
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    detector = ManipulationDetector()
    result = detector.analyze(img)

    assert "metadata_status" in result
    assert "compression_status" in result
    assert "filter_status" in result
    assert "resize_status" in result
    assert "ela" in result
    assert "noise" in result
    assert "dimensions" in result
