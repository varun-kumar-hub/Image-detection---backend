"""
Analysis and Prediction Pydantic Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class PredictorResult(BaseModel):
    classification: str = Field(..., description="'real', 'ai_generated', or 'needs_review'")
    ai_probability: float = Field(..., ge=0.0, le=100.0, description="Probability that the image is AI-generated (0-100)")
    real_probability: float = Field(..., ge=0.0, le=100.0, description="Probability that the image is authentic (0-100)")
    confidence: str = Field(..., description="'high', 'medium', or 'low'")
    confidence_explanation: str = Field(..., description="Human-readable confidence explanation")

class ManipulationAnalysisResult(BaseModel):
    compression_status: str
    resize_status: str
    filter_status: str
    metadata_status: str
    details: Dict[str, Any] = Field(default_factory=dict)

class ImageMetadataInfo(BaseModel):
    filename: str
    file_size_bytes: int
    file_size_human: str
    dimensions: str
    format: str
    color_mode: str

class AnalysisResponse(BaseModel):
    id: str
    upload_id: Optional[str] = None
    classification: str
    ai_probability: float
    real_probability: float
    confidence: str
    confidence_explanation: str
    interpretation: str
    disclaimer: str
    processing_time_ms: int
    model_name: str
    model_version: str
    created_at: str
    image_info: Optional[ImageMetadataInfo] = None
    manipulation: Optional[ManipulationAnalysisResult] = None
    explanation: Optional[Dict[str, Any]] = None
    gradcam: Optional[Dict[str, Any]] = None
    feature_analysis: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None
    ground_truth: Optional[str] = None
    is_evaluation: Optional[bool] = False
    is_correct: Optional[bool] = None
