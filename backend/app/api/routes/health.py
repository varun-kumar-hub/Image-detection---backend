"""
Health Check Endpoint
"""

from fastapi import APIRouter
from backend.app.schemas.response import HealthResponse
from backend.app.services.predictor_service import predictor_service
from backend.app.core.config import settings

router = APIRouter(prefix="", tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=predictor_service.is_loaded(),
        model_version=settings.MODEL_VERSION,
        supported_formats=settings.ALLOWED_EXTENSIONS
    )
