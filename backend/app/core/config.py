"""
Backend Application Configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path

class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "Image Detection"
    APP_VERSION: str = "4.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = False

    # Supabase Configuration
    SUPABASE_URL: str = "https://uurxeccaasfacgnntaxo.supabase.co"
    SUPABASE_ANON_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV1cnhlY2NhYXNmYWNnbm50YXhvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk2NDE1ODYsImV4cCI6MjEwNTIxNzU4Nn0.6PT0guOD1Hv4lzLNpKLc2hSgjbKUkHaTG5U3oDRvsSE"
    SUPABASE_SERVICE_ROLE_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV1cnhlY2NhYXNmYWNnbm50YXhvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4OTY0MTU4NiwiZXhwIjoyMTA1MjE3NTg2fQ.sRBq97dn2JXwWm4n4Au3LKdiiXcuhZIKE_0wFgS3cao"
    SUPABASE_BUCKET_UPLOADS: str = "imageguard"
    SUPABASE_BUCKET_REPORTS: str = "imageguard"
    SUPABASE_STORAGE_BUCKET: str = "imageguard"
    SIGNED_URL_EXPIRES_IN: int = 3600  # 1 hour validity

    # ML Model Configuration
    MODEL_PATH: str = "models/image_detection_v1.keras"
    ENABLE_GRADCAM: bool = False
    MODEL_VERSION: str = "v1.0"
    CONFIDENCE_HIGH_THRESHOLD: float = 0.90
    CONFIDENCE_MEDIUM_THRESHOLD: float = 0.70

    # Uploads & Security
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]

    # CORS
    CORS_ORIGINS: List[str] = [
        "https://image-detection-frontend-wine.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
