"""
Standard API Response and Error Schemas
"""

from pydantic import BaseModel
from typing import Optional, Generic, TypeVar, Any

T = TypeVar("T")

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None

class StandardResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[ErrorDetail] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    supported_formats: list[str]
