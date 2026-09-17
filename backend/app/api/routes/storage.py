"""
Local Storage Stream Route
==========================
Serves locally cached or fallback storage files when running offline
or when Supabase Storage is not directly reachable.
"""

from fastapi import APIRouter, HTTPException, Response, status
from pathlib import Path
import mimetypes

from backend.app.services.storage_service import storage_service

router = APIRouter(prefix="/storage", tags=["Storage"])

@router.get("/{storage_path:path}")
async def get_storage_file(storage_path: str):
    # Prevent directory traversal
    if ".." in storage_path or storage_path.startswith("/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_PATH", "message": "Invalid storage path."}
        )

    file_bytes = storage_service.get_local_file_bytes(storage_path)
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Storage file not found."}
        )

    guessed_type, _ = mimetypes.guess_type(storage_path)
    content_type = guessed_type or "application/octet-stream"

    return Response(content=file_bytes, media_type=content_type)
