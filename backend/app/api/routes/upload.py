"""
Image Upload Endpoint
=====================
Uploads an image directly to private Supabase Storage and returns upload metadata.
"""

import uuid
from fastapi import APIRouter, UploadFile, File, Depends
from typing import Optional

from backend.app.core.security import AuthenticatedUser
from backend.app.api.dependencies import get_optional_current_user
from backend.app.utils.validators import validate_image_upload
from backend.app.services.storage_service import storage_service

router = APIRouter(prefix="", tags=["Upload"])

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    user: Optional[AuthenticatedUser] = Depends(get_optional_current_user)
):
    # Validate image bytes, MIME, dimensions, and integrity
    contents = await validate_image_upload(file)
    user_id = user.id if user else "anonymous"

    upload_id = str(uuid.uuid4())
    orig_name = file.filename or "image.jpg"
    ext = orig_name.split(".")[-1].lower() if "." in orig_name else "jpg"
    filename_id = f"{upload_id}.{ext}"

    storage_path = await storage_service.upload_file(
        user_id=user_id,
        category="uploads",
        filename_id=filename_id,
        file_bytes=contents,
        mime_type=file.content_type or "image/jpeg"
    )

    return {
        "success": True,
        "upload_id": upload_id,
        "filename": orig_name,
        "storage_path": storage_path,
        "status": "uploaded"
    }
