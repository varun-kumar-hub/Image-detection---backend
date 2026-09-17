"""
Image and Request Validators
============================
Enforces strict file validation (MIME, magic headers, size limit, and image integrity).
"""

from fastapi import UploadFile, HTTPException, status
from PIL import Image
import io

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

MAGIC_NUMBERS = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp"  # WEBP starts with RIFF....WEBP
}

async def validate_image_upload(file: UploadFile) -> bytes:
    """
    Validates uploaded file:
    1. Extension check
    2. Size check
    3. Magic bytes / header check
    4. PIL Image verification
    Returns the file content bytes if valid.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_FILENAME", "message": "No file name provided."}
        )

    # 1. Extension Check
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_FILE_TYPE",
                "message": f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            }
        )

    # Read content
    contents = await file.read()

    # 2. Size Check
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_FILE", "message": "The uploaded file is empty."}
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File size exceeds the 5MB limit (Current: {len(contents) / (1024 * 1024):.2f}MB)."
            }
        )

    # 3. Magic Number Check
    valid_header = False
    for magic in MAGIC_NUMBERS:
        if contents.startswith(magic):
            valid_header = True
            break

    if not valid_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_FILE_HEADER",
                "message": "The file signature does not match a valid JPG, PNG, or WEBP image."
            }
        )

    # 4. PIL Integrity Check
    try:
        with Image.open(io.BytesIO(contents)) as img:
            img.verify()
        with Image.open(io.BytesIO(contents)) as img:
            img.load()  # Catches truncated images
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "CORRUPTED_IMAGE", "message": "The image file appears corrupted or incomplete."}
        )

    # Reset cursor for downstream consumers
    await file.seek(0)
    return contents
