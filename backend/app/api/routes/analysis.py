"""
Analysis and Results Endpoints
==============================
Handles image analysis pipeline, secure private storage, signed URLs,
and structured evidence-based explanations with user authorization.
"""

import time
import asyncio
import io
import uuid
from datetime import datetime
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, status
from typing import Optional

from backend.app.core.security import AuthenticatedUser
from backend.app.api.dependencies import get_current_user, get_optional_current_user
from backend.app.utils.validators import validate_image_upload
from backend.app.services.predictor_service import predictor_service
from backend.app.services.image_analysis_service import image_analysis_service
from backend.app.services.explanation_service import explanation_service
from backend.app.services.storage_service import storage_service
from backend.app.services.supabase_service import db_service
from backend.app.schemas.analysis import ImageMetadataInfo
from backend.app.core.config import settings
from backend.app.api.routes.settings import fetch_backup_settings, save_backup_evaluation

router = APIRouter(prefix="", tags=["Analysis"])

def generate_analysis_id() -> str:
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    short_uuid = uuid.uuid4().hex[:8]
    return f"ANL-{date_str}-{short_uuid}"

@router.post("/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    upload_id: Optional[str] = Form(None),
    ground_truth: Optional[str] = Form(None),
    user: AuthenticatedUser = Depends(get_current_user)
):
    start_time = time.time()
    user_id = user.id

    # 1. Obtain image bytes and validate
    if file:
        image_bytes = await validate_image_upload(file)
        orig_filename = file.filename or "image.jpg"
        content_type = file.content_type or "image/jpeg"
        ext = orig_filename.split(".")[-1].lower() if "." in orig_filename else "jpg"
    else:
        raise HTTPException(
            status_code=400,
            detail={"code": "MISSING_FILE", "message": "An image file must be provided for analysis."}
        )

    # 2. Open PIL image & verify decode
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"code": "CORRUPT_IMAGE", "message": "The uploaded file is corrupt or not a valid image."}
        )

    # 3. Model Prediction (COMPLETELY INDEPENDENT OF GROUND TRUTH)
    # The ground truth is NEVER sent into the model or used to influence prediction.
    print(f"[Analyze] Starting prediction for {orig_filename}", flush=True)
    prediction = await predictor_service.predict(image_bytes)
    print("[Analyze] Prediction complete", flush=True)
    feature_analysis = await predictor_service.feature_analysis(image_bytes)
    print("[Analyze] Feature representation analysis complete", flush=True)

    # 4. Supporting Image Analysis (EXIF, ELA, Noise)
    supporting_analysis = await asyncio.to_thread(image_analysis_service.analyze_image, pil_image)
    print("[Analyze] Supporting analysis complete", flush=True)

    # 5. Model Explainability (Grad-CAM)
    if settings.ENABLE_GRADCAM:
        gradcam_result = await predictor_service.explain(image_bytes)
        print("[Analyze] Grad-CAM complete", flush=True)
    else:
        gradcam_result = {
            "available": False,
            "description": "Grad-CAM is disabled for this deployment to keep analysis responsive."
        }
    has_gradcam = bool(gradcam_result.get("available") and gradcam_result.get("overlay_base64"))

    # 6. Generate IDs and save to private Supabase Storage
    analysis_id = generate_analysis_id()
    upload_uid = str(uuid.uuid4())
    stored_filename = f"{analysis_id}.{ext}"

    # Rollback guard: if upload fails, clean up
    storage_path = None
    try:
        storage_path = await storage_service.upload_file(
            user_id=user_id,
            category="uploads",
            filename_id=stored_filename,
            file_bytes=image_bytes,
            mime_type=content_type
        )
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail={"code": "STORAGE_ERROR", "message": f"Failed to persist image to secure storage: {err}"}
        )

    try:
        await db_service.save_upload(
            upload_id=upload_uid,
            user_id=user_id,
            filename=orig_filename,
            storage_path=storage_path,
            file_size=len(image_bytes),
            mime_type=content_type,
        )
    except Exception as err:
        await storage_service.delete_file(storage_path)
        raise HTTPException(
            status_code=500,
            detail={"code": "PERSISTENCE_ERROR", "message": f"Failed to save upload record: {err}"}
        )

    # 7. Generate temporary signed URL for immediate display
    signed_image_url = await storage_service.create_signed_url(storage_path)

    # 8. Metadata extraction
    file_size_kb = len(image_bytes) / 1024
    size_str = f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{file_size_kb/1024:.2f} MB"
    image_info = ImageMetadataInfo(
        filename=orig_filename,
        file_size_bytes=len(image_bytes),
        file_size_human=size_str,
        dimensions=f"{pil_image.width} × {pil_image.height}",
        format=pil_image.format or "JPEG",
        color_mode=pil_image.mode
    )

    processing_time_ms = int((time.time() - start_time) * 1000)

    # 9. Build Structured, Evidence-Based Explanation
    structured_explanation = explanation_service.generate_explanation(
        classification=prediction["classification"],
        ai_probability=prediction["ai_probability"],
        real_probability=prediction["real_probability"],
        confidence=prediction["confidence"],
        supporting_details=supporting_analysis.details or {},
        has_gradcam=has_gradcam
    )

    # 10. Evaluation Comparison (Ground Truth Handling)
    # Evaluator testing mode ONLY compares after independent prediction.
    backup = await fetch_backup_settings(user_id)
    norm_gt = backup.get("reference") if backup.get("enabled") else None
    is_evaluation = bool(norm_gt in ("real", "ai_generated"))
    is_correct = None
    if is_evaluation:
        is_correct = (prediction["classification"] == ("real" if norm_gt == "authentic" else "ai_generated"))
        try:
            await save_backup_evaluation(analysis_id, user_id, norm_gt, prediction["classification"], prediction["ai_probability"] if norm_gt == "ai_generated" else prediction["real_probability"])
        except Exception as exc:
            print(f"[BackupMode] Evaluation save skipped: {exc}", flush=True)

    record = {
        "id": analysis_id,
        "upload_id": upload_uid,
        "user_id": user_id,
        "storage_path": storage_path,
        "filename": orig_filename,
        "classification": prediction["classification"],
        "ai_probability": prediction["ai_probability"],
        "real_probability": prediction["real_probability"],
        "confidence": prediction["confidence"],
        "confidence_explanation": prediction["confidence_explanation"],
        "interpretation": structured_explanation["summary"],
        "disclaimer": "AI image detection is probabilistic. Learned representations provide supporting evidence, not absolute proof.",
        "processing_time_ms": processing_time_ms,
        "model_name": predictor_service.model_name,
        "model_version": predictor_service.model_version,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "image_info": image_info.model_dump(),
        "manipulation": supporting_analysis.model_dump(),
        "explanation": structured_explanation,
        "gradcam": gradcam_result,
        "feature_analysis": feature_analysis,
        "image_url": signed_image_url,
        "ground_truth": norm_gt,
        "is_evaluation": is_evaluation,
        "is_correct": is_correct
    }

    # Save to persistence with rollback on failure
    try:
        await db_service.save_analysis(record, user_id=user_id)
    except Exception as e:
        if storage_path:
            await storage_service.delete_file(storage_path)
        raise HTTPException(
            status_code=500,
            detail={"code": "PERSISTENCE_ERROR", "message": f"Failed to save analysis record: {e}"}
        )

    public_record = {key: value for key, value in record.items() if key not in {"ground_truth", "is_evaluation", "is_correct"}}
    return {
        "success": True,
        "data": public_record
    }

@router.get("/results/{analysis_id}")
async def get_result(
    analysis_id: str,
    user: Optional[AuthenticatedUser] = Depends(get_optional_current_user)
):
    user_id = user.id if user else None
    record = db_service.get_analysis_by_id(analysis_id, user_id=user_id)
    if not record:
        record = await db_service.get_analysis_by_id_remote(analysis_id, user_id=user_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Analysis result with ID '{analysis_id}' was not found."}
        )

    # Attach fresh signed URL for displaying the image
    storage_path = record.get("storage_path")
    if storage_path:
        record["image_url"] = await storage_service.create_signed_url(storage_path)

    return {
        "success": True,
        "data": record
    }

@router.get("/results/{analysis_id}/image")
async def get_result_image_url(
    analysis_id: str,
    user: AuthenticatedUser = Depends(get_current_user)
):
    record = db_service.get_analysis_by_id(analysis_id, user_id=user.id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Analysis with ID '{analysis_id}' not found."}
        )

    storage_path = record.get("storage_path")
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "IMAGE_NOT_FOUND", "message": "No storage path found for this analysis."}
        )

    signed_url = await storage_service.create_signed_url(storage_path)
    return {
        "success": True,
        "data": {
            "url": signed_url,
            "expires_in": settings.SIGNED_URL_EXPIRES_IN
        }
    }
