"""
History and Management Routes
==============================
Supports paginated query, classification filtering, signed thumbnail generation,
and cascade result deletion with strict user authorization.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, status
from typing import Optional

from backend.app.core.security import AuthenticatedUser
from backend.app.api.dependencies import get_current_user
from backend.app.services.supabase_service import db_service
from backend.app.services.storage_service import storage_service

router = APIRouter(prefix="", tags=["History"])

@router.get("/history")
async def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    classification: Optional[str] = Query(None),
    user: AuthenticatedUser = Depends(get_current_user)
):
    history_data = db_service.list_analyses(
        user_id=user.id,
        page=page,
        limit=limit,
        classification=classification
    )

    # Attach signed thumbnail URLs for each item
    for item in history_data.get("items", []):
        storage_path = item.get("storage_path")
        if storage_path:
            item["thumbnail_url"] = await storage_service.create_signed_url(storage_path)

    return {
        "success": True,
        "data": history_data
    }

@router.delete("/results/{analysis_id}")
async def delete_result(
    analysis_id: str,
    user: AuthenticatedUser = Depends(get_current_user)
):
    # Verify ownership before deletion
    record = db_service.get_analysis_by_id(analysis_id, user_id=user.id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Result with ID '{analysis_id}' not found."}
        )

    # 1. Cascade delete storage files (original image, processed artifacts, report)
    await storage_service.delete_analysis_files(user_id=user.id, analysis_id=analysis_id)

    # 2. Delete database record
    success = await db_service.delete_analysis(analysis_id, user_id=user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Result with ID '{analysis_id}' could not be deleted."}
        )

    return {
        "success": True,
        "message": f"Analysis '{analysis_id}' and all associated files deleted successfully."
    }
