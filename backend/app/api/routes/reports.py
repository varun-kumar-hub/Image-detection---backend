"""
PDF Report Route
================
Generates and serves PDF reports with user ownership checks.
"""

from fastapi import APIRouter, HTTPException, Response, Depends, status
from typing import Optional

from backend.app.core.security import AuthenticatedUser
from backend.app.api.dependencies import get_optional_current_user
from backend.app.services.supabase_service import db_service
from backend.app.services.report_service import report_service
from backend.app.services.storage_service import storage_service

router = APIRouter(prefix="", tags=["Reports"])

@router.get("/results/{analysis_id}/report")
async def download_pdf_report(
    analysis_id: str,
    user: Optional[AuthenticatedUser] = Depends(get_optional_current_user)
):
    user_id = user.id if user else None
    record = db_service.get_analysis_by_id(analysis_id, user_id=user_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Analysis '{analysis_id}' not found."}
        )

    # Generate PDF report bytes
    pdf_bytes = report_service.generate_pdf_report(record)

    # Persist report to storage if user is authenticated
    if user_id and user_id != "anonymous":
        try:
            await storage_service.upload_file(
                user_id=user_id,
                category="reports",
                filename_id=f"{analysis_id}.pdf",
                file_bytes=pdf_bytes,
                mime_type="application/pdf"
            )
        except Exception:
            pass  # Non-blocking for immediate download

    headers = {
        "Content-Disposition": f"attachment; filename=ImageGuard_{analysis_id}.pdf"
    }

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers=headers
    )
