from typing import Optional
import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.app.api.dependencies import get_current_user
from backend.app.core.security import AuthenticatedUser
from backend.app.core.config import settings

router = APIRouter(prefix="/settings", tags=["Settings"])

class BackupSettingsUpdate(BaseModel):
    enabled: bool
    reference: Optional[str] = None

def _headers() -> dict:
    key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY
    return {"Authorization": f"Bearer {key}", "apikey": key, "Content-Type": "application/json"}

async def fetch_backup_settings(user_id: str) -> dict:
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/backup_settings"
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(url, headers=_headers(), params={"user_id": f"eq.{user_id}", "select": "enabled,reference_label"})
    if response.status_code not in (200, 206):
        return {"enabled": False, "reference": None}
    rows = response.json()
    row = rows[0] if rows else {}
    return {"enabled": bool(row.get("enabled")), "reference": row.get("reference_label")}

async def save_backup_evaluation(analysis_id: str, user_id: str, reference: str, prediction: str, probability: float) -> None:
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/backup_evaluations"
    body = {"analysis_id": analysis_id, "user_id": user_id, "reference_label": reference, "model_prediction": prediction, "model_probability": probability, "is_match": prediction == ("real" if reference == "authentic" else "ai_generated")}
    async with httpx.AsyncClient(timeout=8.0) as client:
        await client.post(url, headers={**_headers(), "Prefer": "return=minimal"}, json=body)

@router.get("/backup")
async def get_backup_settings(user: AuthenticatedUser = Depends(get_current_user)):
    return await fetch_backup_settings(user.id)

@router.put("/backup")
async def update_backup_settings(payload: BackupSettingsUpdate, user: AuthenticatedUser = Depends(get_current_user)):
    reference = payload.reference if payload.enabled else None
    if payload.enabled and reference not in ("authentic", "ai_generated"):
        raise HTTPException(400, "A valid backup reference is required when Backup Mode is enabled.")
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/backup_settings"
    body = {"user_id": user.id, "enabled": payload.enabled, "reference_label": reference}
    headers = _headers(); headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(url, headers=headers, json=body)
    if response.status_code not in (200, 201):
        raise HTTPException(503, "Backup settings could not be saved.")
    return {"enabled": payload.enabled, "reference": reference}
