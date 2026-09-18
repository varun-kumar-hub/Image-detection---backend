from typing import Optional
import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.app.api.dependencies import get_current_user
from backend.app.core.security import AuthenticatedUser
from backend.app.core.config import settings
from backend.app.services.gemini_service import encrypt_key, decrypt_key, test_key

router = APIRouter(prefix="/settings", tags=["Settings"])

class BackupSettingsUpdate(BaseModel):
    enabled: bool
    reference: Optional[str] = None

class GeminiKeyUpdate(BaseModel):
    api_key: str

def _headers() -> dict:
    key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY
    return {"Authorization": f"Bearer {key}", "apikey": key, "Content-Type": "application/json"}

async def fetch_backup_settings(user_id: str) -> dict:
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/backup_settings"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(url, headers=_headers(), params={"user_id": f"eq.{user_id}", "select": "enabled,reference_label"})
    except httpx.HTTPError:
        return {"enabled": False, "reference": None}
    if response.status_code not in (200, 206):
        return {"enabled": False, "reference": None}
    rows = response.json()
    row = rows[0] if rows else {}
    return {"enabled": bool(row.get("enabled")), "reference": row.get("reference_label")}

async def save_backup_evaluation(analysis_id: str, user_id: str, reference: str, prediction: str, probability: float) -> None:
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/backup_evaluations"
    body = {"analysis_id": analysis_id, "user_id": user_id, "reference_label": reference, "model_prediction": prediction, "model_probability": probability, "is_match": prediction == ("real" if reference in ("authentic", "real") else "ai_generated")}
    async with httpx.AsyncClient(timeout=8.0) as client:
        await client.post(url, headers={**_headers(), "Prefer": "return=minimal"}, json=body)

@router.get("/backup")
async def get_backup_settings(user: AuthenticatedUser = Depends(get_current_user)):
    return await fetch_backup_settings(user.id)

@router.put("/backup")
async def update_backup_settings(payload: BackupSettingsUpdate, user: AuthenticatedUser = Depends(get_current_user)):
    reference = payload.reference if payload.enabled else None
    if payload.enabled and reference not in ("authentic", "real", "ai_generated"):
        raise HTTPException(400, "A valid backup reference is required when Backup Mode is enabled.")
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/backup_settings"
    body = {"user_id": user.id, "enabled": payload.enabled, "reference_label": reference}
    headers = _headers(); headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(url, headers=headers, json=body)
    if response.status_code not in (200, 201):
        raise HTTPException(503, "Backup settings could not be saved.")
    return {"enabled": payload.enabled, "reference": reference}

@router.get("/gemini")
async def get_gemini_settings(user: AuthenticatedUser = Depends(get_current_user)):
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/user_ai_settings"
    async with httpx.AsyncClient(timeout=8) as client:
        response = await client.get(url, headers=_headers(), params={"user_id": f"eq.{user.id}", "select": "gemini_enabled,gemini_model,gemini_api_key_encrypted"})
    row = response.json()[0] if response.status_code in (200, 206) and response.json() else {}
    encrypted = row.get("gemini_api_key_encrypted")
    masked = None
    if encrypted:
        try:
            raw = decrypt_key(encrypted)
            masked = "••••••••" + raw[-4:]
        except RuntimeError:
            masked = "••••••••"
    return {"configured": bool(encrypted), "enabled": bool(row.get("gemini_enabled")), "model": row.get("gemini_model", settings.GEMINI_MODEL), "masked_key": masked}

@router.post("/gemini")
async def save_gemini_settings(payload: GeminiKeyUpdate, user: AuthenticatedUser = Depends(get_current_user)):
    key = payload.api_key.strip()
    if not key or len(key) < 10:
        raise HTTPException(400, "Enter a valid Gemini API key.")
    try:
        await test_key(key)
        encrypted = encrypt_key(key)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/user_ai_settings"
    headers = {**_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(url, headers=headers, json={"user_id": user.id, "gemini_api_key_encrypted": encrypted, "gemini_enabled": True, "gemini_model": settings.GEMINI_MODEL})
    if response.status_code not in (200, 201):
        raise HTTPException(503, "Gemini settings could not be saved.")
    return {"configured": True, "enabled": True, "masked_key": "••••••••" + key[-4:]}

@router.post("/gemini/test")
async def test_gemini_settings(user: AuthenticatedUser = Depends(get_current_user)):
    data = await get_gemini_settings(user)
    if not data["configured"]:
        raise HTTPException(400, "Gemini API is not configured.")
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/user_ai_settings"
    async with httpx.AsyncClient(timeout=8) as client:
        response = await client.get(url, headers=_headers(), params={"user_id": f"eq.{user.id}", "select": "gemini_api_key_encrypted"})
    encrypted = response.json()[0].get("gemini_api_key_encrypted") if response.status_code == 200 and response.json() else None
    try:
        await test_key(decrypt_key(encrypted))
    except Exception:
        raise HTTPException(400, "Gemini connection failed. Check the key and Google AI Studio access.")
    return {"success": True, "message": "Gemini API connection successful."}

@router.delete("/gemini")
async def remove_gemini_settings(user: AuthenticatedUser = Depends(get_current_user)):
    url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/user_ai_settings"
    async with httpx.AsyncClient(timeout=8) as client:
        response = await client.delete(url, headers=_headers(), params={"user_id": f"eq.{user.id}"})
    if response.status_code not in (200, 204):
        raise HTTPException(503, "Gemini settings could not be removed.")
    return {"success": True}
