"""
Supabase Storage Service with Robust Local Fallback
===================================================
Manages private bucket storage for ImageGuard.
Enforces:
- Private bucket 'imageguard'
- Structure: {user_id}/uploads/{analysis_id}.{extension}
             {user_id}/processed/{analysis_id}_gradcam.{extension}
             {user_id}/reports/{analysis_id}.pdf
- Randomized / analysis_id based filenames (never client filenames)
- Temporary signed URLs with configurable expiration
- Cascade deletion of upload + processed artifacts + reports
- Automatic cleanup on analysis / DB insertion failures
"""

import httpx
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

LOCAL_STORAGE_ROOT = Path("data/storage")

class StorageService:
    def __init__(self):
        self.bucket = settings.SUPABASE_STORAGE_BUCKET or "imageguard"
        self.local_bucket_dir = LOCAL_STORAGE_ROOT / self.bucket
        self.local_bucket_dir.mkdir(parents=True, exist_ok=True)

    def is_supabase_configured(self) -> bool:
        return bool(settings.SUPABASE_URL and (settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY))

    def _get_headers(self) -> dict:
        api_key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY
        return {
            "Authorization": f"Bearer {api_key}",
            "apikey": api_key
        }

    async def upload_file(
        self,
        user_id: str,
        category: str,
        filename_id: str,
        file_bytes: bytes,
        mime_type: str = "image/jpeg"
    ) -> str:
        """
        Uploads file to private storage.
        Path format: {user_id}/{category}/{filename_id}
        Returns the canonical storage_path.
        """
        clean_user_id = str(user_id).strip() or "anonymous"
        storage_path = f"{clean_user_id}/{category}/{filename_id}"

        # 1. Always save locally as fallback and cache
        local_file_path = self.local_bucket_dir / clean_user_id / category / filename_id
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_file_path, "wb") as f:
            f.write(file_bytes)

        # 2. Upload to Supabase Storage if configured
        if self.is_supabase_configured():
            upload_url = f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/{self.bucket}/{storage_path}"
            headers = self._get_headers()
            headers["Content-Type"] = mime_type
            headers["x-upsert"] = "true"

            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(upload_url, headers=headers, content=file_bytes)
                    if resp.status_code in (200, 201):
                        logger.info(f"[StorageService] Successfully uploaded to Supabase Storage: {storage_path}")
                    else:
                        logger.warning(
                            f"[StorageService] Supabase upload returned status {resp.status_code}: {resp.text}. "
                            f"Retaining local copy."
                        )
            except Exception as e:
                logger.error(f"[StorageService] Supabase upload failed ({e}). Local copy retained.")

        return storage_path

    async def create_signed_url(self, storage_path: str, expires_in: Optional[int] = None) -> str:
        """
        Generates a temporary signed URL for an image/report.
        Never exposes public URLs for private buckets.
        Falls back to local API endpoint if running offline.
        """
        ttl = expires_in or settings.SIGNED_URL_EXPIRES_IN

        if self.is_supabase_configured():
            sign_url = f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/sign/{self.bucket}/{storage_path}"
            headers = self._get_headers()
            headers["Content-Type"] = "application/json"

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(sign_url, headers=headers, json={"expiresIn": ttl})
                    if resp.status_code == 200:
                        data = resp.json()
                        signed_path = data.get("signedURL")
                        if signed_path:
                            base = settings.SUPABASE_URL.rstrip('/')
                            # If signed_path starts with /object/sign or similar
                            if not signed_path.startswith("http"):
                                full_url = f"{base}/storage/v1{signed_path}"
                            else:
                                full_url = signed_path
                            return full_url
                    else:
                        logger.warning(f"[StorageService] Failed to create signed URL ({resp.status_code}): {resp.text}")
            except Exception as e:
                logger.error(f"[StorageService] Supabase sign request error: {e}")

        # Local fallback stream URL
        return f"/api/storage/{storage_path}"

    async def delete_file(self, storage_path: str) -> bool:
        """
        Deletes a specific file from storage.
        """
        deleted_supabase = False
        if self.is_supabase_configured():
            delete_url = f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/{self.bucket}"
            headers = self._get_headers()
            headers["Content-Type"] = "application/json"

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.request(
                        "DELETE",
                        delete_url,
                        headers=headers,
                        json={"prefixes": [storage_path]}
                    )
                    deleted_supabase = resp.status_code == 200
            except Exception as e:
                logger.error(f"[StorageService] Failed to delete from Supabase ({storage_path}): {e}")

        # Local delete
        local_path = self.local_bucket_dir / storage_path
        if local_path.exists():
            try:
                local_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete local file {local_path}: {e}")

        return deleted_supabase or True

    async def delete_analysis_files(
        self,
        user_id: str,
        analysis_id: str,
        extension: str = "jpg"
    ) -> None:
        """
        Cascades deletion for all associated storage files for an analysis:
        - Upload image: {user_id}/uploads/{analysis_id}.{ext}
        - Processed Grad-CAM: {user_id}/processed/{analysis_id}_gradcam.png
        - Processed ELA: {user_id}/processed/{analysis_id}_ela.png
        - PDF Report: {user_id}/reports/{analysis_id}.pdf
        """
        possible_paths = [
            f"{user_id}/uploads/{analysis_id}.{extension}",
            f"{user_id}/uploads/{analysis_id}.jpg",
            f"{user_id}/uploads/{analysis_id}.jpeg",
            f"{user_id}/uploads/{analysis_id}.png",
            f"{user_id}/uploads/{analysis_id}.webp",
            f"{user_id}/processed/{analysis_id}_gradcam.png",
            f"{user_id}/processed/{analysis_id}_gradcam.jpg",
            f"{user_id}/processed/{analysis_id}_ela.png",
            f"{user_id}/processed/{analysis_id}_ela.jpg",
            f"{user_id}/reports/{analysis_id}.pdf",
        ]

        for p in possible_paths:
            await self.delete_file(p)

    def get_local_file_bytes(self, storage_path: str) -> Optional[bytes]:
        """Reads local copy of file bytes if present."""
        local_path = self.local_bucket_dir / storage_path
        if local_path.exists():
            with open(local_path, "rb") as f:
                return f.read()
        return None

storage_service = StorageService()
