"""
Database Service (Supabase PostgreSQL + Local SQLite Fallback)
==============================================================
Manages persistence for user analyses, uploads, and manipulation details.
Enforces strict user ownership and history isolation.
"""

import os
import json
import sqlite3
import httpx
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.app.core.config import settings

logger = logging.getLogger(__name__)
LOCAL_DB_PATH = Path("data/imageguard_local.db")

class DatabaseService:
    def __init__(self):
        self._init_local_db()

    def _init_local_db(self):
        LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_records (
                    id TEXT PRIMARY KEY,
                    upload_id TEXT,
                    user_id TEXT,
                    storage_path TEXT,
                    filename TEXT,
                    classification TEXT,
                    ai_probability REAL,
                    real_probability REAL,
                    confidence TEXT,
                    confidence_explanation TEXT,
                    processing_time_ms INTEGER,
                    model_name TEXT,
                    model_version TEXT,
                    image_info TEXT,
                    manipulation TEXT,
                    explanation TEXT,
                    created_at TEXT
                )
            """)
            # Check if columns exist (for existing databases)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(analysis_records)")
            cols = [col[1] for col in cursor.fetchall()]
            if "storage_path" not in cols:
                conn.execute("ALTER TABLE analysis_records ADD COLUMN storage_path TEXT")
            if "filename" not in cols:
                conn.execute("ALTER TABLE analysis_records ADD COLUMN filename TEXT")
            if "explanation" not in cols:
                conn.execute("ALTER TABLE analysis_records ADD COLUMN explanation TEXT")
            if "ground_truth" not in cols:
                conn.execute("ALTER TABLE analysis_records ADD COLUMN ground_truth TEXT")
            if "is_evaluation" not in cols:
                conn.execute("ALTER TABLE analysis_records ADD COLUMN is_evaluation INTEGER DEFAULT 0")
            if "is_correct" not in cols:
                conn.execute("ALTER TABLE analysis_records ADD COLUMN is_correct INTEGER")
            conn.commit()

    def is_supabase_configured(self) -> bool:
        return bool(settings.SUPABASE_URL and (settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY))

    def _get_supabase_headers(self) -> dict:
        api_key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY
        return {
            "Authorization": f"Bearer {api_key}",
            "apikey": api_key,
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

    async def save_analysis(self, record: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Saves analysis record with strict user_id association."""
        clean_user_id = str(user_id).strip() or "anonymous"
        record["user_id"] = clean_user_id

        # 1. Attempt Supabase PostgREST sync if configured
        if self.is_supabase_configured():
            rest_url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/analysis_results"
            payload: Dict[str, Any] = {
                "classification": record["classification"],
                "ai_probability": record["ai_probability"],
                "real_probability": record["real_probability"],
                "confidence": record["confidence"],
                "processing_time_ms": record["processing_time_ms"],
                "model_version": record.get("model_version", "v1.0"),
            }
            try:
                import uuid as _uuid
                if record.get("id"):
                    _uuid.UUID(str(record["id"]))
                    payload["id"] = record["id"]
            except Exception:
                pass

            try:
                import uuid as _uuid
                if record.get("upload_id"):
                    _uuid.UUID(str(record["upload_id"]))
                    payload["upload_id"] = record["upload_id"]
            except Exception:
                pass

            if clean_user_id and clean_user_id != "anonymous" and not clean_user_id.startswith("00000000-"):
                try:
                    import uuid as _uuid
                    _uuid.UUID(clean_user_id)
                    payload["user_id"] = clean_user_id
                except Exception:
                    pass

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(rest_url, headers=self._get_supabase_headers(), json=payload)
                    if resp.status_code not in (200, 201):
                        logger.warning(f"[DatabaseService] Supabase insert warning ({resp.status_code}): {resp.text}")
            except Exception as e:
                logger.error(f"[DatabaseService] Supabase insert failed: {e}")

        # 2. Local SQLite persistence (guaranteed backup & offline support)
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO analysis_records (
                    id, upload_id, user_id, storage_path, filename, classification, ai_probability, real_probability,
                    confidence, confidence_explanation, processing_time_ms, model_name,
                    model_version, image_info, manipulation, explanation, ground_truth, is_evaluation, is_correct, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record["id"],
                record.get("upload_id", ""),
                clean_user_id,
                record.get("storage_path", ""),
                record.get("filename") or record.get("image_info", {}).get("filename", ""),
                record["classification"],
                record["ai_probability"],
                record["real_probability"],
                record["confidence"],
                record.get("confidence_explanation", ""),
                record["processing_time_ms"],
                record.get("model_name", "EfficientNet-B0"),
                record.get("model_version", "v1.0"),
                json.dumps(record.get("image_info", {})),
                json.dumps(record.get("manipulation", {})),
                json.dumps(record.get("explanation", {})),
                record.get("ground_truth"),
                1 if record.get("is_evaluation") else 0,
                1 if record.get("is_correct") is True else 0 if record.get("is_correct") is False else None,
                record.get("created_at", datetime.utcnow().isoformat())
            ))
            conn.commit()

        return record

    def get_analysis_by_id(self, analysis_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves an analysis by ID.
        If user_id is provided, verifies that the analysis belongs to this user.
        """
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if user_id:
                cursor.execute(
                    "SELECT * FROM analysis_records WHERE id = ? AND (user_id = ? OR user_id = 'anonymous')",
                    (analysis_id, user_id)
                )
            else:
                cursor.execute("SELECT * FROM analysis_records WHERE id = ?", (analysis_id,))

            row = cursor.fetchone()
            if not row:
                return None

            data = dict(row)
            data["image_info"] = json.loads(data["image_info"]) if data.get("image_info") else {}
            data["manipulation"] = json.loads(data["manipulation"]) if data.get("manipulation") else {}
            data["explanation"] = json.loads(data["explanation"]) if data.get("explanation") else {}
            data["is_evaluation"] = bool(data.get("is_evaluation"))
            data["is_correct"] = True if data.get("is_correct") == 1 else False if data.get("is_correct") == 0 else None
            return data

    def list_analyses(
        self,
        user_id: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
        classification: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Lists analysis records strictly isolated for the user.
        User A will NEVER see User B's records.
        """
        offset = (page - 1) * limit
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM analysis_records WHERE 1=1"
            params: List[Any] = []

            if user_id:
                query += " AND (user_id = ?)"
                params.append(user_id)

            if classification:
                query += " AND classification = ?"
                params.append(classification)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Total count
            count_query = "SELECT COUNT(*) FROM analysis_records WHERE 1=1"
            count_params: List[Any] = []
            if user_id:
                count_query += " AND (user_id = ?)"
                count_params.append(user_id)
            if classification:
                count_query += " AND classification = ?"
                count_params.append(classification)

            cursor.execute(count_query, count_params)
            total = cursor.fetchone()[0]

            records = []
            for r in rows:
                item = dict(r)
                item["image_info"] = json.loads(item["image_info"]) if item.get("image_info") else {}
                item["manipulation"] = json.loads(item["manipulation"]) if item.get("manipulation") else {}
                item["explanation"] = json.loads(item["explanation"]) if item.get("explanation") else {}
                item["is_evaluation"] = bool(item.get("is_evaluation"))
                item["is_correct"] = True if item.get("is_correct") == 1 else False if item.get("is_correct") == 0 else None
                records.append(item)

            return {
                "total": total,
                "page": page,
                "limit": limit,
                "items": records
            }

    async def delete_analysis(self, analysis_id: str, user_id: Optional[str] = None) -> bool:
        """
        Deletes analysis record after ownership check.
        Also deletes from Supabase if configured.
        """
        # First check ownership
        record = self.get_analysis_by_id(analysis_id, user_id=user_id)
        if not record:
            return False

        if self.is_supabase_configured():
            rest_url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/analysis_results?id=eq.{analysis_id}"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.delete(rest_url, headers=self._get_supabase_headers())
            except Exception as e:
                logger.error(f"[DatabaseService] Supabase delete error: {e}")

        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("DELETE FROM analysis_records WHERE id = ? AND user_id = ?", (analysis_id, user_id))
            else:
                cursor.execute("DELETE FROM analysis_records WHERE id = ?", (analysis_id,))
            conn.commit()
            return cursor.rowcount > 0

db_service = DatabaseService()
