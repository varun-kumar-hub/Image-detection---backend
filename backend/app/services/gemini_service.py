import base64
import hashlib
import httpx
from cryptography.fernet import Fernet, InvalidToken
from backend.app.core.config import settings

def _cipher() -> Fernet:
    secret = settings.GEMINI_USER_KEY_ENCRYPTION_SECRET
    if not secret:
        raise RuntimeError("Gemini encryption secret is not configured")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)

def encrypt_key(value: str) -> str:
    return _cipher().encrypt(value.strip().encode()).decode()

def decrypt_key(value: str) -> str:
    try:
        return _cipher().decrypt(value.encode()).decode()
    except InvalidToken as exc:
        raise RuntimeError("Stored Gemini key could not be decrypted") from exc

async def test_key(api_key: str, model: str | None = None) -> None:
    selected_model = model or settings.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent"
    payload = {"contents": [{"parts": [{"text": "Reply with: connected"}]}]}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, params={"key": api_key}, json=payload)
    if response.status_code != 200:
        raise RuntimeError("Gemini connection failed")
