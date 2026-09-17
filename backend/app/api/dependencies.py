"""
FastAPI Security & Authentication Dependencies
==============================================
Provides reusable dependencies for authenticated routes.
"""

from typing import Optional
from fastapi import Header, HTTPException, status
from backend.app.core.security import verify_supabase_token, AuthenticatedUser

async def get_current_user(authorization: Optional[str] = Header(None)) -> AuthenticatedUser:
    """
    Mandatory authentication dependency.
    Validates the Bearer token in the Authorization header.
    Returns the verified AuthenticatedUser or raises HTTP 401.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Authentication required. Missing Authorization header."
            }
        )

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid Authorization header format. Expected 'Bearer <token>'."
            }
        )

    token = parts[1]
    user = await verify_supabase_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid, revoked, or expired authentication session."
            }
        )

    return user

async def get_optional_current_user(authorization: Optional[str] = Header(None)) -> Optional[AuthenticatedUser]:
    """
    Optional authentication dependency.
    If a valid Bearer token is provided, returns AuthenticatedUser.
    Otherwise returns None (enabling guest operations without error).
    """
    if not authorization:
        return None

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    token = parts[1]
    return await verify_supabase_token(token)
