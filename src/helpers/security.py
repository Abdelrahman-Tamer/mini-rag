from fastapi import Header, HTTPException, status, Depends
from typing import Optional
from helpers.config import get_settings, Settings
import logging

logger = logging.getLogger('uvicorn.error')

async def verify_admin_access(
    x_admin_api_key: Optional[str] = Header(None, alias="X-Admin-Api-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    app_settings: Settings = Depends(get_settings)
) -> bool:
    """
    Validates administrative access for privileged internal RAG ingestion and sync endpoints.
    Accepts:
    1. 'X-Admin-Api-Key' header
    2. 'Authorization: Bearer <token>' header
    Fails closed if no administrative credentials are configured.
    Never falls back to predictable defaults in production.
    """
    valid_keys = set()
    if app_settings.ADMIN_API_KEY and app_settings.ADMIN_API_KEY.strip():
        valid_keys.add(app_settings.ADMIN_API_KEY.strip())
    if app_settings.SYNC_API_KEY and app_settings.SYNC_API_KEY.strip():
        valid_keys.add(app_settings.SYNC_API_KEY.strip())

    # Only in explicit development mode with DEV_ALLOW_DEFAULT_KEY enabled may a local dev key be used
    env = (app_settings.ENVIRONMENT or "development").strip().lower()
    if env == "development" and getattr(app_settings, "DEV_ALLOW_DEFAULT_KEY", False):
        valid_keys.add("nileguide-admin-secret-2026")

    # If no valid keys are configured, fail closed immediately
    if not valid_keys:
        logger.error("SECURITY ALERT: Administrative endpoint accessed but NO valid admin API keys are configured. Failing closed.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Administrative credentials are not configured on this server."
        )

    provided_token = None
    if x_admin_api_key and x_admin_api_key.strip():
        provided_token = x_admin_api_key.strip()
    elif authorization and authorization.strip().lower().startswith("bearer "):
        provided_token = authorization.strip()[7:].strip()

    if not provided_token or provided_token not in valid_keys:
        logger.warning("Unauthorized access attempt to privileged internal endpoint.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid administrative API key is required."
        )

    return True
