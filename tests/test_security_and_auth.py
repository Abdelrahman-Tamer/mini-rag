import pytest
import asyncio
from fastapi import HTTPException
from helpers.security import verify_admin_access
from helpers.config import Settings, validate_admin_security_config

def test_verify_admin_access_missing_token():
    settings = Settings(ADMIN_API_KEY="test-admin-key-123456", SYNC_API_KEY="test-sync-key-123456")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(verify_admin_access(x_admin_api_key=None, authorization=None, app_settings=settings))
    assert exc_info.value.status_code == 401

def test_verify_admin_access_invalid_token():
    settings = Settings(ADMIN_API_KEY="test-admin-key-123456", SYNC_API_KEY="test-sync-key-123456")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(verify_admin_access(x_admin_api_key="wrong-key", authorization=None, app_settings=settings))
    assert exc_info.value.status_code == 401

def test_verify_admin_access_valid_x_admin_api_key():
    settings = Settings(ADMIN_API_KEY="test-admin-key-123456", SYNC_API_KEY="test-sync-key-123456")
    result = asyncio.run(verify_admin_access(x_admin_api_key="test-admin-key-123456", authorization=None, app_settings=settings))
    assert result is True

def test_verify_admin_access_valid_bearer_token():
    settings = Settings(ADMIN_API_KEY="test-admin-key-123456", SYNC_API_KEY="test-sync-key-123456")
    result = asyncio.run(verify_admin_access(x_admin_api_key=None, authorization="Bearer test-sync-key-123456", app_settings=settings))
    assert result is True

def test_verify_admin_access_unconfigured_fails_closed():
    """Missing administrative keys must fail closed; no silent fallback to predictable keys."""
    settings = Settings(ENVIRONMENT="development", ADMIN_API_KEY="", SYNC_API_KEY="", DEV_ALLOW_DEFAULT_KEY=False)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(verify_admin_access(x_admin_api_key="nileguide-admin-secret-2026", authorization=None, app_settings=settings))
    assert exc_info.value.status_code == 401

def test_validate_admin_security_config_prod_missing_keys_raises():
    """Production must fail closed at startup if admin keys are missing."""
    settings = Settings(ENVIRONMENT="production", ADMIN_API_KEY="", SYNC_API_KEY="")
    with pytest.raises(ValueError, match="CRITICAL SECURITY CONFIGURATION ERROR"):
        validate_admin_security_config(settings)

def test_validate_admin_security_config_prod_predictable_key_raises():
    """Production must fail closed at startup if predictable development keys are configured."""
    settings = Settings(ENVIRONMENT="production", ADMIN_API_KEY="nileguide-admin-secret-2026")
    with pytest.raises(ValueError, match="Predictable or weak administrative key detected"):
        validate_admin_security_config(settings)

def test_validate_admin_security_config_prod_valid_key_succeeds():
    """Production startup succeeds when strong random key >= 16 chars is configured."""
    settings = Settings(ENVIRONMENT="production", ADMIN_API_KEY="strong-random-key-for-prod-32chars!")
    # Should not raise
    validate_admin_security_config(settings)
