import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure src is on sys.path
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from main import app
from helpers.config import get_settings, Settings
from unittest.mock import MagicMock, AsyncMock

client = TestClient(app)

def test_public_root_endpoint_accessible():
    """Public health and discovery endpoint must be accessible without credentials."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "endpoints" in data

def test_data_files_endpoint_unauthenticated_rejected():
    """Direct unauthenticated access to /api/v1/data/files must be rejected with 401."""
    response = client.get("/api/v1/data/files")
    assert response.status_code == 401
    assert "Unauthorized" in response.json().get("detail", "")

def test_data_files_endpoint_invalid_key_rejected():
    """Access with an invalid API key must be rejected with 401."""
    response = client.get("/api/v1/data/files", headers={"X-Admin-Api-Key": "completely-invalid-key"})
    assert response.status_code == 401

def test_multi_upload_endpoint_unauthenticated_rejected():
    """Unauthenticated multi-upload requests must be rejected with 401."""
    response = client.post("/api/v1/data/multi-upload")
    assert response.status_code == 401

def test_delete_file_endpoint_unauthenticated_rejected():
    """Unauthenticated file deletion requests must be rejected with 401."""
    response = client.delete("/api/v1/data/file/test_file_id")
    assert response.status_code == 401

def test_sync_activity_unauthenticated_rejected():
    """Activity synchronization without internal admin credentials must be rejected with 401."""
    payload = {
        "activity_id": 99,
        "title": "Unauthorized Activity",
        "action": "upsert"
    }
    response = client.post("/api/v1/nlp/sync/activity", json=payload)
    assert response.status_code == 401

def test_sync_activity_invalid_credentials_rejected():
    """Activity synchronization with incorrect internal credentials must be rejected with 401."""
    payload = {
        "activity_id": 99,
        "title": "Unauthorized Activity",
        "action": "upsert"
    }
    response = client.post(
        "/api/v1/nlp/sync/activity",
        json=payload,
        headers={"X-Admin-Api-Key": "wrong-secret-token"}
    )
    assert response.status_code == 401

def test_privileged_endpoints_authorized_with_valid_key():
    """When a valid administrative key is configured, authorized requests pass verify_admin_access."""
    mock_settings = Settings(
        ENVIRONMENT="development",
        ADMIN_API_KEY="valid-staging-test-key-32chars!",
        SYNC_API_KEY="valid-staging-test-key-32chars!"
    )
    app.dependency_overrides[get_settings] = lambda: mock_settings

    # Mock MongoDB on app.db_client to verify endpoint completes successfully
    mock_collection = MagicMock()
    mock_find = MagicMock()
    mock_sort = MagicMock()
    mock_sort.to_list = AsyncMock(return_value=[])
    mock_find.sort.return_value = mock_sort
    mock_collection.find.return_value = mock_find
    mock_collection.create_index = AsyncMock()

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_db.list_collection_names = AsyncMock(return_value=["assets"])

    original_db = getattr(app, "db_client", None)
    app.db_client = mock_db

    try:
        # Request with correct header
        response = client.get(
            "/api/v1/data/files",
            headers={"X-Admin-Api-Key": "valid-staging-test-key-32chars!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "assets" in data
        assert isinstance(data["assets"], list)
    finally:
        app.dependency_overrides.clear()
        app.db_client = original_db
