import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
import uuid
import json

from main import app
from controllers.NLPController import NLPController
from helpers.rate_limiter import InMemorySlidingWindowLimiter, get_client_ip, check_rate_limit
from helpers.config import get_settings, Settings
from models.db_schemes import DataChunk
from stores.vectordb.providers.QdrantDBProvider import QdrantDBProvider


@pytest.fixture
def mock_app_state():
    client = TestClient(app)
    app.vectordb_client = MagicMock()
    app.generation_client = MagicMock()
    app.embedding_client = MagicMock()
    app.template_parser = MagicMock()
    app.db_client = MagicMock()
    yield client
    app.dependency_overrides.clear()


# --- P0-RAG-01 & P1-RAG-03: Vector Indexing & ID Collision Tests ---

def test_nlp_controller_index_chunks_generates_deterministic_uuids():
    mock_vectordb = MagicMock()
    mock_vectordb.insert_many.return_value = True

    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3]

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    chunks = [
        DataChunk(chunk_text="First historical chunk", chunk_metadata={"asset_id": "test_asset_123"}, chunk_order=1),
        DataChunk(chunk_text="Second historical chunk", chunk_metadata={"asset_id": "test_asset_123"}, chunk_order=2),
    ]

    result = nlp.index_chunks(chunks)

    assert result["status"] == "indexed"
    assert result["indexed"] == 2
    assert result["asset_id"] == "test_asset_123"

    mock_vectordb.insert_many.assert_called_once()
    call_kwargs = mock_vectordb.insert_many.call_args[1]
    record_ids = call_kwargs["record_ids"]

    assert len(record_ids) == 2
    expected_id_1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "nileguide_asset_test_asset_123_chunk_1"))
    expected_id_2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "nileguide_asset_test_asset_123_chunk_2"))
    assert record_ids[0] == expected_id_1
    assert record_ids[1] == expected_id_2

    # Retrying with same chunks generates exact same point IDs (idempotency)
    result_retry = nlp.index_chunks(chunks)
    assert result_retry["status"] == "indexed"
    retry_ids = mock_vectordb.insert_many.call_args_list[1][1]["record_ids"]
    assert retry_ids == record_ids


def test_nlp_controller_delete_file_vectors():
    mock_vectordb = MagicMock()
    mock_vectordb.delete_by_filter.return_value = True

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )

    result = nlp.delete_file_vectors(asset_id="test_asset_123")
    assert result["status"] == "deleted"
    assert result["deleted"] is True
    assert result["asset_id"] == "test_asset_123"

    mock_vectordb.delete_by_filter.assert_called_once_with(
        collection_name=nlp.collection_name,
        filter={"asset_id": "test_asset_123"}
    )


def test_qdrant_provider_insert_many_prevents_point_id_collision():
    provider = QdrantDBProvider(qdrant_db_url="http://localhost:6333", qdrant_db_api_key="none", distance_method="cosine")
    provider.client = MagicMock()
    provider.is_collection_existed = MagicMock(return_value=True)

    texts = ["Snippet A", "Snippet B"]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    # Omit record_ids -> Provider must generate UUIDs instead of [0, 1]
    provider.insert_many("tourism_knowledge_base", texts=texts, vectors=vectors, record_ids=None)

    assert provider.client.upload_records.called
    uploaded_records = provider.client.upload_records.call_args[1]["records"]
    assert len(uploaded_records) == 2

    # Assert IDs are non-empty UUID strings, NOT integer 0, 1
    id1 = uploaded_records[0].id
    id2 = uploaded_records[1].id
    assert isinstance(id1, str)
    assert isinstance(id2, str)
    assert id1 != id2
    # Verify valid UUID format
    uuid.UUID(id1)
    uuid.UUID(id2)


# --- P0-SEC-01: Admin Authentication Tests ---

def test_admin_route_unauthenticated_rejected(mock_app_state):
    response = mock_app_state.get("/api/v1/data/files")
    assert response.status_code == 401


def test_admin_route_invalid_key_rejected(mock_app_state):
    headers = {"X-Admin-Api-Key": "completely-invalid-key-attempt"}
    response = mock_app_state.get("/api/v1/data/files", headers=headers)
    assert response.status_code == 401


def test_admin_route_valid_key_accepted(mock_app_state):
    def get_override_settings():
        return Settings(
            ADMIN_API_KEY="valid-test-admin-secret-2026",
            ENVIRONMENT="development",
            DEV_ALLOW_DEFAULT_KEY=False
        )

    app.dependency_overrides[get_settings] = get_override_settings

    # Mock mongo collection
    mock_collection = MagicMock()
    mock_collection.find.return_value.sort.return_value.to_list = AsyncMock(return_value=[])
    app.db_client = MagicMock()
    app.db_client.list_collection_names = AsyncMock(return_value=["assets"])
    app.db_client.__getitem__.return_value = mock_collection

    headers = {"X-Admin-Api-Key": "valid-test-admin-secret-2026"}
    response = mock_app_state.get("/api/v1/data/files", headers=headers)
    assert response.status_code == 200


# --- P0-SEC-02: Public Chat Abuse Protection & Rate Limiting Tests ---

def test_public_chat_remains_accessible_without_auth(mock_app_state):
    with patch.object(NLPController, "chat", return_value=[]), \
         patch.object(NLPController, "generate_chat_answer", return_value=("Egypt is home to the majestic Pyramids.", "prompt", False)), \
         patch.object(NLPController, "is_greeting_query", return_value=False):

        payload = {"text": "Tell me about the Pyramids of Giza"}
        response = mock_app_state.post("/api/v1/nlp/chat", json=payload)
        assert response.status_code == 200
        assert "Egypt" in response.json()["answer"]


def test_public_chat_rate_limiting():
    limiter = InMemorySlidingWindowLimiter()
    client_key = "test_ip_192_168_1_50"

    allowed1, _ = limiter.is_allowed(client_key, max_requests=3, window_seconds=10)
    allowed2, _ = limiter.is_allowed(client_key, max_requests=3, window_seconds=10)
    allowed3, _ = limiter.is_allowed(client_key, max_requests=3, window_seconds=10)
    allowed4, retry_after = limiter.is_allowed(client_key, max_requests=3, window_seconds=10)

    assert allowed1 is True
    assert allowed2 is True
    assert allowed3 is True
    assert allowed4 is False
    assert retry_after > 0


def test_public_chat_rejects_oversized_query(mock_app_state):
    long_query = "A" * 1001  # Exceeds max 1000 characters
    payload = {"text": long_query}
    response = mock_app_state.post("/api/v1/nlp/chat", json=payload)
    assert response.status_code == 400
    assert response.json()["signal"] == "message_too_long"


def test_public_chat_rejects_empty_query(mock_app_state):
    payload = {"text": "   "}
    response = mock_app_state.post("/api/v1/nlp/chat", json=payload)
    assert response.status_code == 400
    assert response.json()["signal"] == "invalid_query"


# --- P1-SEC-03: Stack Trace Elimination Tests ---

def test_public_chat_exception_sanitized_no_traceback_leaked(mock_app_state):
    with patch.object(NLPController, "chat", side_effect=RuntimeError("Simulated internal Qdrant failure")):
        payload = {"text": "How do I visit Karnak Temple?"}
        response = mock_app_state.post("/api/v1/nlp/chat", json=payload)

        assert response.status_code == 500
        data = response.json()

        # Critical: Assert no traceback or internal details leaked
        assert "traceback" not in data
        assert "RuntimeError" not in json.dumps(data)
        assert "File " not in json.dumps(data)
        assert data["signal"] == "chat_error"
        assert "error_id" in data
