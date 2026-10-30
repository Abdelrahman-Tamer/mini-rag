from collections import defaultdict
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import uuid
import ipaddress
from fastapi import Request
from fastapi.testclient import TestClient

from main import app
from controllers.NLPController import NLPController
from helpers.rate_limiter import (
    InMemorySlidingWindowLimiter,
    get_client_ip,
    check_rate_limit,
    parse_and_validate_ip,
    is_ip_trusted,
    LUA_SLIDING_WINDOW_SCRIPT,
)
from helpers.config import Settings
from models.db_schemes import DataChunk
from routes.schemes.nlp import ChatRequest
from stores.vectordb.providers.QdrantDBProvider import QdrantDBProvider


@pytest.fixture
def mock_app_state():
    app.state.initialized = True
    app.vectordb_client = MagicMock()
    app.generation_client = MagicMock()
    app.embedding_client = MagicMock()
    app.template_parser = MagicMock()
    app.db_client = MagicMock()
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def make_mock_request(client_ip="127.0.0.1", headers=None):
    request = MagicMock(spec=Request)
    request.client.host = client_ip
    request.headers = headers or {}
    return request


# ==============================================================================
# SECTION 2: QDRANT SAFE RE-INDEXING TESTS
# ==============================================================================

def test_safe_reindex_chunks_success():
    mock_vectordb = MagicMock()
    mock_vectordb.insert_many.return_value = True

    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3, 0.4]

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    chunks = [
        DataChunk(chunk_text="Intro to Luxor", chunk_metadata={"asset_id": "luxor_1"}, chunk_order=1),
        DataChunk(chunk_text="Karnak Temple details", chunk_metadata={"asset_id": "luxor_1"}, chunk_order=2),
        DataChunk(chunk_text="Valley of Kings", chunk_metadata={"asset_id": "luxor_1"}, chunk_order=3),
    ]

    res = nlp.safe_reindex_chunks(asset_id="luxor_1", chunks=chunks, old_chunk_count=3)

    assert res["status"] == "indexed"
    assert res["indexed"] == 3
    assert res["obsolete_pruned"] == 0
    assert mock_vectordb.insert_many.called


def test_safe_reindex_chunks_shrinks_and_prunes_obsolete():
    """Document shrinks from 5 chunks to 3 chunks. Chunks 4 and 5 must be pruned."""
    mock_vectordb = MagicMock()
    mock_vectordb.insert_many.return_value = True
    mock_vectordb.delete_by_ids.return_value = True

    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3, 0.4]

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    new_chunks = [
        DataChunk(chunk_text="Chunk 1", chunk_metadata={"asset_id": "asset_shrink"}, chunk_order=1),
        DataChunk(chunk_text="Chunk 2", chunk_metadata={"asset_id": "asset_shrink"}, chunk_order=2),
        DataChunk(chunk_text="Chunk 3", chunk_metadata={"asset_id": "asset_shrink"}, chunk_order=3),
    ]

    res = nlp.safe_reindex_chunks(asset_id="asset_shrink", chunks=new_chunks, old_chunk_count=5)

    assert res["status"] == "indexed"
    assert res["indexed"] == 3
    assert res["obsolete_pruned"] == 2

    # Verify that obsolete IDs for chunks 4 and 5 were deleted
    mock_vectordb.delete_by_ids.assert_called_once()
    deleted_ids = mock_vectordb.delete_by_ids.call_args[1]["record_ids"]
    assert len(deleted_ids) == 2
    expected_chunk_4_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "nileguide_asset_asset_shrink_chunk_4"))
    expected_chunk_5_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "nileguide_asset_asset_shrink_chunk_5"))
    assert expected_chunk_4_id in deleted_ids
    assert expected_chunk_5_id in deleted_ids


def test_safe_reindex_chunks_grows_zero_obsolete():
    """Document grows from 3 chunks to 5 chunks."""
    mock_vectordb = MagicMock()
    mock_vectordb.insert_many.return_value = True

    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3, 0.4]

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    new_chunks = [
        DataChunk(chunk_text=f"Chunk {i}", chunk_metadata={"asset_id": "asset_grow"}, chunk_order=i)
        for i in range(1, 6)
    ]

    res = nlp.safe_reindex_chunks(asset_id="asset_grow", chunks=new_chunks, old_chunk_count=3)

    assert res["status"] == "indexed"
    assert res["indexed"] == 5
    assert res["obsolete_pruned"] == 0
    assert not mock_vectordb.delete_by_ids.called


def test_safe_reindex_embedding_failure_preserves_old_vectors():
    """If embedding generation fails, Qdrant upsert and deletion must NEVER be called."""
    mock_vectordb = MagicMock()

    mock_embedding = MagicMock()
    # Fails on second chunk
    mock_embedding.embed_text.side_effect = [
        [0.1, 0.2, 0.3],
        RuntimeError("Embedding API rate limit or outage"),
    ]

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    chunks = [
        DataChunk(chunk_text="Chunk 1", chunk_metadata={"asset_id": "asset_fail"}, chunk_order=1),
        DataChunk(chunk_text="Chunk 2", chunk_metadata={"asset_id": "asset_fail"}, chunk_order=2),
    ]

    res = nlp.safe_reindex_chunks(asset_id="asset_fail", chunks=chunks, old_chunk_count=2)

    assert res["status"] == "embedding_failed"
    assert res["indexed"] == 0
    # Crucial safety assertion: Vector DB was never touched
    assert not mock_vectordb.insert_many.called
    assert not mock_vectordb.delete_by_ids.called


def test_safe_reindex_dimension_mismatch_aborts():
    mock_vectordb = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embed_text.side_effect = [
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2],  # Mismatched dimension
    ]

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    chunks = [
        DataChunk(chunk_text="Chunk 1", chunk_metadata={"asset_id": "asset_dim"}, chunk_order=1),
        DataChunk(chunk_text="Chunk 2", chunk_metadata={"asset_id": "asset_dim"}, chunk_order=2),
    ]

    res = nlp.safe_reindex_chunks(asset_id="asset_dim", chunks=chunks, old_chunk_count=2)
    assert res["status"] == "dimension_mismatch"
    assert not mock_vectordb.insert_many.called


def test_safe_reindex_qdrant_failure_aborts():
    mock_vectordb = MagicMock()
    mock_vectordb.insert_many.return_value = False  # Qdrant reports error

    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3]

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    chunks = [DataChunk(chunk_text="Chunk 1", chunk_metadata={"asset_id": "asset_qfail"}, chunk_order=1)]

    res = nlp.safe_reindex_chunks(asset_id="asset_qfail", chunks=chunks, old_chunk_count=5)
    assert res["status"] == "vectordb_failed"
    assert not mock_vectordb.delete_by_ids.called


def test_safe_reindex_empty_chunks_cleans_up():
    mock_vectordb = MagicMock()
    mock_vectordb.delete_by_filter.return_value = True

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )

    res = nlp.safe_reindex_chunks(asset_id="asset_empty", chunks=[], old_chunk_count=3)
    assert res["status"] == "indexed"
    assert res["indexed"] == 0
    assert mock_vectordb.delete_by_filter.called


def test_safe_reindex_idempotent_retry():
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

    chunks = [DataChunk(chunk_text="Chunk 1", chunk_metadata={"asset_id": "asset_retry"}, chunk_order=1)]

    res1 = nlp.safe_reindex_chunks(asset_id="asset_retry", chunks=chunks)
    res2 = nlp.safe_reindex_chunks(asset_id="asset_retry", chunks=chunks)

    assert res1["status"] == "indexed"
    assert res2["status"] == "indexed"
    # Ensure point IDs are identical across calls
    ids1 = mock_vectordb.insert_many.call_args_list[0][1]["record_ids"]
    ids2 = mock_vectordb.insert_many.call_args_list[1][1]["record_ids"]
    assert ids1 == ids2


# ==============================================================================
# SECTION 3: TRUSTED PROXY RESOLUTION SECURITY TESTS
# ==============================================================================

def test_trusted_proxy_direct_client_forged_header_ignored():
    """Direct untrusted client cannot forge X-Forwarded-For."""
    req = make_mock_request(client_ip="203.0.113.195", headers={"X-Forwarded-For": "1.1.1.1"})
    trusted_proxies = ["10.0.0.1", "127.0.0.1"]

    resolved = get_client_ip(req, trusted_proxies=trusted_proxies)
    assert resolved == "203.0.113.195"


def test_trusted_proxy_single_hop_resolves_client():
    """Direct peer is trusted proxy; real client IP is in X-Forwarded-For."""
    req = make_mock_request(client_ip="10.0.0.1", headers={"X-Forwarded-For": "198.51.100.22"})
    trusted_proxies = ["10.0.0.1", "127.0.0.1"]

    resolved = get_client_ip(req, trusted_proxies=trusted_proxies)
    assert resolved == "198.51.100.22"


def test_trusted_proxy_forged_leftmost_ignored_rightmost_untrusted_chosen():
    """Attacker injected 1.1.1.1; reverse proxy appended real IP 203.0.113.50."""
    req = make_mock_request(
        client_ip="10.0.0.1",
        headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.50"}
    )
    trusted_proxies = ["10.0.0.1", "127.0.0.1"]

    resolved = get_client_ip(req, trusted_proxies=trusted_proxies)
    # Right-to-left traversal must pick 203.0.113.50, completely ignoring 1.1.1.1!
    assert resolved == "203.0.113.50"


def test_trusted_proxy_multiple_hops_skips_internal_proxies():
    """Client -> Proxy A (10.0.0.2) -> Proxy B (10.0.0.1) -> App"""
    req = make_mock_request(
        client_ip="10.0.0.1",
        headers={"X-Forwarded-For": "203.0.113.77, 10.0.0.2"}
    )
    trusted_proxies = ["10.0.0.1", "10.0.0.2", "127.0.0.1"]

    resolved = get_client_ip(req, trusted_proxies=trusted_proxies)
    assert resolved == "203.0.113.77"


def test_trusted_proxy_invalid_ip_tokens_safely_skipped():
    """Malformed tokens like <script> or text are rejected."""
    req = make_mock_request(
        client_ip="10.0.0.1",
        headers={"X-Forwarded-For": "<script>alert(1)</script>, not_an_ip, 203.0.113.99"}
    )
    trusted_proxies = ["10.0.0.1"]

    resolved = get_client_ip(req, trusted_proxies=trusted_proxies)
    assert resolved == "203.0.113.99"


def test_trusted_proxy_ipv6_handling():
    req = make_mock_request(
        client_ip="10.0.0.1",
        headers={"X-Forwarded-For": "2001:db8:85a3::8a2e:370:7334"}
    )
    trusted_proxies = ["10.0.0.1"]

    resolved = get_client_ip(req, trusted_proxies=trusted_proxies)
    assert resolved == "2001:db8:85a3::8a2e:370:7334"


def test_trusted_proxy_cloudflare_cf_connecting_ip_when_peer_trusted():
    req = make_mock_request(
        client_ip="172.64.0.1",
        headers={"CF-Connecting-IP": "198.51.100.88", "X-Forwarded-For": "1.1.1.1"}
    )
    trusted_cf = ["172.64.0.1"]

    resolved = get_client_ip(req, trusted_proxies=["172.64.0.1"], trusted_cloudflare_proxies=trusted_cf)
    assert resolved == "198.51.100.88"


def test_trusted_proxy_cloudflare_cf_connecting_ip_ignored_when_peer_untrusted():
    """Attacker direct socket connection cannot spoof CF-Connecting-IP."""
    req = make_mock_request(
        client_ip="203.0.113.44",
        headers={"CF-Connecting-IP": "1.1.1.1"}
    )
    trusted_cf = ["172.64.0.1"]

    resolved = get_client_ip(req, trusted_proxies=["172.64.0.1"], trusted_cloudflare_proxies=trusted_cf)
    assert resolved == "203.0.113.44"


# ==============================================================================
# SECTION 4: REDIS DISTRIBUTED RATE LIMITING TESTS
# ==============================================================================

class FakeRedisStore:
    """In-memory simulation of Redis ZSET for testing Lua script sliding window."""
    def __init__(self):
        self.store = defaultdict(list)  # key -> list of (score, member)

    def eval(self, script, numkeys, key, now_str, window_str, max_str, *args):
        now = float(now_str)
        window = float(window_str)
        max_requests = int(max_str)
        cutoff = now - window

        # ZREMRANGEBYSCORE
        self.store[key] = [(s, m) for s, m in self.store[key] if s > cutoff]
        current_count = len(self.store[key])

        if current_count < max_requests:
            self.store[key].append((now, str(now)))
            return [1, 0, current_count + 1]
        else:
            oldest_ts = self.store[key][0][0]
            retry_after = max(1, int(oldest_ts + window - now))
            return [0, retry_after, current_count]


def test_distributed_redis_two_instances_share_state():
    """Verify two independent limiter callers evaluate against shared state."""
    import sys
    fake_redis = FakeRedisStore()
    fake_redis_mod = MagicMock()
    fake_redis_mod.from_url.return_value = fake_redis
    req = make_mock_request(client_ip="192.168.1.100")

    with patch.dict(sys.modules, {"redis": fake_redis_mod}):
        # Instance 1: make 2 requests (limit 3)
        ok1, _ = check_rate_limit(req, max_requests=3, window_seconds=60, redis_url="redis://fake:6379")
        ok2, _ = check_rate_limit(req, max_requests=3, window_seconds=60, redis_url="redis://fake:6379")
        assert ok1 is True
        assert ok2 is True

        # Instance 2 (representing a separate replica): makes 3rd and 4th request
        ok3, _ = check_rate_limit(req, max_requests=3, window_seconds=60, redis_url="redis://fake:6379")
        ok4, retry = check_rate_limit(req, max_requests=3, window_seconds=60, redis_url="redis://fake:6379")

        assert ok3 is True
        assert ok4 is False
        assert retry > 0


def test_distributed_redis_limiter_mandatory_in_production_raises_on_missing():
    req = make_mock_request(client_ip="192.168.1.100")
    with pytest.raises(RuntimeError) as exc_info:
        check_rate_limit(req, redis_url=None, require_redis=True)
    assert "mandatory in production" in str(exc_info.value)


# ==============================================================================
# SECTION 5: DTO SCHEMA & GREETING FAST PATH TESTS
# ==============================================================================

def test_chat_request_accepts_both_text_and_prompt():
    req1 = ChatRequest(text="Hello Egypt")
    assert req1.text == "Hello Egypt"

    req2 = ChatRequest(prompt="Tell me about Alexandria")
    assert req2.text == "Tell me about Alexandria"


def test_chat_endpoint_greeting_returns_fast_without_llm(mock_app_state):
    """Greeting like 'hello' returns immediately without calling LLM generation."""
    response = mock_app_state.post("/api/v1/nlp/chat", json={"text": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    # LLM was not called
    assert not app.generation_client.generate_text.called
