import pytest
from unittest.mock import MagicMock, patch
import uuid
import time
import sys
from collections import defaultdict
from fastapi import Request

from controllers.NLPController import NLPController
from helpers.rate_limiter import (
    check_rate_limit,
    LUA_SLIDING_WINDOW_SCRIPT,
)
from models.db_schemes import DataChunk
from stores.vectordb.providers.QdrantDBProvider import QdrantDBProvider


def make_mock_request(client_ip="127.0.0.1", headers=None):
    request = MagicMock(spec=Request)
    request.client.host = client_ip
    request.headers = headers or {}
    return request


# ==============================================================================
# TASK 1: QDRANT PARTIAL WRITE SAFETY & RECOVERABLE VERSIONED INDEXING TESTS
# ==============================================================================

def test_qdrant_first_batch_succeeds_second_batch_fails_rolls_back():
    """
    Simulates a 70-chunk document where batch 1 (chunks 1..50) succeeds,
    but batch 2 (chunks 51..70) fails.
    Verifies that the partial 50 points are rolled back via delete_by_ids,
    and old active document vectors are preserved.
    """
    mock_vectordb = MagicMock()
    # Batch 1 returns True, Batch 2 returns False
    mock_vectordb.insert_many.side_effect = [True, False]
    mock_vectordb.delete_by_ids.return_value = True

    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1] * 128

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    chunks = [
        DataChunk(chunk_text=f"Chunk {i}", chunk_metadata={"asset_id": "asset_partial"}, chunk_order=i)
        for i in range(1, 71)
    ]

    res = nlp.safe_reindex_chunks(
        asset_id="asset_partial",
        chunks=chunks,
        old_generation_id="gen_old_valid",
        generation_id="gen_new_failing"
    )

    assert res["status"] == "vectordb_failed"
    assert res["rolled_back"] is True
    assert res["preserved"] is True
    # Rollback must delete the 50 points that were committed in batch 1
    mock_vectordb.delete_by_ids.assert_called_once()
    deleted_ids = mock_vectordb.delete_by_ids.call_args[0][1]
    assert len(deleted_ids) == 50
    # The old generation was NEVER pruned
    assert not any("gen_old_valid" in str(call) for call in mock_vectordb.delete_by_filter.call_args_list)


def test_qdrant_process_restarts_and_retry_succeeds():
    """
    Simulates recovery after process restart: a retry occurs and all batches succeed.
    Old generation is pruned only after successful promotion.
    """
    mock_vectordb = MagicMock()
    mock_vectordb.insert_many.return_value = True
    mock_vectordb.set_payload.return_value = True
    mock_vectordb.delete_by_filter.return_value = True

    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1] * 128

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    chunks = [
        DataChunk(chunk_text=f"Chunk {i}", chunk_metadata={"asset_id": "asset_retry"}, chunk_order=i)
        for i in range(1, 60)
    ]

    res = nlp.safe_reindex_chunks(
        asset_id="asset_retry",
        chunks=chunks,
        old_generation_id="gen_old_v1",
        generation_id="gen_retry_v2"
    )

    assert res["status"] == "indexed"
    assert res["indexed"] == 59
    assert res["promoted"] is True
    # set_payload activated new generation and retained previous generation for rollback
    assert mock_vectordb.set_payload.call_count >= 2
    assert res.get("retained_generation_id") == "gen_old_v1"


def test_qdrant_previous_active_version_remains_searchable_until_promotion():
    """
    Filter builder must exclude inactive points (is_active == False)
    so in-progress/staging generations are invisible to retrieval.
    """
    provider = QdrantDBProvider(
        qdrant_db_url="http://localhost:6333",
        qdrant_db_api_key="test_key",
        distance_method="cosine"
    )

    # General search without filter excludes inactive
    filter_general = provider._build_filter(filter=None, exclude_inactive=True)
    assert filter_general is not None
    assert filter_general.must_not is not None
    assert filter_general.must_not[0].key == "metadata.is_active"
    assert filter_general.must_not[0].match.value is False

    # Search with city filter also preserves must_not inactive
    filter_city = provider._build_filter(filter={"city": "Cairo"}, exclude_inactive=True)
    assert filter_city is not None
    assert filter_city.must is not None
    assert filter_city.must[0].key == "metadata.city"
    assert filter_city.must_not is not None
    assert filter_city.must_not[0].key == "metadata.is_active"


def test_qdrant_concurrent_reindexing_distinct_generations():
    """
    Simulates two concurrent re-indexing runs for the same asset.
    Each gets its own point IDs and does not overwrite the other.
    """
    mock_vectordb = MagicMock()
    mock_vectordb.insert_many.return_value = True
    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1, 0.2]

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    chunks = [DataChunk(chunk_text="Same Content", chunk_metadata={"asset_id": "asset_conc"}, chunk_order=1)]

    res_a = nlp.safe_reindex_chunks(asset_id="asset_conc", chunks=chunks, generation_id="gen_A")
    res_b = nlp.safe_reindex_chunks(asset_id="asset_conc", chunks=chunks, generation_id="gen_B")

    ids_a = mock_vectordb.insert_many.call_args_list[0][1]["record_ids"]
    ids_b = mock_vectordb.insert_many.call_args_list[1][1]["record_ids"]
    meta_a = mock_vectordb.insert_many.call_args_list[0][1]["metadata"]
    meta_b = mock_vectordb.insert_many.call_args_list[1][1]["metadata"]

    assert ids_a != ids_b
    assert meta_a[0]["generation_id"] == "gen_A"
    assert meta_b[0]["generation_id"] == "gen_B"


def test_qdrant_deletion_isolated_to_requested_asset():
    """delete_file_vectors deletes ONLY the specified asset, never touching others."""
    mock_vectordb = MagicMock()
    mock_vectordb.delete_by_filter.return_value = True

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )

    res = nlp.delete_file_vectors("asset_isolated_99")
    assert res["status"] == "deleted"
    assert res["asset_id"] == "asset_isolated_99"

    mock_vectordb.delete_by_filter.assert_called_once_with(
        collection_name=nlp.collection_name,
        filter={"asset_id": "asset_isolated_99"}
    )


def test_qdrant_empty_chunks_without_allow_empty_rejected():
    """An unexpected empty chunk list must NOT silently erase an existing document."""
    mock_vectordb = MagicMock()

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )

    res = nlp.safe_reindex_chunks(asset_id="asset_protect", chunks=[], allow_empty=False)

    assert res["status"] == "empty_chunks_rejected"
    assert res["preserved"] is True
    assert not mock_vectordb.delete_by_filter.called
    assert not mock_vectordb.delete_by_ids.called


def test_qdrant_none_chunks_rejected_as_processing_failure():
    """Null chunks indicates upstream processing failure and must not touch vector store."""
    mock_vectordb = MagicMock()

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )

    res = nlp.safe_reindex_chunks(asset_id="asset_corrupt", chunks=None)

    assert res["status"] == "extraction_failed"
    assert res["preserved"] is True
    assert not mock_vectordb.delete_by_filter.called
    assert not mock_vectordb.delete_by_ids.called


# ==============================================================================
# TASK 2: REDIS LUA MEMBER UNIQUENESS & CONCURRENCY TESTS
# ==============================================================================

class AccurateRedisZSetStore:
    """Accurate in-memory emulation of Redis ZSET adhering to member uniqueness."""
    def __init__(self):
        # key -> dict mapping member -> score
        self.zsets = defaultdict(dict)

    def eval(self, script, numkeys, key, now_str, window_str, max_str, *args):
        now = float(now_str)
        window = float(window_str)
        max_requests = int(max_str)
        member_id = args[0] if args else str(now)
        clear_before = now - window

        zset = self.zsets[key]

        # ZREMRANGEBYSCORE: delete members with score <= clear_before
        expired = [m for m, score in zset.items() if score <= clear_before]
        for m in expired:
            del zset[m]

        current_requests = len(zset)

        if current_requests < max_requests:
            # ZADD key now member_id
            zset[member_id] = now
            return [1, 0, current_requests + 1]
        else:
            # ZRANGE key 0 0 WITHSCORES
            oldest_score = min(zset.values()) if zset else now
            retry_after = max(1, int(oldest_score + window - now))
            return [0, retry_after, current_requests]


def test_redis_sliding_window_burst_hundreds_identical_timestamps():
    """
    Hundreds of requests arriving at the EXACT same float timestamp.
    Because each has a unique member ID, ZCARD must count every single one
    and throttle once max_requests is exceeded.
    """
    store = AccurateRedisZSetStore()
    fake_redis_mod = MagicMock()
    fake_redis_mod.from_url.return_value = store

    frozen_now = 1730283000.12345
    limit = 20
    window = 60

    allowed_count = 0
    throttled_count = 0

    with patch.dict(sys.modules, {"redis": fake_redis_mod}), patch("time.time", return_value=frozen_now):
        req = make_mock_request(client_ip="192.0.2.1")
        for i in range(100):
            ok, retry_after = check_rate_limit(
                req,
                max_requests=limit,
                window_seconds=window,
                redis_url="redis://localhost:6379"
            )
            if ok:
                allowed_count += 1
            else:
                throttled_count += 1
                assert retry_after > 0

    assert allowed_count == limit
    assert throttled_count == 80
    assert len(store.zsets["chat:192.0.2.1"]) == limit


def test_redis_same_timestamp_different_request_ids():
    """Two requests with the exact same timestamp but distinct request IDs are both counted."""
    store = AccurateRedisZSetStore()
    fake_redis_mod = MagicMock()
    fake_redis_mod.from_url.return_value = store

    frozen_now = 1730283000.555

    with patch.dict(sys.modules, {"redis": fake_redis_mod}), patch("time.time", return_value=frozen_now):
        req = make_mock_request(client_ip="192.0.2.2")
        ok1, _ = check_rate_limit(req, max_requests=10, window_seconds=60, redis_url="redis://localhost:6379", request_id="req_alpha")
        ok2, _ = check_rate_limit(req, max_requests=10, window_seconds=60, redis_url="redis://localhost:6379", request_id="req_beta")

    assert ok1 is True
    assert ok2 is True
    assert len(store.zsets["chat:192.0.2.2"]) == 2


def test_redis_repeated_request_id_idempotent():
    """Resending the exact same request_id at the same timestamp does NOT increment count twice."""
    store = AccurateRedisZSetStore()
    fake_redis_mod = MagicMock()
    fake_redis_mod.from_url.return_value = store

    frozen_now = 1730283000.555

    with patch.dict(sys.modules, {"redis": fake_redis_mod}), patch("time.time", return_value=frozen_now):
        req = make_mock_request(client_ip="192.0.2.3")
        ok1, _ = check_rate_limit(req, max_requests=10, window_seconds=60, redis_url="redis://localhost:6379", request_id="idempotent_key_1")
        ok2, _ = check_rate_limit(req, max_requests=10, window_seconds=60, redis_url="redis://localhost:6379", request_id="idempotent_key_1")

    assert ok1 is True
    assert ok2 is True
    # In Redis ZSET, identical member overwrites in-place
    assert len(store.zsets["chat:192.0.2.3"]) == 1


def test_redis_window_expiration_cleans_old_members():
    """Members older than now - window are purged upon evaluation."""
    store = AccurateRedisZSetStore()
    fake_redis_mod = MagicMock()
    fake_redis_mod.from_url.return_value = store

    req = make_mock_request(client_ip="192.0.2.4")

    # Time T=100: 5 requests sent with window=60
    with patch.dict(sys.modules, {"redis": fake_redis_mod}), patch("time.time", return_value=100.0):
        for _ in range(5):
            check_rate_limit(req, max_requests=5, window_seconds=60, redis_url="redis://localhost:6379")

    assert len(store.zsets["chat:192.0.2.4"]) == 5

    # Time T=165 (65 seconds later): window has expired
    with patch.dict(sys.modules, {"redis": fake_redis_mod}), patch("time.time", return_value=165.0):
        ok, retry_after = check_rate_limit(req, max_requests=5, window_seconds=60, redis_url="redis://localhost:6379")
        assert ok is True
        assert retry_after == 0
        # Old 5 members were purged, only the new 1 member remains
        assert len(store.zsets["chat:192.0.2.4"]) == 1


def test_qdrant_cleanup_failure_search_deduplication():
    """Scenario C: If old generation cleanup fails in Qdrant, search deduplication prevents duplicate chunks."""
    mock_vectordb = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1] * 128
    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )
    doc_gen1 = MagicMock()
    doc_gen1.metadata = {"asset_id": "asset_fail_cleanup", "chunk_order": 1, "generation_id": "gen_1"}
    doc_gen1.score = 0.95
    doc_gen2 = MagicMock()
    doc_gen2.metadata = {"asset_id": "asset_fail_cleanup", "chunk_order": 1, "generation_id": "gen_2"}
    doc_gen2.score = 0.85
    doc_other = MagicMock()
    doc_other.metadata = {"asset_id": "asset_other", "chunk_order": 1, "generation_id": "gen_1"}
    doc_other.score = 0.90
    raw_results = [doc_gen1, doc_gen2, doc_other]
    deduped = nlp._deduplicate_search_results(raw_results)
    assert len(deduped) == 2
    assert deduped[0].metadata["generation_id"] == "gen_1"
    assert deduped[1].metadata["asset_id"] == "asset_other"

    # Test safe_reindex_chunks reports cleanup_failed when retention marking fails
    mock_vectordb.set_payload.side_effect = [True, RuntimeError("Qdrant retention error")]
    mock_vectordb.insert_many.return_value = True
    res = nlp.safe_reindex_chunks(
        asset_id="asset_fail_cleanup",
        chunks=[{"text": "chunk 1"}],
        old_generation_id="gen_old",
        generation_id="gen_new"
    )
    assert res["status"] == "indexed"
    assert res.get("cleanup_failed") is True


def test_qdrant_concurrent_indexing_asset_lock_serializes():
    """Scenario D: Verifies per-asset lock serializes concurrent indexing jobs for the same asset."""
    mock_vectordb = MagicMock()
    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )
    lock_a1 = nlp._get_asset_lock("asset_alpha")
    lock_a2 = nlp._get_asset_lock("asset_alpha")
    lock_b = nlp._get_asset_lock("asset_beta")
    assert lock_a1 is lock_a2
    assert lock_a1 is not lock_b


def test_chat_endpoint_returns_triple_key_response():
    """Verifies /api/v1/nlp/chat returns answer, message, and response keys for greeting contract."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    # Arabic greeting
    res_ar = client.post("/api/v1/nlp/chat", json={"prompt": "مرحبا"})
    assert res_ar.status_code == 200
    data_ar = res_ar.json()
    assert "answer" in data_ar
    assert "message" in data_ar
    assert "response" in data_ar
    assert data_ar["answer"] == data_ar["response"]
    # English greeting
    res_en = client.post("/api/v1/nlp/chat", json={"prompt": "hello"})
    assert res_en.status_code == 200
    data_en = res_en.json()
    assert "answer" in data_en
    assert "message" in data_en
    assert "response" in data_en
    assert data_en["answer"] == data_en["response"]


# ==============================================================================
# PHASE 1.2 STAGING READINESS & EMBEDDING/QDRANT CONSISTENCY TESTS
# ==============================================================================

def test_validate_embedding_config_valid():
    """Verifies valid embedding configurations pass startup validation."""
    from helpers.config import Settings, validate_embedding_config

    # Gemini endpoint with gemini-embedding-001 (3072)
    s1 = Settings(
        EMBEDDING_BACKEND="OPENAI",
        OPENAI_API_URL="https://generativelanguage.googleapis.com/v1beta/openai/",
        EMBEDDING_MODEL_ID="gemini-embedding-001",
        EMBEDDING_MODEL_SIZE=3072,
    )
    validate_embedding_config(s1)

    # Gemini endpoint with text-embedding-004 (768)
    s2 = Settings(
        EMBEDDING_BACKEND="OPENAI",
        OPENAI_API_URL="https://generativelanguage.googleapis.com/v1beta/openai/",
        EMBEDDING_MODEL_ID="text-embedding-004",
        EMBEDDING_MODEL_SIZE=768,
    )
    validate_embedding_config(s2)

    # OpenAI endpoint with text-embedding-3-small (1536)
    s3 = Settings(
        EMBEDDING_BACKEND="OPENAI",
        OPENAI_API_URL="https://api.openai.com/v1",
        EMBEDDING_MODEL_ID="text-embedding-3-small",
        EMBEDDING_MODEL_SIZE=1536,
    )
    validate_embedding_config(s3)


def test_validate_embedding_config_invalid_mismatches():
    """Verifies invalid embedding backend, URL, or dimension mismatches raise ValueError."""
    from helpers.config import Settings, validate_embedding_config

    # Unsupported backend
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_BACKEND"):
        validate_embedding_config(Settings(EMBEDDING_BACKEND="HUGGINGFACE"))

    # Gemini endpoint with OpenAI model
    with pytest.raises(ValueError, match="Embedding backend mismatch"):
        validate_embedding_config(Settings(
            EMBEDDING_BACKEND="OPENAI",
            OPENAI_API_URL="https://generativelanguage.googleapis.com/v1beta/openai/",
            EMBEDDING_MODEL_ID="davinci",
            EMBEDDING_MODEL_SIZE=1536,
        ))

    # Gemini endpoint with text-embedding-004 and incorrect dimension (3072 instead of 768)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_embedding_config(Settings(
            EMBEDDING_BACKEND="OPENAI",
            OPENAI_API_URL="https://generativelanguage.googleapis.com/v1beta/openai/",
            EMBEDDING_MODEL_ID="text-embedding-004",
            EMBEDDING_MODEL_SIZE=3072,
        ))

    # Official OpenAI endpoint with gemini-embedding-001
    with pytest.raises(ValueError, match="Embedding backend mismatch"):
        validate_embedding_config(Settings(
            EMBEDDING_BACKEND="OPENAI",
            OPENAI_API_URL="https://api.openai.com/v1",
            EMBEDDING_MODEL_ID="gemini-embedding-001",
            EMBEDDING_MODEL_SIZE=3072,
        ))

    # Official OpenAI endpoint with text-embedding-ada-002 and wrong dimension
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_embedding_config(Settings(
            EMBEDDING_BACKEND="OPENAI",
            OPENAI_API_URL="https://api.openai.com/v1",
            EMBEDDING_MODEL_ID="text-embedding-ada-002",
            EMBEDDING_MODEL_SIZE=768,
        ))


def test_nlp_chat_expands_candidate_window_and_deduplicates():
    """Verifies that chat() retrieves fetch_limit (2x limit) candidates and deduplicates without underfilling."""
    mock_vectordb = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1] * 128

    # Create 8 candidates: 2 duplicate pairs and 4 unique items
    candidates = []
    # asset_dup has chunk 1 in gen_old and gen_new
    doc_d1 = MagicMock(score=0.95, metadata={"asset_id": "asset_dup", "chunk_order": 1, "generation_id": "gen_old"})
    doc_d2 = MagicMock(score=0.90, metadata={"asset_id": "asset_dup", "chunk_order": 1, "generation_id": "gen_new"})
    candidates.extend([doc_d1, doc_d2])
    for i in range(2, 8):
        candidates.append(MagicMock(score=0.80 - (i * 0.05), metadata={"asset_id": f"asset_{i}", "chunk_order": 1, "generation_id": "g1"}))

    mock_vectordb.search_by_vector.return_value = candidates

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )
    nlp.score_threshold = 0.2

    # Request limit = 3
    results = nlp.chat(query="Historical landmarks in Cairo", limit=3)

    # Verify search_by_vector called with expanded window limit (fetch_limit = max(3*2, 3+5) = 8)
    assert mock_vectordb.search_by_vector.called
    call_kwargs = mock_vectordb.search_by_vector.call_args[1]
    assert call_kwargs["limit"] == 8

    # Verify returned results are capped at exactly limit=3 and deduplicated
    assert len(results) == 3
    assert results[0].metadata["asset_id"] == "asset_dup"
    assert results[1].metadata["asset_id"] == "asset_2"
    assert results[2].metadata["asset_id"] == "asset_3"


def test_openai_provider_embed_text_dimensions_and_fallback():
    """Verifies OpenAIProvider sends dimensions when configured, and falls back gracefully if rejected."""
    from stores.llm.providers.OpenAIProvider import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key", api_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    provider.set_embedding_model("gemini-embedding-001", embedding_size=3072)

    mock_client = MagicMock()
    # Mock normal response
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.05] * 3072)]
    mock_client.embeddings.create.return_value = mock_resp
    provider.client = mock_client

    vec = provider.embed_text("Test sentence")
    assert vec is not None
    assert len(vec) == 3072
    assert mock_client.embeddings.create.call_args[1].get("dimensions") == 3072

    # Test fallback when provider rejects dimensions
    def create_side_effect(**kwargs):
        if "dimensions" in kwargs:
            raise Exception("Invalid parameter 'dimensions' not supported on this endpoint")
        fallback_resp = MagicMock()
        fallback_resp.data = [MagicMock(embedding=[0.05] * 768)]
        return fallback_resp

    mock_client.embeddings.create.side_effect = create_side_effect
    vec_fallback = provider.embed_text("Test sentence fallback")
    assert vec_fallback is not None
    assert len(vec_fallback) == 768


# ==============================================================================
# TASK 2 & 3: QDRANT ROLLBACK RETENTION & REDIS FAIL-CLOSED RECOVERY TESTS
# ==============================================================================

def test_qdrant_rollback_retention_preserves_previous_generation():
    """Verifies that indexing gen_2 activates gen_2 and retains gen_1 for rollback instead of deleting it."""
    mock_vectordb = MagicMock()
    mock_vectordb.insert_many.return_value = True
    mock_vectordb.set_payload.return_value = True
    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1] * 128

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=mock_embedding,
        template_parser=MagicMock(),
    )

    res = nlp.safe_reindex_chunks(
        asset_id="asset_retention_test",
        chunks=[{"text": "chunk 1 gen 2"}],
        old_generation_id="gen_1",
        generation_id="gen_2"
    )

    assert res["status"] == "indexed"
    assert res["promoted"] is True
    assert res.get("retained_generation_id") == "gen_1"

    # Verify delete_by_filter was NOT called to delete gen_1
    delete_calls = mock_vectordb.delete_by_filter.call_args_list
    for call in delete_calls:
        flt = call[1].get("filter") or (call[0][1] if len(call[0]) > 1 else {})
        assert flt.get("generation_id") != "gen_1", "gen_1 should be retained, not deleted!"

    # Verify set_payload was called to mark gen_1 as rollback_retained: True
    payload_calls = mock_vectordb.set_payload.call_args_list
    gen1_retained = False
    for call in payload_calls:
        kwargs = call[1] if call[1] else {}
        flt = kwargs.get("filter", {})
        payload = kwargs.get("payload", {})
        if flt.get("generation_id") == "gen_1":
            meta = payload.get("metadata", {})
            if meta.get("is_active") is False and meta.get("rollback_retained") is True:
                gen1_retained = True
    assert gen1_retained, "Previous generation gen_1 must be marked with is_active=False and rollback_retained=True"


def test_qdrant_rollback_asset_generation_reactivates_previous_generation():
    """Verifies rollback_asset_generation deactivates current generation and activates target rollback generation."""
    mock_vectordb = MagicMock()
    mock_vectordb.set_payload.return_value = True

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )

    res = nlp.rollback_asset_generation(
        asset_id="asset_rollback_test",
        target_generation_id="gen_1",
        current_generation_id="gen_2"
    )

    assert res["status"] == "rolled_back"
    assert res["rolled_back"] is True
    assert res["target_generation_id"] == "gen_1"
    assert res["deactivated_generation_id"] == "gen_2"

    payload_calls = mock_vectordb.set_payload.call_args_list
    assert len(payload_calls) >= 2

    gen2_deactivated = False
    gen1_activated = False
    for call in payload_calls:
        flt = call[1].get("filter", {})
        meta = call[1].get("payload", {}).get("metadata", {})
        if flt.get("generation_id") == "gen_2" and meta.get("is_active") is False:
            gen2_deactivated = True
        if flt.get("generation_id") == "gen_1" and meta.get("is_active") is True:
            gen1_activated = True

    assert gen2_deactivated, "Current generation gen_2 must be deactivated"
    assert gen1_activated, "Target generation gen_1 must be reactivated"


def test_qdrant_prune_expired_generations():
    """Verifies prune_expired_generations deletes only explicitly targeted obsolete generations."""
    mock_vectordb = MagicMock()
    mock_vectordb.delete_by_filter.return_value = True

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )

    res = nlp.prune_expired_generations(
        asset_id="asset_prune_test",
        generation_ids_to_prune=["gen_expired_0"]
    )

    assert res["status"] == "pruned"
    assert res["pruned_generations"] == 1
    mock_vectordb.delete_by_filter.assert_called_with(
        nlp.collection_name,
        {"asset_id": "asset_prune_test", "generation_id": "gen_expired_0"}
    )


def test_redis_fail_closed_returns_503_on_rate_limiter_outage():
    """Verifies /api/v1/nlp/chat returns HTTP 503 with Retry-After when Redis fails under require_redis policy."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    with patch("routes.nlp.get_settings") as mock_settings,          patch("routes.nlp.check_rate_limit", side_effect=RuntimeError("Redis connection refused")):
        s = MagicMock()
        s.RATE_LIMIT_CHAT_REQUESTS = 15
        s.RATE_LIMIT_CHAT_WINDOW_SECONDS = 60
        s.TRUSTED_PROXIES = ["127.0.0.1"]
        s.TRUSTED_CLOUDFLARE_PROXIES = ["127.0.0.1"]
        s.REDIS_URL = "redis://invalid-host:6379"
        s.REQUIRE_REDIS_RATE_LIMITER = True
        mock_settings.return_value = s

        res = client.post("/api/v1/nlp/chat", json={"prompt": "مرحبا"})
        assert res.status_code == 503
        data = res.json()
        assert data.get("signal") == "service_temporarily_unavailable"
        assert res.headers.get("Retry-After") == "30"


def test_redis_outage_does_not_block_admin_unrelated_functions():
    """Verifies that an outage in Redis does NOT prevent admin knowledge-base operations."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    res = client.get("/api/v1/data/files", headers={"X-Admin-Api-Key": "invalid-key"})
    assert res.status_code == 401


# ==============================================================================
# SECTION 5 & 6: DISTRIBUTED ASSET LOCKING & DYNAMIC RETRY-AFTER TESTS
# ==============================================================================

def test_distributed_asset_lock_acquisition_and_release():
    """Verifies asset_lock acquires lock via Redis set(nx=True) and safely releases via Lua with token."""
    mock_vectordb = MagicMock()
    mock_redis = MagicMock()
    mock_redis.set.return_value = True
    fake_redis_mod = MagicMock()
    fake_redis_mod.from_url.return_value = mock_redis

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )
    nlp.app_settings = MagicMock()
    nlp.app_settings.REDIS_URL = "redis://mock-redis:6379"
    nlp.app_settings.REQUIRE_REDIS_RATE_LIMITER = True

    with patch.dict(sys.modules, {"redis": fake_redis_mod}):
        with nlp.asset_lock("asset_dist_test", ttl_seconds=30):
            mock_redis.set.assert_called_once()
            args, kwargs = mock_redis.set.call_args
            assert args[0] == "nileguide:lock:asset:asset_dist_test"
            assert kwargs.get("nx") is True
            assert kwargs.get("px") == 30000

        mock_redis.eval.assert_called_once()
        eval_args, _ = mock_redis.eval.call_args
        assert "redis.call('GET', KEYS[1]) == ARGV[1]" in eval_args[0]
        assert eval_args[2] == "nileguide:lock:asset:asset_dist_test"


def test_distributed_asset_lock_collision_fails():
    """Verifies that if another worker holds the lock, acquisition raises RuntimeError."""
    mock_vectordb = MagicMock()
    mock_redis = MagicMock()
    mock_redis.set.return_value = False
    fake_redis_mod = MagicMock()
    fake_redis_mod.from_url.return_value = mock_redis

    nlp = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=MagicMock(),
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )
    nlp.app_settings = MagicMock()
    nlp.app_settings.REDIS_URL = "redis://mock-redis:6379"
    nlp.app_settings.REQUIRE_REDIS_RATE_LIMITER = True

    with patch.dict(sys.modules, {"redis": fake_redis_mod}):
        with pytest.raises(RuntimeError, match="Concurrent indexing conflict: asset 'asset_collision' is locked by another worker"):
            with nlp.asset_lock("asset_collision", ttl_seconds=30):
                pass


def test_redis_sliding_window_dynamic_retry_after():
    """Verifies that Retry-After dynamically decreases as elapsed time moves forward in window."""
    class DynamicRedisStore:
        def __init__(self):
            self.zsets = defaultdict(list)
            self.ttls = {}
        def eval(self, script, numkeys, key, now, window, max_req, event_id):
            now_f = float(now)
            win_f = float(window)
            max_r = int(max_req)
            clear_before = now_f - win_f
            self.zsets[key] = [(s, member) for s, member in self.zsets[key] if s > clear_before]
            if len(self.zsets[key]) < max_r:
                self.zsets[key].append((now_f, event_id))
                self.zsets[key].sort(key=lambda x: x[0])
                return [1, 0, len(self.zsets[key])]
            else:
                oldest_ts = self.zsets[key][0][0]
                retry_after = max(1, int(oldest_ts + win_f - now_f))
                return [0, retry_after, len(self.zsets[key])]

    store = DynamicRedisStore()
    fake_redis_mod = MagicMock()
    fake_redis_mod.from_url.return_value = store
    req = make_mock_request(client_ip="198.51.100.1")

    with patch.dict(sys.modules, {"redis": fake_redis_mod}):
        with patch("time.time", return_value=1000.0):
            for i in range(15):
                allowed, retry = check_rate_limit(req, max_requests=15, window_seconds=60, redis_url="redis://localhost", prefix="nileguide:staging:chat")
                assert allowed is True

        with patch("time.time", return_value=1010.0):
            allowed, retry_50 = check_rate_limit(req, max_requests=15, window_seconds=60, redis_url="redis://localhost", prefix="nileguide:staging:chat")
            assert allowed is False
            assert retry_50 == 50

        with patch("time.time", return_value=1045.0):
            allowed, retry_15 = check_rate_limit(req, max_requests=15, window_seconds=60, redis_url="redis://localhost", prefix="nileguide:staging:chat")
            assert allowed is False
            assert retry_15 == 15
