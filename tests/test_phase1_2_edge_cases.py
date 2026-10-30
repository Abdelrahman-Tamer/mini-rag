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
    # set_payload activated new generation
    mock_vectordb.set_payload.assert_called_once()
    # Old generation is deleted after promotion
    mock_vectordb.delete_by_filter.assert_any_call(
        nlp.collection_name,
        {"asset_id": "asset_retry", "generation_id": "gen_old_v1"}
    )


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

    # Test safe_reindex_chunks reports cleanup_failed when delete_by_filter raises
    mock_vectordb.delete_by_filter.side_effect = RuntimeError("Qdrant delete timeout")
    mock_vectordb.set_payload.return_value = True
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
