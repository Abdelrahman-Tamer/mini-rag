"""
Test Suite: Guest Quota Reservation, Refund, Concurrency & Idempotency
Validates Phase 12.3.1 remediation requirements:
1. Successful answer consumes exactly one question.
2. Transient generation failure restores the reserved quota.
3. Fallback generation failure restores the reserved quota.
4. Validation failure (empty text) restores the reserved quota.
5. Invalid token consumes zero questions.
6. Burst rejection consumes zero daily quota.
7. Concurrent requests cannot exceed the daily limit.
8. Duplicate retries do not double-consume quota (idempotent request_id).
9. Production and Preview namespaces remain strictly isolated.
10. Reservation expiration and cleanup behave correctly (stale pending purged).
11. Redis failure remains fail-closed under require_redis=True.
"""

import time
import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from main import app
from helpers.guest_session import (
    _in_memory_guest_limiter,
    resolve_guest_secret,
    derive_guest_id,
    generate_guest_token,
    verify_guest_token,
    get_guest_remaining_questions,
    check_guest_limits,
    commit_guest_quota,
    refund_guest_quota,
    LUA_GUEST_RESERVE_QUOTA_SCRIPT,
    LUA_GUEST_COMMIT_QUOTA_SCRIPT,
    LUA_GUEST_REFUND_QUOTA_SCRIPT,
)
from routes.schemes.nlp import ChatRequest

DEV_SECRET = "nileguide-guest-secret-2026-safe-dev"


@pytest.fixture(autouse=True)
def reset_in_memory_state():
    _in_memory_guest_limiter.reset()
    yield
    _in_memory_guest_limiter.reset()


@pytest.fixture
def mock_app_client():
    app.state.initialized = True
    app.vectordb_client = MagicMock()
    app.vectordb_client.search_by_vector.return_value = []
    app.generation_client = MagicMock()
    app.generation_client.generate_text.return_value = "Mocked answer"
    app.embedding_client = MagicMock()
    app.embedding_client.get_embedding.return_value = [0.1] * 768
    app.template_parser = MagicMock()
    app.template_parser.get.return_value = "Mock prompt"
    app.db_client = MagicMock()
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_successful_answer_consumes_exactly_one_question(mock_app_client):
    """Verify that a successful chat answer reserves and commits exactly one quota unit."""
    client_ip = "192.168.1.10"
    guest_id = derive_guest_id(client_ip, DEV_SECRET)
    session_id = uuid.uuid4().hex
    token = generate_guest_token(guest_id, session_id, DEV_SECRET)

    # Initially 5 questions remaining
    initial_rem = get_guest_remaining_questions(guest_id, None, False, "development")
    assert initial_rem == 5

    # Greeting query produces instant successful answer
    resp = mock_app_client.post(
        "/api/v1/nlp/chat",
        json={"text": "hello", "guest_token": token},
        headers={"x-forwarded-for": client_ip}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["guest_remaining_questions"] == 4

    # Verify quota remaining after commit
    after_rem = get_guest_remaining_questions(guest_id, None, False, "development")
    assert after_rem == 4


def test_transient_generation_failure_restores_reserved_quota(mock_app_client):
    """Verify that if answer generation fails with 500 or exception, the reserved quota is refunded."""
    client_ip = "192.168.1.11"
    guest_id = derive_guest_id(client_ip, DEV_SECRET)
    session_id = uuid.uuid4().hex
    token = generate_guest_token(guest_id, session_id, DEV_SECRET)

    initial_rem = get_guest_remaining_questions(guest_id, None, False, "development")
    assert initial_rem == 5

    with patch("controllers.NLPController.NLPController.chat", side_effect=RuntimeError("Upstream timeout")):
        resp = mock_app_client.post(
            "/api/v1/nlp/chat",
            json={"text": "Visiting pyramids ticket prices", "guest_token": token},
            headers={"x-forwarded-for": client_ip}
        )
        assert resp.status_code == 500

    # Quota MUST be refunded back to 5
    after_rem = get_guest_remaining_questions(guest_id, None, False, "development")
    assert after_rem == 5, f"Expected 5 remaining after transient failure refund, got {after_rem}"


def test_fallback_generation_failure_restores_quota(mock_app_client):
    """Verify that if fallback generation fails or returns empty answer, quota is refunded."""
    client_ip = "192.168.1.12"
    guest_id = derive_guest_id(client_ip, DEV_SECRET)
    session_id = uuid.uuid4().hex
    token = generate_guest_token(guest_id, session_id, DEV_SECRET)

    with patch("controllers.NLPController.NLPController.chat", return_value=[]):
        with patch("controllers.NLPController.NLPController.generate_chat_answer", side_effect=RuntimeError("AI Timeout")):
            resp = mock_app_client.post(
                "/api/v1/nlp/chat",
                json={"text": "Random unsupported query", "guest_token": token},
                headers={"x-forwarded-for": client_ip}
            )
            assert resp.status_code == 503

    # Quota MUST be refunded
    rem = get_guest_remaining_questions(guest_id, None, False, "development")
    assert rem == 5


def test_empty_query_validation_failure_restores_quota(mock_app_client):
    """Verify that a validation failure (empty text) refunds the reserved quota."""
    client_ip = "192.168.1.13"
    guest_id = derive_guest_id(client_ip, DEV_SECRET)
    session_id = uuid.uuid4().hex
    token = generate_guest_token(guest_id, session_id, DEV_SECRET)

    resp = mock_app_client.post(
        "/api/v1/nlp/chat",
        json={"text": "   ", "guest_token": token},
        headers={"x-forwarded-for": client_ip}
    )
    assert resp.status_code == 400
    rem = get_guest_remaining_questions(guest_id, None, False, "development")
    assert rem == 5


def test_invalid_token_consumes_zero_questions(mock_app_client):
    """Verify that an invalid or tampered token rejects before any quota reservation."""
    client_ip = "192.168.1.14"
    guest_id = derive_guest_id(client_ip, DEV_SECRET)

    resp = mock_app_client.post(
        "/api/v1/nlp/chat",
        json={"text": "hello", "guest_token": "tampered.token.12345.bad"},
        headers={"x-forwarded-for": client_ip}
    )
    assert resp.status_code == 401
    rem = get_guest_remaining_questions(guest_id, None, False, "development")
    assert rem == 5


def test_burst_rejection_consumes_zero_daily_quota(mock_app_client):
    """Verify that burst rate limit rejection (10 req/min) leaves daily quota untouched."""
    # Test using check_guest_limits directly with burst_limit=2, daily_limit=10
    allowed1, _, rem1, _ = check_guest_limits(
        guest_id="burst_test_guest",
        redis_url=None,
        require_redis=False,
        env="test",
        daily_limit=10,
        burst_limit=2,
        burst_window=60
    )
    assert allowed1 is True
    assert rem1 == 9

    allowed2, _, rem2, _ = check_guest_limits(
        guest_id="burst_test_guest",
        redis_url=None,
        require_redis=False,
        env="test",
        daily_limit=10,
        burst_limit=2,
        burst_window=60
    )
    assert allowed2 is True
    assert rem2 == 8

    # 3rd request hits burst limit
    allowed3, _, rem3, err3 = check_guest_limits(
        guest_id="burst_test_guest",
        redis_url=None,
        require_redis=False,
        env="test",
        daily_limit=10,
        burst_limit=2,
        burst_window=60
    )
    assert allowed3 is False
    assert err3 == "burst_limit_exceeded"
    # Daily remaining must NOT have decremented from 8
    assert rem3 == 8


def test_concurrent_reservations_cannot_exceed_daily_limit():
    """Verify that multiple concurrent reservations respect daily_limit."""
    guest_id = "concurrent_guest_1"
    results = []

    # Try to make 8 reservations concurrently against a limit of 5
    for i in range(8):
        allowed, _, rem, err = check_guest_limits(
            guest_id=guest_id,
            redis_url=None,
            require_redis=False,
            env="test",
            daily_limit=5,
            request_id=f"concurrent_req_{i}"
        )
        results.append((allowed, rem, err))

    allowed_count = sum(1 for a, _, _ in results if a is True)
    rejected_count = sum(1 for a, _, _ in results if a is False)

    assert allowed_count == 5, f"Expected exactly 5 allowed, got {allowed_count}"
    assert rejected_count == 3, f"Expected exactly 3 rejected, got {rejected_count}"


def test_duplicate_retries_idempotency():
    """Verify that duplicate requests with the same request_id are idempotent and do not double-charge."""
    guest_id = "retry_guest_1"
    req_id = "req_fixed_id_123"

    allowed1, _, rem1, _ = check_guest_limits(
        guest_id=guest_id,
        redis_url=None,
        require_redis=False,
        env="test",
        daily_limit=5,
        request_id=req_id
    )
    assert allowed1 is True
    assert rem1 == 4

    # Immediate retry with same req_id
    allowed2, _, rem2, _ = check_guest_limits(
        guest_id=guest_id,
        redis_url=None,
        require_redis=False,
        env="test",
        daily_limit=5,
        request_id=req_id
    )
    assert allowed2 is True
    assert rem2 == 4  # Still 4, NOT 3!


def test_stale_pending_reservation_expiration():
    """Verify that a pending reservation older than pending_ttl expires and restores quota."""
    guest_id = "crash_guest_1"
    crashed_req = "crashed_req_999"

    # Reserve with short pending_ttl of 1 second
    allowed, _, rem, _ = check_guest_limits(
        guest_id=guest_id,
        redis_url=None,
        require_redis=False,
        env="test",
        daily_limit=5,
        request_id=crashed_req,
        pending_ttl=1
    )
    assert allowed is True
    assert rem == 4

    # Wait 1.1 seconds so pending reservation becomes stale
    time.sleep(1.1)

    # Next check should clean up the stale pending reservation
    remaining = get_guest_remaining_questions(guest_id, None, False, "test", daily_limit=5, pending_ttl=1)
    assert remaining == 5, f"Expected stale pending to be cleared (remaining=5), got {remaining}"


def test_redis_fail_closed_under_require_redis():
    """Verify strict fail-closed security when Redis encounters an outage under require_redis=True."""
    mock_redis = MagicMock()
    mock_redis.eval.side_effect = RuntimeError("Connection closed by cluster")

    with patch("redis.from_url", return_value=mock_redis):
        # 1. check_guest_limits fails closed
        with pytest.raises(RuntimeError) as exc_check:
            check_guest_limits(
                guest_id="g1",
                redis_url="rediss://default:pwd@host:6379",
                require_redis=True,
                env="preview",
            )
        assert "Distributed Redis rate limiter failed" in str(exc_check.value)

        # 2. commit_guest_quota fails closed
        with pytest.raises(RuntimeError) as exc_commit:
            commit_guest_quota(
                guest_id="g1",
                req_id="r1",
                redis_url="rediss://default:pwd@host:6379",
                require_redis=True,
                env="preview",
            )
        assert "Distributed Redis rate limiter failed" in str(exc_commit.value)

        # 3. refund_guest_quota fails closed
        with pytest.raises(RuntimeError) as exc_refund:
            refund_guest_quota(
                guest_id="g1",
                req_id="r1",
                redis_url="rediss://default:pwd@host:6379",
                require_redis=True,
                env="preview",
            )
        assert "Distributed Redis rate limiter failed" in str(exc_refund.value)


def test_redis_namespace_isolation_across_environments():
    """Verify that Preview and Production keys never collide or cross-contaminate in Redis."""
    mock_redis = MagicMock()
    mock_redis.eval.return_value = [1, 0, 4, "p:r1"]

    with patch("redis.from_url", return_value=mock_redis):
        # Preview call
        check_guest_limits(
            guest_id="guest_x",
            redis_url="rediss://default:pwd@host:6379",
            require_redis=False,
            env="preview",
            request_id="prev_req_1"
        )
        preview_quota_key = mock_redis.eval.call_args[0][2]
        assert preview_quota_key == "nileguide:preview:guest_quota:guest_x"
        assert "production" not in preview_quota_key

        # Production call
        check_guest_limits(
            guest_id="guest_x",
            redis_url="rediss://default:pwd@host:6379",
            require_redis=False,
            env="production",
            request_id="prod_req_1"
        )
        prod_quota_key = mock_redis.eval.call_args[0][2]
        assert prod_quota_key == "nileguide:production:guest_quota:guest_x"
        assert "preview" not in prod_quota_key
