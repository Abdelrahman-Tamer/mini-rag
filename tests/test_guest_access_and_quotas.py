import time
import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request
from fastapi.testclient import TestClient

from main import app
from helpers.guest_session import (
    generate_guest_token,
    verify_guest_token,
    resolve_guest_secret,
    InMemoryGuestLimiter,
    check_guest_limits,
    get_guest_remaining_questions,
    get_guest_conversation_history,
    append_guest_conversation_turn,
    _in_memory_guest_limiter,
)
from helpers.config import Settings


@pytest.fixture(autouse=True)
def reset_in_memory_limiter():
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


# ==============================================================================
# SECTION 1: CRYPTOGRAPHIC GUEST TOKEN TESTS
# ==============================================================================

def test_generate_and_verify_guest_token_roundtrip():
    secret = "test-secret-key-for-hmac-sha256"
    guest_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex

    token = generate_guest_token(guest_id, session_id, secret)
    assert isinstance(token, str)
    assert len(token.split(".")) == 4

    verified = verify_guest_token(token, secret)
    assert verified is not None
    v_guest_id, v_session_id, v_ts = verified
    assert v_guest_id == guest_id
    assert v_session_id == session_id
    assert abs(time.time() - v_ts) < 5


def test_verify_guest_token_tampering_detected():
    secret = "test-secret-key-for-hmac-sha256"
    guest_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    token = generate_guest_token(guest_id, session_id, secret)
    parts = token.split(".")

    # Tamper guest_id
    tampered_guest = f"hacked_{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"
    assert verify_guest_token(tampered_guest, secret) is None

    # Tamper session_id
    tampered_session = f"{parts[0]}.hacked_{parts[1]}.{parts[2]}.{parts[3]}"
    assert verify_guest_token(tampered_session, secret) is None

    # Tamper timestamp
    tampered_ts = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 10}.{parts[3]}"
    assert verify_guest_token(tampered_ts, secret) is None

    # Tamper signature
    tampered_sig = f"{parts[0]}.{parts[1]}.{parts[2]}.{'0' if parts[3][0] != '0' else '1'}{parts[3][1:]}"
    assert verify_guest_token(tampered_sig, secret) is None

    # Wrong secret
    assert verify_guest_token(token, "wrong-secret-key") is None


def test_verify_guest_token_edge_cases():
    secret = "test-secret"
    assert verify_guest_token("", secret) is None
    assert verify_guest_token(None, secret) is None
    assert verify_guest_token("a.b.c", secret) is None
    assert verify_guest_token("a.b.c.d.e", secret) is None
    assert verify_guest_token("guest.sess.notanumber.sig", secret) is None

    # Future timestamp (> 300s)
    future_ts = int(time.time()) + 500
    future_token = generate_guest_token("g", "s", secret)
    parts = future_token.split(".")
    import hmac, hashlib
    payload = f"g.s.{future_ts}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    assert verify_guest_token(f"{payload}.{sig}", secret) is None

    # Expired token (> 7 days)
    past_ts = int(time.time()) - (86400 * 8)
    past_payload = f"g.s.{past_ts}"
    past_sig = hmac.new(secret.encode(), past_payload.encode(), hashlib.sha256).hexdigest()
    assert verify_guest_token(f"{past_payload}.{past_sig}", secret) is None


# ==============================================================================
# SECTION 2: SLIDING WINDOW QUOTA & RATE LIMITING
# ==============================================================================

def test_guest_quota_enforcement_five_questions():
    guest_id = "guest_test_quota_1"
    env = "test"

    for expected_remaining in [4, 3, 2, 1, 0]:
        allowed, retry_after, remaining, err = check_guest_limits(
            guest_id=guest_id,
            redis_url=None,
            require_redis=False,
            env=env,
            daily_limit=5,
            daily_window=86400,
            burst_limit=10,
            burst_window=60,
        )
        assert allowed is True
        assert remaining == expected_remaining
        assert err == ""

    # 6th question must be rejected
    allowed, retry_after, remaining, err = check_guest_limits(
        guest_id=guest_id,
        redis_url=None,
        require_redis=False,
        env=env,
        daily_limit=5,
        daily_window=86400,
        burst_limit=10,
        burst_window=60,
    )
    assert allowed is False
    assert remaining == 0
    assert err == "daily_quota_exceeded"
    assert retry_after > 0


def test_guest_burst_limit_ten_requests():
    guest_id = "guest_test_burst_1"
    env = "test"

    # 10 burst requests with 100 daily limit to isolate burst
    for _ in range(10):
        allowed, _, _, err = check_guest_limits(
            guest_id=guest_id,
            redis_url=None,
            require_redis=False,
            env=env,
            daily_limit=100,
            daily_window=86400,
            burst_limit=10,
            burst_window=60,
        )
        assert allowed is True
        assert err == ""

    # 11th request exceeds burst limit
    allowed, retry_after, _, err = check_guest_limits(
        guest_id=guest_id,
        redis_url=None,
        require_redis=False,
        env=env,
        daily_limit=100,
        daily_window=86400,
        burst_limit=10,
        burst_window=60,
    )
    assert allowed is False
    assert err == "burst_limit_exceeded"
    assert retry_after > 0


# ==============================================================================
# SECTION 3: TEMPORARY CONVERSATION BUFFER
# ==============================================================================

def test_guest_temporary_conversation_turns_and_ttl():
    session_id = uuid.uuid4().hex
    env = "test"

    # Initially empty
    history = get_guest_conversation_history(session_id, None, False, env, ttl=1800)
    assert history == []

    # Append first turn
    append_guest_conversation_turn(
        session_id=session_id,
        user_msg="?? ?? ???? ??????? ?? ????????",
        assistant_msg="???? ??????? ?? ??? ??????? ???? ???????.",
        sources=[{"title": "Khan el-Khalili"}],
        redis_url=None,
        require_redis=False,
        env=env,
        ttl=1800,
    )

    history = get_guest_conversation_history(session_id, None, False, env, ttl=1800)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert "??? ???????" in history[1]["content"]

    # Append follow-up turn
    append_guest_conversation_turn(
        session_id=session_id,
        user_msg="??? ????????",
        assistant_msg="?????? ??? ??? ??????? ?????.",
        sources=[],
        redis_url=None,
        require_redis=False,
        env=env,
        ttl=1800,
    )

    history2 = get_guest_conversation_history(session_id, None, False, env, ttl=1800)
    assert len(history2) == 4
    assert history2[2]["content"] == "??? ????????"


# ==============================================================================
# SECTION 4: API ENDPOINT INTEGRATION TESTS
# ==============================================================================

def test_api_create_guest_session_endpoint(mock_app_client):
    response = mock_app_client.post("/api/v1/nlp/guest/session")
    assert response.status_code == 200
    data = response.json()
    assert "guest_token" in data
    assert "guest_id" in data
    assert "session_id" in data
    assert data["daily_limit"] == 5
    assert data["remaining_questions"] == 5
    assert data["inactivity_ttl_seconds"] == 1800


def test_api_guest_chat_greeting(mock_app_client):
    sess_res = mock_app_client.post("/api/v1/nlp/guest/session")
    token = sess_res.json()["guest_token"]

    chat_res = mock_app_client.post(
        "/api/v1/nlp/chat",
        json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": token}
    )
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert data["signal"] == "chat_success"
    assert "NileGuide" in data["answer"]
    assert data["guest_remaining_questions"] == 4
    assert data["guest_daily_limit"] == 5


def test_api_guest_chat_invalid_token(mock_app_client):
    chat_res = mock_app_client.post(
        "/api/v1/nlp/chat",
        json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": "fake.invalid.token.123"}
    )
    assert chat_res.status_code == 401
    data = chat_res.json()
    assert data["signal"] == "invalid_guest_token"


def test_api_guest_chat_quota_exhaustion(mock_app_client):
    sess_res = mock_app_client.post("/api/v1/nlp/guest/session")
    token = sess_res.json()["guest_token"]

    for i in range(5):
        res = mock_app_client.post(
            "/api/v1/nlp/chat",
            json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": token}
        )
        assert res.status_code == 200
        assert res.json()["guest_remaining_questions"] == 4 - i

    # 6th request must trigger 429 guest_quota_exceeded
    res6 = mock_app_client.post(
        "/api/v1/nlp/chat",
        json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": token}
    )
    assert res6.status_code == 429
    data = res6.json()
    assert data["signal"] == "guest_quota_exceeded"
    assert data["remaining_questions"] == 0
    assert "???? ???? ??????" in data["message"]


def test_api_guest_chat_zero_mongodb_writes(mock_app_client):
    sess_res = mock_app_client.post("/api/v1/nlp/guest/session")
    token = sess_res.json()["guest_token"]

    mock_db = app.db_client
    mock_db.reset_mock()

    res = mock_app_client.post(
        "/api/v1/nlp/chat",
        json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": token, "conversation_id": "malicious_conv_id"}
    )
    assert res.status_code == 200
    # MongoDB db_client must NEVER be accessed
    assert mock_db.get_database.call_count == 0


def test_api_guest_fail_closed_under_require_redis(mock_app_client):
    with patch("routes.nlp.get_settings") as mock_settings:
        mock_settings.return_value = Settings(
            REQUIRE_REDIS_RATE_LIMITER=True,
            REDIS_URL=None,
            ADMIN_API_KEY="test-admin-key-123456"
        )
        # Attempt to create guest session without Redis when mandatory
        sess_res = mock_app_client.post("/api/v1/nlp/guest/session")
        assert sess_res.status_code == 503
        assert sess_res.json()["signal"] == "service_temporarily_unavailable"


def test_api_guest_storage_clear_cannot_reset_quota(mock_app_client):
    """
    Phase B Security Proof:
    Clearing browser storage or requesting a new session from the same network
    MUST NOT reset the 5-question daily quota.
    """
    # 1. Exhaust quota for client
    sess1 = mock_app_client.post("/api/v1/nlp/guest/session")
    token1 = sess1.json()["guest_token"]
    for _ in range(5):
        res = mock_app_client.post("/api/v1/nlp/chat", json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": token1})
        assert res.status_code == 200

    # 2. Simulate user clearing browser storage and requesting a fresh guest session
    sess2 = mock_app_client.post("/api/v1/nlp/guest/session")
    assert sess2.status_code == 200
    data2 = sess2.json()
    assert data2["remaining_questions"] == 0
    token2 = data2["guest_token"]

    # 3. Chatting with new token must immediately trigger 429 quota exceeded
    chat_res = mock_app_client.post("/api/v1/nlp/chat", json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": token2})
    assert chat_res.status_code == 429
    assert chat_res.json()["signal"] == "guest_quota_exceeded"
    assert chat_res.json()["remaining_questions"] == 0


def test_api_guest_ip_mismatch_rejected(mock_app_client):
    """
    Security Proof:
    Tokens bound to IP A cannot be stolen or used by a caller on IP B.
    """
    # Session created by client at 198.51.100.1
    sess = mock_app_client.post(
        "/api/v1/nlp/guest/session",
        headers={"X-Forwarded-For": "198.51.100.1"}
    )
    token = sess.json()["guest_token"]

    # Attacker or different network at 203.0.113.99 tries to use token
    chat_res = mock_app_client.post(
        "/api/v1/nlp/chat",
        json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": token},
        headers={"X-Forwarded-For": "203.0.113.99"}
    )
    assert chat_res.status_code == 403
    data = chat_res.json()
    assert data["signal"] == "guest_session_ip_mismatch"


def test_api_guest_different_ips_isolated_quotas(mock_app_client):
    """
    Security Proof:
    Quotas for distinct client networks remain strictly isolated.
    """
    # Client A exhausts all 5 questions
    sess_a = mock_app_client.post(
        "/api/v1/nlp/guest/session",
        headers={"X-Forwarded-For": "198.51.100.10"}
    )
    token_a = sess_a.json()["guest_token"]
    for _ in range(5):
        res = mock_app_client.post(
            "/api/v1/nlp/chat",
            json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": token_a},
            headers={"X-Forwarded-For": "198.51.100.10"}
        )
        assert res.status_code == 200

    # Client B starts a fresh session
    sess_b = mock_app_client.post(
        "/api/v1/nlp/guest/session",
        headers={"X-Forwarded-For": "198.51.100.20"}
    )
    assert sess_b.status_code == 200
    assert sess_b.json()["remaining_questions"] == 5
    token_b = sess_b.json()["guest_token"]

    # Client B can chat normally
    res_b = mock_app_client.post(
        "/api/v1/nlp/chat",
        json={"text": "\u0645\u0631\u062d\u0628\u0627", "guest_token": token_b},
        headers={"X-Forwarded-For": "198.51.100.20"}
    )
    assert res_b.status_code == 200
    assert res_b.json()["guest_remaining_questions"] == 4
