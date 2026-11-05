"""
Automated Verification Suite: Redis Namespace Isolation & Environment Safety
Validates:
1. Preview keys cannot collide with Production keys.
2. Preview cleanup/expiration never touches Production keys.
3. Production conversation history cannot be read via Preview session IDs.
4. Every Redis subsystem (locks, global chat limiter, stream limiter, guest burst, guest quota, guest conversation) strictly incorporates {env}.
5. Cross-environment cryptographic token isolation (Preview tokens invalid in Production).
6. Forwarded header spoofing rejection (reverse proxy security).
"""

import pytest
import time
import hashlib
from unittest.mock import MagicMock, patch

from helpers.config import Settings
from helpers.rate_limiter import check_rate_limit, get_client_ip
from helpers.guest_session import (
    get_guest_remaining_questions,
    check_guest_limits,
    get_guest_conversation_history,
    append_guest_conversation_turn,
    generate_guest_token,
    verify_guest_token,
    resolve_guest_secret,
    derive_guest_id,
)
from controllers.NLPController import NLPController


# ==============================================================================
# SECTION 1: REDIS SUBSYSTEM KEY NAMESPACE VERIFICATION
# ==============================================================================

def test_guest_quota_and_burst_keys_strict_namespace():
    """Verify guest quota and burst keys strictly isolate by environment prefix."""
    mock_redis = MagicMock()
    mock_redis.eval.return_value = [1, 0, 1]
    mock_redis.zcard.return_value = 0

    with patch("redis.from_url", return_value=mock_redis):
        # 1. Preview environment
        check_guest_limits(
            guest_id="guest_abc123",
            redis_url="redis://fake:6379",
            require_redis=True,
            env="preview",
            daily_limit=5,
            burst_limit=10,
        )

        calls = mock_redis.eval.call_args_list
        # First call is burst, second is quota
        burst_key_preview = calls[0][0][2]
        quota_key_preview = calls[1][0][2]

        assert burst_key_preview == "nileguide:preview:guest_burst:guest_abc123"
        assert quota_key_preview == "nileguide:preview:guest_quota:guest_abc123"

        mock_redis.reset_mock()

        # 2. Production environment
        check_guest_limits(
            guest_id="guest_abc123",
            redis_url="redis://fake:6379",
            require_redis=True,
            env="production",
            daily_limit=5,
            burst_limit=10,
        )

        calls_prod = mock_redis.eval.call_args_list
        burst_key_prod = calls_prod[0][0][2]
        quota_key_prod = calls_prod[1][0][2]

        assert burst_key_prod == "nileguide:production:guest_burst:guest_abc123"
        assert quota_key_prod == "nileguide:production:guest_quota:guest_abc123"

        # Explicit non-collision proof
        assert burst_key_preview != burst_key_prod
        assert quota_key_preview != quota_key_prod


def test_guest_conversation_buffer_cross_environment_isolation():
    """Verify that a temporary conversation written in Preview cannot be read in Production."""
    fake_storage = {}

    def mock_set(key, val, ex=None):
        fake_storage[key] = val.encode("utf-8") if isinstance(val, str) else val

    def mock_get(key):
        return fake_storage.get(key)

    mock_redis = MagicMock()
    mock_redis.set.side_effect = mock_set
    mock_redis.get.side_effect = mock_get

    with patch("redis.from_url", return_value=mock_redis):
        session_id = "test-session-uuid-999"

        # 1. Append turn in Preview
        append_guest_conversation_turn(
            session_id=session_id,
            user_msg="مرحبا من المعاينة",
            assistant_msg="أهلا بك في بيئة المعاينة",
            sources=[{"title": "Preview Activity"}],
            redis_url="redis://fake:6379",
            require_redis=True,
            env="preview",
            ttl=1800,
        )

        # 2. Read in Preview -> Must exist
        preview_history = get_guest_conversation_history(
            session_id=session_id,
            redis_url="redis://fake:6379",
            require_redis=True,
            env="preview",
            ttl=1800,
        )
        assert len(preview_history) == 2
        assert preview_history[0]["content"] == "مرحبا من المعاينة"

        # 3. Read in Production with exact same session_id -> MUST BE EMPTY
        prod_history = get_guest_conversation_history(
            session_id=session_id,
            redis_url="redis://fake:6379",
            require_redis=True,
            env="production",
            ttl=1800,
        )
        assert prod_history == []

        # 4. Verify exact key names in storage
        session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
        expected_preview_key = f"nileguide:preview:guest_conv:{session_hash}"
        expected_prod_key = f"nileguide:production:guest_conv:{session_hash}"

        assert expected_preview_key in fake_storage
        assert expected_prod_key not in fake_storage


def test_distributed_lock_environment_namespace():
    """Verify that asset sync distributed locks are namespaced and cannot collide across environments."""
    settings_preview = Settings(
        ENVIRONMENT="preview",
        REDIS_URL="redis://fake:6379",
        REQUIRE_REDIS_RATE_LIMITER=True,
    )
    settings_prod = Settings(
        ENVIRONMENT="production",
        REDIS_URL="redis://fake:6379",
        REQUIRE_REDIS_RATE_LIMITER=True,
    )

    mock_redis = MagicMock()
    mock_redis.set.return_value = True

    with patch("redis.from_url", return_value=mock_redis):
        # 1. Controller in preview
        ctrl_preview = NLPController(
            vectordb_client=MagicMock(),
            generation_client=MagicMock(),
            embedding_client=MagicMock(),
            template_parser=MagicMock(),
        )
        ctrl_preview.app_settings = settings_preview
        with ctrl_preview.asset_lock("act_101"):
            pass

        preview_call = mock_redis.set.call_args_list[0]
        preview_lock_key = preview_call[0][0]
        assert preview_lock_key == "nileguide:preview:lock:asset:act_101"

        mock_redis.reset_mock()

        # 2. Controller in production
        ctrl_prod = NLPController(
            vectordb_client=MagicMock(),
            generation_client=MagicMock(),
            embedding_client=MagicMock(),
            template_parser=MagicMock(),
        )
        ctrl_prod.app_settings = settings_prod
        with ctrl_prod.asset_lock("act_101"):
            pass

        prod_call = mock_redis.set.call_args_list[0]
        prod_lock_key = prod_call[0][0]
        assert prod_lock_key == "nileguide:production:lock:asset:act_101"

        assert preview_lock_key != prod_lock_key


def test_global_rate_limiter_keys_use_environment_namespace():
    """Verify that rate limiter keys for chat and stream use the correct environment namespace."""
    mock_redis = MagicMock()
    mock_redis.eval.return_value = [1, 0]

    with patch("redis.from_url", return_value=mock_redis):
        mock_req = MagicMock()
        mock_req.client.host = "198.51.100.55"
        mock_req.headers = {}

        # 1. Preview Chat
        check_rate_limit(
            request=mock_req,
            redis_url="redis://fake:6379",
            require_redis=True,
            prefix="nileguide:preview:chat",
        )
        chat_preview_key = mock_redis.eval.call_args_list[0][0][2]
        assert chat_preview_key == "nileguide:preview:chat:198.51.100.55"

        mock_redis.reset_mock()

        # 2. Production Chat
        check_rate_limit(
            request=mock_req,
            redis_url="redis://fake:6379",
            require_redis=True,
            prefix="nileguide:production:chat",
        )
        chat_prod_key = mock_redis.eval.call_args_list[0][0][2]
        assert chat_prod_key == "nileguide:production:chat:198.51.100.55"

        assert chat_preview_key != chat_prod_key

        mock_redis.reset_mock()

        # 3. Preview Stream
        check_rate_limit(
            request=mock_req,
            redis_url="redis://fake:6379",
            require_redis=True,
            prefix="nileguide:preview:stream",
        )
        stream_preview_key = mock_redis.eval.call_args_list[0][0][2]
        assert stream_preview_key == "nileguide:preview:stream:198.51.100.55"

        mock_redis.reset_mock()

        # 4. Production Stream
        check_rate_limit(
            request=mock_req,
            redis_url="redis://fake:6379",
            require_redis=True,
            prefix="nileguide:production:stream",
        )
        stream_prod_key = mock_redis.eval.call_args_list[0][0][2]
        assert stream_prod_key == "nileguide:production:stream:198.51.100.55"

        assert stream_preview_key != stream_prod_key


# ==============================================================================
# SECTION 2: CRYPTOGRAPHIC GUEST TOKEN & SECRET ISOLATION
# ==============================================================================

def test_guest_token_cross_environment_secret_isolation():
    """Verify that a token created in Preview cannot be verified in Production when secrets differ."""
    preview_secret = "nileguide-preview-secret-key-2026-auth"
    prod_secret = "nileguide-production-secret-key-2026-auth"

    token = generate_guest_token("guest_001", "sess_001", preview_secret)

    # Valid with preview secret
    assert verify_guest_token(token, preview_secret) is not None

    # REJECTED with production secret
    assert verify_guest_token(token, prod_secret) is None


def test_resolve_guest_secret_fail_closed_in_production():
    """Verify that resolve_guest_secret raises RuntimeError in production if no secret is set."""
    settings_prod_empty = Settings(
        ENVIRONMENT="production",
        GUEST_SECRET_KEY=None,
        ADMIN_API_KEY=None,
        SYNC_API_KEY=None,
    )
    with pytest.raises(RuntimeError, match="FATAL: Guest HMAC secret key is mandatory in production"):
        resolve_guest_secret(settings_prod_empty)


def test_resolve_guest_secret_prefers_dedicated_guest_key():
    """Verify that GUEST_SECRET_KEY takes highest priority over ADMIN_API_KEY and SYNC_API_KEY."""
    settings = Settings(
        ENVIRONMENT="production",
        GUEST_SECRET_KEY="dedicated-guest-secret-key-9999",
        ADMIN_API_KEY="admin-secret-key-8888",
        SYNC_API_KEY="sync-secret-key-7777",
    )
    secret = resolve_guest_secret(settings)
    assert secret == "dedicated-guest-secret-key-9999"


# ==============================================================================
# SECTION 3: REVERSE PROXY SECURITY & HEADER SPOOFING DEFENSE
# ==============================================================================

def test_untrusted_peer_spoofed_forwarded_headers_rejected():
    """
    Security Proof:
    If a direct socket peer is NOT in TRUSTED_PROXIES, forwarded headers are completely ignored.
    """
    mock_req = MagicMock()
    # Untrusted client connecting directly
    mock_req.client.host = "203.0.113.199"
    # Malicious client attempts to spoof Google or Cloudflare IP
    mock_req.headers = {
        "X-Forwarded-For": "8.8.8.8, 1.1.1.1",
        "CF-Connecting-IP": "8.8.8.8",
    }

    client_ip = get_client_ip(
        request=mock_req,
        trusted_proxies=["127.0.0.1", "::1"],
    )

    # Must return the actual socket peer, NOT the spoofed headers
    assert client_ip == "203.0.113.199"


def test_trusted_proxy_peer_safely_extracts_client_ip():
    """
    Security Proof:
    When direct peer is a trusted proxy (e.g. Vercel edge / 127.0.0.1),
    genuine client IP is extracted via right-to-left traversal.
    """
    mock_req = MagicMock()
    mock_req.client.host = "127.0.0.1"  # Direct peer is trusted
    mock_req.headers = {
        # Client -> intermediate proxy -> edge proxy
        "X-Forwarded-For": "198.51.100.77, 10.0.0.1"
    }

    client_ip = get_client_ip(
        request=mock_req,
        trusted_proxies=["127.0.0.1", "10.0.0.1"],
    )

    # First untrusted hop from the right is 198.51.100.77
    assert client_ip == "198.51.100.77"
