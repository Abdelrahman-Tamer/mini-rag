import sys
import logging
import pytest
from unittest.mock import MagicMock, patch
from qdrant_client import models

from scripts.ensure_qdrant_payload_indexes import (
    run_remediation,
    REQUIRED_PAYLOAD_INDEXES,
    mask_url,
    sanitize_error_message,
    resolve_credential_value,
    get_qdrant_credentials,
    inspect_collection_schema,
    main,
)


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.collection_exists.return_value = True

    # Default: mock empty payload schema
    info_mock = MagicMock()
    info_mock.payload_schema = {}
    info_mock.points_count = 100
    client.get_collection.return_value = info_mock

    # Mock scroll returning sample points
    client.scroll.return_value = ([], None)

    # Mock create_payload_index
    client.create_payload_index.return_value = models.UpdateResult(
        operation_id=1,
        status=models.UpdateStatus.COMPLETED
    )
    return client


# ==============================================================================
# REMEDIATION & SCHEMA TESTS
# ==============================================================================

def test_dry_run_does_not_mutate_qdrant(mock_qdrant_client):
    """Verify that dry-run mode inspects but never invokes create_payload_index."""
    exit_code = run_remediation(
        client=mock_qdrant_client,
        collection_name="tourism_knowledge_base",
        apply=False
    )
    assert exit_code == 0
    mock_qdrant_client.create_payload_index.assert_not_called()


def test_missing_indexes_detected(mock_qdrant_client, capsys):
    """Verify that when no indexes exist, all required indexes are detected as missing."""
    exit_code = run_remediation(
        client=mock_qdrant_client,
        collection_name="tourism_knowledge_base",
        apply=False
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    for field in REQUIRED_PAYLOAD_INDEXES.keys():
        assert field in captured
        assert "MISSING" in captured


def test_existing_correct_indexes_skipped(mock_qdrant_client):
    """Verify that existing indexes with matching schema are skipped in apply mode."""
    existing_schema = {}
    for f, req in REQUIRED_PAYLOAD_INDEXES.items():
        existing_schema[f] = MagicMock(data_type=req["schema"])

    info_mock = MagicMock()
    info_mock.payload_schema = existing_schema
    mock_qdrant_client.get_collection.return_value = info_mock

    exit_code = run_remediation(
        client=mock_qdrant_client,
        collection_name="tourism_knowledge_base",
        apply=True
    )
    assert exit_code == 0
    mock_qdrant_client.create_payload_index.assert_not_called()


def test_type_mismatch_stops_execution(mock_qdrant_client):
    """Verify that a schema type mismatch (e.g. metadata.is_active indexed as KEYWORD instead of BOOL) stops execution safely."""
    existing_schema = {
        "metadata.is_active": MagicMock(data_type=models.PayloadSchemaType.KEYWORD)
    }
    info_mock = MagicMock()
    info_mock.payload_schema = existing_schema
    mock_qdrant_client.get_collection.return_value = info_mock

    exit_code = run_remediation(
        client=mock_qdrant_client,
        collection_name="tourism_knowledge_base",
        apply=False
    )
    assert exit_code == 1
    mock_qdrant_client.create_payload_index.assert_not_called()


def test_unused_fields_not_indexed(mock_qdrant_client):
    """Verify that fields not required by retrieval pipeline are not indexed."""
    required_schema = {
        f: MagicMock(data_type=req["schema"])
        for f, req in REQUIRED_PAYLOAD_INDEXES.items()
    }
    info_mock = MagicMock()
    info_mock.payload_schema = required_schema
    mock_qdrant_client.get_collection.side_effect = [
        MagicMock(payload_schema={}),
        info_mock
    ]

    exit_code = run_remediation(
        client=mock_qdrant_client,
        collection_name="tourism_knowledge_base",
        apply=True
    )
    assert exit_code == 0

    called_fields = [
        call.kwargs.get("field_name")
        for call in mock_qdrant_client.create_payload_index.call_args_list
    ]
    assert "metadata.arbitrary_unused_field" not in called_fields
    assert "unrelated_field" not in called_fields
    assert len(called_fields) == len(REQUIRED_PAYLOAD_INDEXES)


def test_apply_mode_creates_only_confirmed_missing_indexes(mock_qdrant_client):
    """Verify that apply mode only creates missing fields, keeping existing untouched."""
    existing_schema = {
        "metadata.city": MagicMock(data_type=models.PayloadSchemaType.KEYWORD)
    }
    pre_info = MagicMock(payload_schema=existing_schema)

    all_schema = {
        f: MagicMock(data_type=req["schema"])
        for f, req in REQUIRED_PAYLOAD_INDEXES.items()
    }
    post_info = MagicMock(payload_schema=all_schema)

    mock_qdrant_client.get_collection.side_effect = [pre_info, post_info]

    exit_code = run_remediation(
        client=mock_qdrant_client,
        collection_name="tourism_knowledge_base",
        apply=True
    )
    assert exit_code == 0

    called_fields = [
        call.kwargs.get("field_name")
        for call in mock_qdrant_client.create_payload_index.call_args_list
    ]
    assert "metadata.city" not in called_fields
    assert "metadata.is_active" in called_fields


def test_verification_failure_returns_nonzero(mock_qdrant_client):
    """Verify that if post-apply verification finds missing indexes, utility returns non-zero."""
    pre_info = MagicMock(payload_schema={})
    post_info = MagicMock(payload_schema={})

    mock_qdrant_client.get_collection.side_effect = [pre_info, post_info]

    exit_code = run_remediation(
        client=mock_qdrant_client,
        collection_name="tourism_knowledge_base",
        apply=True
    )
    assert exit_code == 1


# ==============================================================================
# 8 REQUIRED CREDENTIAL & SECURITY TESTS
# ==============================================================================

def test_qdrant_db_api_key_resolved_successfully(monkeypatch):
    """Test 1: Verify QDRANT_DB_API_KEY is resolved successfully with source tracking."""
    monkeypatch.setenv("QDRANT_DB_URL", "https://prod-cluster.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_DB_API_KEY", "primary-secret-key-12345")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    url, api_key, collection, url_source, key_source, coll_source = get_qdrant_credentials()
    assert url == "https://prod-cluster.qdrant.io:6333"
    assert api_key == "primary-secret-key-12345"
    assert "QDRANT_DB_API_KEY" in key_source
    assert collection == "tourism_knowledge_base"


def test_qdrant_api_key_fallback_works(monkeypatch):
    """Test 2: Verify QDRANT_API_KEY alias fallback works when QDRANT_DB_API_KEY is not set."""
    monkeypatch.delenv("QDRANT_DB_API_KEY", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "fallback-key-67890")

    url, api_key, collection, url_source, key_source, coll_source = get_qdrant_credentials()
    assert api_key == "fallback-key-67890"
    assert "QDRANT_API_KEY" in key_source


def test_empty_environment_variables_handled(monkeypatch):
    """Test 3: Verify empty strings, spaces, and quotes are treated as unconfigured."""
    # Empty string should not be returned as valid key
    monkeypatch.setenv("QDRANT_DB_API_KEY", "")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    # Patch settings to empty so no fallback occurs
    with patch("helpers.config.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(QDRANT_DB_API_KEY="", QDRANT_DB_URL="", VECTOR_DB_COLLECTION_NAME="")
        val, source = resolve_credential_value(
            cli_val=None,
            env_names=["QDRANT_DB_API_KEY", "QDRANT_API_KEY"],
            settings_attr="QDRANT_DB_API_KEY"
        )
        assert val is None
        assert source == "[NOT_CONFIGURED]"

    # Whitespace and quotes
    monkeypatch.setenv("QDRANT_DB_API_KEY", "   ")
    with patch("helpers.config.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(QDRANT_DB_API_KEY="")
        val, source = resolve_credential_value(
            cli_val=None,
            env_names=["QDRANT_DB_API_KEY"],
            settings_attr="QDRANT_DB_API_KEY"
        )
        assert val is None
        assert source == "[NOT_CONFIGURED]"

    monkeypatch.setenv("QDRANT_DB_API_KEY", "''")
    with patch("helpers.config.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(QDRANT_DB_API_KEY="")
        val, source = resolve_credential_value(
            cli_val=None,
            env_names=["QDRANT_DB_API_KEY"],
            settings_attr="QDRANT_DB_API_KEY"
        )
        assert val is None
        assert source == "[NOT_CONFIGURED]"

    # Empty primary with valid fallback
    monkeypatch.setenv("QDRANT_DB_API_KEY", "")
    monkeypatch.setenv("QDRANT_API_KEY", "valid-fallback-key")
    val, source = resolve_credential_value(
        cli_val=None,
        env_names=["QDRANT_DB_API_KEY", "QDRANT_API_KEY"],
        settings_attr="QDRANT_DB_API_KEY"
    )
    assert val == "valid-fallback-key"
    assert "QDRANT_API_KEY" in source


def test_env_vars_precedence_over_defaults(monkeypatch):
    """Test 4: Verify environment variables take precedence over settings defaults, and CLI takes precedence over env."""
    monkeypatch.setenv("QDRANT_DB_API_KEY", "env-precedence-key")

    with patch("helpers.config.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(QDRANT_DB_API_KEY="settings-default-key")
        val, source = resolve_credential_value(
            cli_val=None,
            env_names=["QDRANT_DB_API_KEY"],
            settings_attr="QDRANT_DB_API_KEY"
        )
        assert val == "env-precedence-key"
        assert "QDRANT_DB_API_KEY" in source

    # CLI override should take precedence over env
    val, source = resolve_credential_value(
        cli_val="cli-override-key",
        env_names=["QDRANT_DB_API_KEY"],
        settings_attr="QDRANT_DB_API_KEY"
    )
    assert val == "cli-override-key"
    assert source == "CLI argument"


def test_qdrant_client_mock_receives_expected_api_key(monkeypatch):
    """Test 5: Verify QdrantClient mock is instantiated with the exact resolved API key."""
    monkeypatch.setenv("QDRANT_DB_URL", "https://cluster.eu-central-1.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_DB_API_KEY", "exact-expected-api-key-998877")

    with patch("scripts.ensure_qdrant_payload_indexes.QdrantClient") as mock_client_cls, \
         patch("scripts.ensure_qdrant_payload_indexes.run_remediation", return_value=0), \
         patch.object(sys, "argv", ["ensure_qdrant_payload_indexes.py"]):
        
        exit_code = main()
        assert exit_code == 0
        mock_client_cls.assert_called_once_with(
            url="https://cluster.eu-central-1.qdrant.io:6333",
            api_key="exact-expected-api-key-998877",
            timeout=10.0
        )


def test_missing_credentials_detected_before_network_requests(monkeypatch):
    """Test 6: Verify missing credentials exit with code 1 before any QdrantClient network requests."""
    # Missing API key
    monkeypatch.setenv("QDRANT_DB_URL", "https://cluster.eu-central-1.qdrant.io:6333")
    monkeypatch.delenv("QDRANT_DB_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with patch("helpers.config.get_settings") as mock_settings, \
         patch("scripts.ensure_qdrant_payload_indexes.QdrantClient") as mock_client_cls, \
         patch.object(sys, "argv", ["ensure_qdrant_payload_indexes.py"]):
        mock_settings.return_value = MagicMock(QDRANT_DB_API_KEY="")

        exit_code = main()
        assert exit_code == 1
        mock_client_cls.assert_not_called()

    # Missing URL
    monkeypatch.delenv("QDRANT_DB_URL", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with patch("helpers.config.get_settings") as mock_settings, \
         patch("scripts.ensure_qdrant_payload_indexes.QdrantClient") as mock_client_cls, \
         patch.object(sys, "argv", ["ensure_qdrant_payload_indexes.py"]):
        mock_settings.return_value = MagicMock(QDRANT_DB_URL="")

        exit_code = main()
        assert exit_code == 1
        mock_client_cls.assert_not_called()


def test_no_credentials_appear_in_logs_or_exceptions(monkeypatch, caplog):
    """Test 7: Verify zero credentials or partial characters appear in logs or exceptions."""
    secret_key = "super_secret_jwt_token_payload_xyz123"
    monkeypatch.setenv("QDRANT_DB_URL", "https://cluster.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_DB_API_KEY", secret_key)

    with patch("scripts.ensure_qdrant_payload_indexes.QdrantClient"), \
         patch("scripts.ensure_qdrant_payload_indexes.run_remediation", return_value=0), \
         patch.object(sys, "argv", ["ensure_qdrant_payload_indexes.py"]), \
         caplog.at_level(logging.INFO):

        exit_code = main()
        assert exit_code == 0

        # Verify secret_key is not in any logged message
        for record in caplog.records:
            assert secret_key not in record.message
            assert "super_secret" not in record.message

        # Verify only boolean status is logged
        log_text = " ".join([r.message for r in caplog.records])
        assert "API Key Status             : CONFIGURED (source: environment variable 'QDRANT_DB_API_KEY')" in log_text

    # Verify exception sanitization
    raw_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6OTY3In0.1234567890abcdef"
    err_msg = f"HTTP 403 Forbidden with Bearer my_secret_token_123 and token {raw_jwt} at https://user:pass@qdrant.io"
    sanitized = sanitize_error_message(err_msg)
    assert raw_jwt not in sanitized
    assert "my_secret_token_123" not in sanitized
    assert "pass" not in sanitized
    assert "[REDACTED_JWT]" in sanitized
    assert "[REDACTED_TOKEN]" in sanitized
    assert "://***:***@" in sanitized


def test_dry_run_and_apply_resolve_identically(monkeypatch):
    """Test 8: Verify both dry-run and apply modes resolve credentials and configuration identically."""
    monkeypatch.setenv("QDRANT_DB_URL", "https://cluster.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_DB_API_KEY", "identical-test-key-554433")
    monkeypatch.setenv("VECTOR_DB_COLLECTION_NAME", "tourism_knowledge_base")

    # Dry-run
    res_dry = get_qdrant_credentials()
    # Apply mode (credentials resolution does not depend on --apply flag)
    res_apply = get_qdrant_credentials()

    assert res_dry == res_apply
    assert res_dry[0] == "https://cluster.qdrant.io:6333"
    assert res_dry[1] == "identical-test-key-554433"
    assert res_dry[2] == "tourism_knowledge_base"


# ==============================================================================
# PREFLIGHT CHECK CREDENTIALS TEST
# ==============================================================================

def test_preflight_check_credentials(monkeypatch, mock_qdrant_client):
    """Verify --check-credentials verifies connection and exits 0 without calling run_remediation."""
    monkeypatch.setenv("QDRANT_DB_URL", "https://cluster.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_DB_API_KEY", "valid-key-123")

    with patch("scripts.ensure_qdrant_payload_indexes.QdrantClient", return_value=mock_qdrant_client), \
         patch("scripts.ensure_qdrant_payload_indexes.run_remediation") as mock_remediation, \
         patch.object(sys, "argv", ["ensure_qdrant_payload_indexes.py", "--check-credentials"]):

        exit_code = main()
        assert exit_code == 0
        mock_qdrant_client.collection_exists.assert_called_once_with(collection_name="tourism_knowledge_base")
        mock_remediation.assert_not_called()
