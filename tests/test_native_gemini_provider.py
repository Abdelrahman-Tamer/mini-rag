import os
import sys
import uuid
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

# Add project root and api to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "api"))
sys.path.insert(0, str(ROOT_DIR))

from google.genai import types
from stores.llm.providers.GoogleGenAIProvider import GoogleGenAIProvider, _sanitize_error_message
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.LLMEnums import LLMEnums, GoogleGenAIEnums
from helpers.config import Settings
from helpers.guest_session import (
    _in_memory_guest_limiter,
    derive_guest_id,
    generate_guest_token,
    get_guest_remaining_questions,
)
from models.db_schemes import RetrievedDocument

DEV_SECRET = "nileguide-guest-secret-2026-safe-dev"


@pytest.fixture(autouse=True)
def reset_in_memory_state():
    _in_memory_guest_limiter.reset()
    yield
    _in_memory_guest_limiter.reset()


def _create_mock_response(text: str = "مرحباً بك في الجيزة", finish_reason = types.FinishReason.STOP, prompt_tokens: int = 1200, candidate_tokens: int = 250):
    resp = MagicMock()
    resp.text = text
    cand = MagicMock()
    cand.finish_reason = finish_reason
    resp.candidates = [cand]
    resp.usage_metadata = MagicMock(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidate_tokens,
        thoughts_token_count=50,
        total_token_count=prompt_tokens + candidate_tokens + 50
    )
    return resp


# ---------------------------------------------------------------------
# 1. Successful native generation
# ---------------------------------------------------------------------
def test_successful_native_generation():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    mock_resp = _create_mock_response(text="Here are 3 activities in Giza.")
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, return_value=mock_resp) as mock_gen:
        result = asyncio.run(provider.generate_text_async("Tell me about Giza"))
        assert result == "Here are 3 activities in Giza."
        assert mock_gen.called

    # Also test sync wrapper
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, return_value=mock_resp):
        sync_result = provider.generate_text("Tell me about Giza")
        assert sync_result == "Here are 3 activities in Giza."


# ---------------------------------------------------------------------
# 2. Correct Arabic answer extraction
# ---------------------------------------------------------------------
def test_correct_arabic_answer_extraction():
    arabic_text = "أهلاً بك! إليك 3 أنشطة سياحية رائعة في الجيزة: أهرامات الجيزة، والمتحف المصري الكبير، وركوب الخيل."
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    mock_resp = _create_mock_response(text=arabic_text)
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, return_value=mock_resp):
        result = asyncio.run(provider.generate_text_async("أنشطة الجيزة"))
        assert result == arabic_text
        assert "أهرامات الجيزة" in result


# ---------------------------------------------------------------------
# 3. Correct model selection
# ---------------------------------------------------------------------
def test_correct_model_selection():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    provider.set_generation_model("gemini-3.5-flash")
    mock_resp = _create_mock_response()
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, return_value=mock_resp) as mock_gen:
        asyncio.run(provider.generate_text_async("test prompt"))
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.5-flash"


# ---------------------------------------------------------------------
# 4. Correct native credential selection
# ---------------------------------------------------------------------
def test_correct_native_credential_selection():
    cfg = Settings(
        GEMINI_API_KEY="AIzaSyDedicatedNativeGeminiKey12345",
        OPENAI_API_KEY="sk-OpenAIKeyForEmbeddingsOnly",
        LLM_GENERATION_PROVIDER="native_gemini",
    )
    factory = LLMProviderFactory(cfg)
    provider = factory.create("native_gemini")
    
    assert isinstance(provider, GoogleGenAIProvider)
    assert provider.api_key == "AIzaSyDedicatedNativeGeminiKey12345"
    assert provider.api_key != "sk-OpenAIKeyForEmbeddingsOnly"


# ---------------------------------------------------------------------
# 5. Missing GEMINI_API_KEY raises ValueError
# ---------------------------------------------------------------------
def test_missing_gemini_api_key():
    cfg = Settings(
        GEMINI_API_KEY="",
        OPENAI_API_KEY="sk-some-key",
        LLM_GENERATION_PROVIDER="native_gemini",
    )
    factory = LLMProviderFactory(cfg)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        factory.create("native_gemini")

    with pytest.raises(ValueError, match="GEMINI_API_KEY must be provided"):
        GoogleGenAIProvider(api_key="")


# ---------------------------------------------------------------------
# 6. Missing response text returns None
# ---------------------------------------------------------------------
def test_missing_response_text():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    mock_resp = _create_mock_response(text="")
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, return_value=mock_resp):
        result = asyncio.run(provider.generate_text_async("query"))
        assert result is None


# ---------------------------------------------------------------------
# 7. MAX_TOKENS is treated as incomplete (fails closed, returns None)
# ---------------------------------------------------------------------
def test_max_tokens_treated_as_incomplete():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    mock_resp = _create_mock_response(text="أهلاً بك! إليك ترشيح لـ 3 أنشطة:\n\n###", finish_reason=types.FinishReason.MAX_TOKENS)
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, return_value=mock_resp):
        result = asyncio.run(provider.generate_text_async("query"))
        assert result is None, "MAX_TOKENS must return None to trigger controlled quota refund"


# ---------------------------------------------------------------------
# 8. STOP is treated as successful
# ---------------------------------------------------------------------
def test_stop_treated_as_successful():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    mock_resp = _create_mock_response(text="Complete tourist answer.", finish_reason=types.FinishReason.STOP)
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, return_value=mock_resp):
        result = asyncio.run(provider.generate_text_async("query"))
        assert result == "Complete tourist answer."


# ---------------------------------------------------------------------
# 9. 403 permission error handled safely without key exposure
# ---------------------------------------------------------------------
def test_403_permission_error_handled_safely():
    provider = GoogleGenAIProvider(api_key="AIzaSySecretKey987654321")
    ex = Exception("403 PERMISSION_DENIED: API_KEY_SERVICE_BLOCKED with AIzaSySecretKey987654321")
    setattr(ex, "code", 403)
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, side_effect=ex):
        result = asyncio.run(provider.generate_text_async("query"))
        assert result is None
    
    sanitized = _sanitize_error_message(str(ex))
    assert "AIzaSySecretKey987654321" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


# ---------------------------------------------------------------------
# 10. 429 rate limit handled safely
# ---------------------------------------------------------------------
def test_429_rate_limit_handled_safely():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    ex = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
    setattr(ex, "code", 429)
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, side_effect=ex):
        result = asyncio.run(provider.generate_text_async("query"))
        assert result is None


# ---------------------------------------------------------------------
# 11. 503 service unavailable handled safely
# ---------------------------------------------------------------------
def test_503_service_unavailable_handled_safely():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    ex = Exception("503 UNAVAILABLE: This model is currently experiencing high demand")
    setattr(ex, "code", 503)
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, side_effect=ex):
        result = asyncio.run(provider.generate_text_async("query"))
        assert result is None


# ---------------------------------------------------------------------
# 12. Timeout behavior handled safely
# ---------------------------------------------------------------------
def test_timeout_behavior_handled_safely():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    ex = TimeoutError("Deadline exceeded after 25000ms")
    
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, side_effect=ex):
        result = asyncio.run(provider.generate_text_async("query"))
        assert result is None


# ---------------------------------------------------------------------
# 13. No automatic retries configured
# ---------------------------------------------------------------------
def test_no_automatic_retries_configured():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    assert provider.retry_options.attempts == 1
    assert provider.http_options.retry_options.attempts == 1


# ---------------------------------------------------------------------
# 14. No legacy bridge fallback in native mode
# ---------------------------------------------------------------------
def test_no_legacy_bridge_fallback_in_native_mode():
    provider = GoogleGenAIProvider(api_key="test_gemini_key_12345")
    provider.set_generation_model("gemini-3.5-flash")
    provider.set_fallback_model("gemini-3.5-flash-lite")
    
    ex = Exception("503 Server Error")
    with patch.object(provider.client.aio.models, "generate_content", new_callable=AsyncMock, side_effect=ex) as mock_gen:
        result = asyncio.run(provider.generate_text_async("query"))
        assert result is None
        assert mock_gen.call_count == 1


# ---------------------------------------------------------------------
# 15. No change to embedding credential usage
# ---------------------------------------------------------------------
def test_no_change_to_embedding_credential_usage():
    cfg = Settings(
        GEMINI_API_KEY="AIzaSyNativeGeminiKey",
        OPENAI_API_KEY="sk-OpenAIKeyForEmbeddings",
        LLM_GENERATION_PROVIDER="native_gemini",
        EMBEDDING_BACKEND="OPENAI",
    )
    factory = LLMProviderFactory(cfg)
    
    gen_provider = factory.create("native_gemini")
    emb_provider = factory.create(cfg.EMBEDDING_BACKEND)
    
    assert isinstance(gen_provider, GoogleGenAIProvider)
    assert gen_provider.api_key == "AIzaSyNativeGeminiKey"
    
    assert emb_provider.__class__.__name__ == "OpenAIProvider"
    assert emb_provider.api_key == "sk-OpenAIKeyForEmbeddings"
    
    with pytest.raises(NotImplementedError, match="GoogleGenAIProvider does not provide embeddings"):
        gen_provider.embed_text("test chunk")


# ---------------------------------------------------------------------
# 16. Guest quota refund on generation failure
# ---------------------------------------------------------------------
def test_guest_quota_refund_on_native_generation_failure():
    from fastapi.testclient import TestClient
    from api.index import app
    
    app.state.initialized = True
    app.vectordb_client = MagicMock()
    app.generation_client = MagicMock(spec=GoogleGenAIProvider)
    app.generation_client.generate_text_async = AsyncMock(return_value=None)  # Fails generation
    app.generation_client.construct_prompt = lambda prompt, role: {"role": role, "content": prompt}
    app.generation_client.enums = GoogleGenAIEnums
    app.embedding_client = MagicMock()
    app.embedding_client.embed_text.return_value = [0.1] * 768
    app.template_parser = MagicMock()
    app.template_parser.get.return_value = "Prompt template"
    app.template_parser.set_language = MagicMock()
    app.db_client = None
    
    app.vectordb_client.search_by_vector.return_value = [
        RetrievedDocument(text="Giza pyramids details", score=0.9, document_id="1", metadata={"title": "Pyramids", "price": "20 USD"})
    ]

    client_ip = "192.168.1.100"
    guest_id = derive_guest_id(client_ip, DEV_SECRET)
    session_id = uuid.uuid4().hex
    token = generate_guest_token(guest_id, session_id, DEV_SECRET)
    
    # Initial quota is 5
    assert get_guest_remaining_questions(guest_id, None, False, "development") == 5

    client = TestClient(app)
    headers = {"x-forwarded-for": client_ip, "x-guest-session-token": token}
    
    with patch("helpers.guest_session.resolve_guest_secret", return_value=DEV_SECRET):
        res = client.post("/api/v1/nlp/chat", json={"text": "رشح لي أنشطة في الجيزة", "guest_token": token}, headers=headers)
        
        assert res.status_code == 503
        
        # Quota MUST be refunded back to 5
        rem = get_guest_remaining_questions(guest_id, None, False, "development")
        assert rem == 5, f"Expected 5 remaining after generation failure refund, got {rem}"


# ---------------------------------------------------------------------
# 17. Guest quota commit only after successful generation
# ---------------------------------------------------------------------
def test_guest_quota_commit_only_after_successful_generation():
    from fastapi.testclient import TestClient
    from api.index import app
    
    app.state.initialized = True
    app.vectordb_client = MagicMock()
    app.generation_client = MagicMock(spec=GoogleGenAIProvider)
    app.generation_client.generate_text_async = AsyncMock(return_value="إليك 3 أنشطة سياحية في الجيزة...")
    app.generation_client.construct_prompt = lambda prompt, role: {"role": role, "content": prompt}
    app.generation_client.enums = GoogleGenAIEnums
    app.embedding_client = MagicMock()
    app.embedding_client.embed_text.return_value = [0.1] * 768
    app.template_parser = MagicMock()
    app.template_parser.get.return_value = "Prompt template"
    app.template_parser.set_language = MagicMock()
    app.db_client = None
    
    app.vectordb_client.search_by_vector.return_value = [
        RetrievedDocument(text="Giza pyramids details", score=0.9, document_id="1", metadata={"title": "Pyramids", "price": "20 USD"})
    ]

    client_ip = "192.168.1.101"
    guest_id = derive_guest_id(client_ip, DEV_SECRET)
    session_id = uuid.uuid4().hex
    token = generate_guest_token(guest_id, session_id, DEV_SECRET)
    
    # Initial quota is 5
    assert get_guest_remaining_questions(guest_id, None, False, "development") == 5

    client = TestClient(app)
    headers = {"x-forwarded-for": client_ip, "x-guest-session-token": token}
    
    with patch("helpers.guest_session.resolve_guest_secret", return_value=DEV_SECRET):
        res = client.post("/api/v1/nlp/chat", json={"text": "رشح لي أنشطة في الجيزة", "guest_token": token}, headers=headers)
        
        assert res.status_code == 200
        data = res.json()
        assert "أنشطة" in data["answer"]
        
        # Quota MUST be decremented from 5 to 4
        rem = get_guest_remaining_questions(guest_id, None, False, "development")
        assert rem == 4, f"Expected 4 remaining after successful generation commit, got {rem}"
