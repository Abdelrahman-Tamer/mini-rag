from unittest.mock import MagicMock
import pytest
from controllers.NLPController import NLPController


@pytest.fixture
def nlp_controller():
    mock_vector = MagicMock()
    mock_gen = MagicMock()
    mock_embed = MagicMock()
    mock_template = MagicMock()
    return NLPController(
        vectordb_client=mock_vector,
        generation_client=mock_gen,
        embedding_client=mock_embed,
        template_parser=mock_template,
    )


def test_extract_city_arabic(nlp_controller):
    assert nlp_controller.extract_city("أفضل الأنشطة في القاهرة") == "Cairo"
    assert nlp_controller.extract_city("عايز اروح اسكندرية الأسبوع الجاي") == "Alexandria"
    assert nlp_controller.extract_city("رحلات الغردقة للبحر") == "Hurghada"
    assert nlp_controller.extract_city("معابد الأقصر الرائعة") == "Luxor"
    assert nlp_controller.extract_city("السد العالي في اسوان") == "Aswan"
    assert nlp_controller.extract_city("شرم الشيخ وسفاري الجبل") == "Sharm El Sheikh"
    assert nlp_controller.extract_city("أهرامات الجيزة") == "Giza"


def test_extract_city_english(nlp_controller):
    assert nlp_controller.extract_city("Best historical places in Cairo") == "Cairo"
    assert nlp_controller.extract_city("Day trips from Alexandria") == "Alexandria"
    assert nlp_controller.extract_city("Diving and snorkeling in Hurghada") == "Hurghada"
    assert nlp_controller.extract_city("Hot air balloon in Luxor") == "Luxor"
    assert nlp_controller.extract_city("Hotels in Giza near pyramids") == "Giza"
    assert nlp_controller.extract_city("General travel tips for Egypt") is None


def test_needs_query_rewriting_without_history(nlp_controller):
    # When history is empty, rewriting is completely unnecessary
    assert nlp_controller.needs_query_rewriting("How much does it cost?", []) is False
    assert nlp_controller.needs_query_rewriting("بكام ده؟", []) is False
    assert nlp_controller.needs_query_rewriting("Activities in Cairo", None) is False


def test_needs_query_rewriting_with_history(nlp_controller):
    history = [
        {"role": "user", "content": "Tell me about the Pyramids tour."},
        {"role": "assistant", "content": "The Pyramids tour is in Giza and costs 20 USD."},
    ]

    # Contextual references in English should trigger rewrite
    assert nlp_controller.needs_query_rewriting("how much does it cost?", history) is True
    assert nlp_controller.needs_query_rewriting("is it nearby?", history) is True
    assert nlp_controller.needs_query_rewriting("and what about Alexandria?", history) is True
    assert nlp_controller.needs_query_rewriting("add it to my schedule", history) is True

    # Contextual references in Arabic should trigger rewrite
    assert nlp_controller.needs_query_rewriting("بكام ده؟", history) is True
    assert nlp_controller.needs_query_rewriting("طب وبالنسبة لإسكندرية؟", history) is True
    assert nlp_controller.needs_query_rewriting("ضيفها في الجدول", history) is True

    # Standalone queries that are descriptive and independent should bypass rewrite
    standalone_query = "What are the top three archaeological museums in Cairo Egypt?"
    assert nlp_controller.needs_query_rewriting(standalone_query, history) is False


def test_is_greeting_query(nlp_controller):
    # Standard English greetings
    assert nlp_controller.is_greeting_query("hello") is True
    assert nlp_controller.is_greeting_query("hi") is True
    assert nlp_controller.is_greeting_query("hey") is True
    assert nlp_controller.is_greeting_query("good morning") is True
    assert nlp_controller.is_greeting_query("good morning!") is True
    assert nlp_controller.is_greeting_query("Hello!") is True
    assert nlp_controller.is_greeting_query("HELLO???") is True
    assert nlp_controller.is_greeting_query("hi there") is True
    assert nlp_controller.is_greeting_query("hello there") is True

    # Arabic greetings: base forms, tanween, tashkeel, hamza, punctuation
    assert nlp_controller.is_greeting_query("مرحبا") is True
    assert nlp_controller.is_greeting_query("مرحباً") is True
    assert nlp_controller.is_greeting_query("مَرْحَبًا") is True
    assert nlp_controller.is_greeting_query("مرحبا؟") is True
    assert nlp_controller.is_greeting_query("مرحبا!") is True
    assert nlp_controller.is_greeting_query("مرحبا،") is True
    assert nlp_controller.is_greeting_query("مرحبا بك") is True
    assert nlp_controller.is_greeting_query("مرحبا بكم") is True
    assert nlp_controller.is_greeting_query("مرحبا يا غالي") is True
    assert nlp_controller.is_greeting_query("اهلا") is True
    assert nlp_controller.is_greeting_query("أهلا") is True
    assert nlp_controller.is_greeting_query("أهلاً") is True
    assert nlp_controller.is_greeting_query("أهلاً وسهلاً") is True
    assert nlp_controller.is_greeting_query("اهلا وسهلا") is True
    assert nlp_controller.is_greeting_query("السلام عليكم") is True
    assert nlp_controller.is_greeting_query("السلام عليكم ورحمة الله") is True
    assert nlp_controller.is_greeting_query("السلام عليكم ورحمة الله وبركاته") is True
    assert nlp_controller.is_greeting_query("صباح الخير") is True
    assert nlp_controller.is_greeting_query("مساء الخير") is True
    assert nlp_controller.is_greeting_query("يا هلا") is True
    assert nlp_controller.is_greeting_query("اهلين") is True

    # Bidirectional and Zero-Width Unicode characters
    assert nlp_controller.is_greeting_query("\u200fمرحبا\u200f") is True  # RLM
    assert nlp_controller.is_greeting_query("\u200eمرحبا\u200e") is True  # LRM
    assert nlp_controller.is_greeting_query("\ufeffمرحبا") is True          # BOM
    assert nlp_controller.is_greeting_query("\u200bمرحبا\u200b") is True  # ZWSP

    # Non-greeting queries MUST return False to allow search & RAG
    assert nlp_controller.is_greeting_query("What are the best tours in Cairo?") is False
    assert nlp_controller.is_greeting_query("أفضل الأنشطة في مصر") is False
    assert nlp_controller.is_greeting_query("مرحبا، أين يقع المتحف المصري؟") is False
    assert nlp_controller.is_greeting_query("السلام عليكم أريد حجز فندق في القاهرة") is False


def test_sync_activity_upsert(nlp_controller):
    activity_data = {
        "activity_id": 99,
        "title": "Nile River Felucca Ride",
        "city": "Cairo",
        "category": "Water",
        "price": "15 USD",
        "description": "Scenic sunset sailing on the Nile.",
        "image_url": "https://example.com/felucca.jpg",
        "link": "https://www.nileguide.online/activities/99",
        "is_active": True,
    }

    nlp_controller.embedding_client.embed_text.return_value = [0.1] * 768

    result = nlp_controller.sync_activity(activity_data)
    assert result == {"status": "upserted", "activity_id": 99}
    nlp_controller.embedding_client.embed_text.assert_called_once()
    nlp_controller.vectordb_client.insert_one.assert_called_once()


def test_delete_activity(nlp_controller):
    result = nlp_controller.delete_activity(activity_id=99)
    assert result == {"status": "deleted", "activity_id": 99}
    nlp_controller.vectordb_client.delete_by_filter.assert_called_once_with(
        collection_name=nlp_controller.collection_name,
        filter={"activity_id": 99},
    )


def test_greeting_endpoint_bypasses_llm_and_vector_search():
    from fastapi.testclient import TestClient
    from api.index import app
    from unittest.mock import MagicMock

    # Setup mock app state
    app.state.initialized = True
    app.vectordb_client = MagicMock()
    app.generation_client = MagicMock()
    app.embedding_client = MagicMock()
    app.template_parser = MagicMock()
    app.db_client = None

    client = TestClient(app)

    # 1. Arabic greeting with tanween (was failing before fix)
    res_ar_tanween = client.post("/api/v1/nlp/chat", json={"query": "مرحباً"})
    assert res_ar_tanween.status_code == 200
    data = res_ar_tanween.json()
    assert data["signal"] == "chat_success"
    assert "NileGuide" in data["message"]
    assert "أهلاً بك" in data["message"]
    # Verify LLM generation and embedding were NEVER called
    app.generation_client.generate_text.assert_not_called()
    app.embedding_client.embed_text.assert_not_called()

    # 2. Arabic greeting with hamza
    res_ar_hamza = client.post("/api/v1/nlp/chat", json={"text": "أهلاً وسهلاً"})
    assert res_ar_hamza.status_code == 200
    assert res_ar_hamza.json()["signal"] == "chat_success"
    app.generation_client.generate_text.assert_not_called()
    app.embedding_client.embed_text.assert_not_called()

    # 3. Arabic greeting with punctuation
    res_ar_punct = client.post("/api/v1/nlp/chat", json={"text": "مرحبا؟!"})
    assert res_ar_punct.status_code == 200
    assert res_ar_punct.json()["signal"] == "chat_success"
    app.generation_client.generate_text.assert_not_called()
    app.embedding_client.embed_text.assert_not_called()

    # 4. English greeting
    res_en = client.post("/api/v1/nlp/chat", json={"text": "Hello there!"})
    assert res_en.status_code == 200
    data_en = res_en.json()
    assert data_en["signal"] == "chat_success"
    assert "welcome to NileGuide" in data_en["message"]
    app.generation_client.generate_text.assert_not_called()
    app.embedding_client.embed_text.assert_not_called()


def test_non_greeting_tourist_question_invokes_full_pipeline():
    from fastapi.testclient import TestClient
    from api.index import app
    from unittest.mock import MagicMock
    from models.db_schemes import RetrievedDocument

    app.state.initialized = True
    app.vectordb_client = MagicMock()
    app.generation_client = MagicMock()
    app.embedding_client = MagicMock()
    app.template_parser = MagicMock()
    app.db_client = None

    app.embedding_client.embed_text.return_value = [0.1] * 768
    app.vectordb_client.search_by_vector.return_value = [
        RetrievedDocument(text="The Giza Pyramids are located on the Giza Plateau near Cairo.", score=0.92, document_id="1", metadata={"title": "Pyramids"})
    ]
    app.template_parser.get.return_value = "Mock template content"
    app.generation_client.generate_text.return_value = "تقع أهرامات الجيزة في هضبة الجيزة قرب القاهرة."
    app.generation_client.construct_prompt.side_effect = lambda prompt, role: {"role": role, "content": prompt}
    app.generation_client.enums.USER.value = "user"
    app.generation_client.enums.SYSTEM.value = "system"
    app.generation_client.enums.ASSISTANT.value = "assistant"

    client = TestClient(app)
    res = client.post("/api/v1/nlp/chat", json={"query": "أين تقع أهرامات الجيزة؟"})

    assert res.status_code == 200
    data = res.json()
    assert data["signal"] == "chat_success"
    assert app.embedding_client.embed_text.called
    assert app.vectordb_client.search_by_vector.called
    assert app.generation_client.generate_text.called


def test_openai_provider_generate_text_exception_handling():
    from api.stores.llm.providers.OpenAIProvider import OpenAIProvider
    from unittest.mock import MagicMock
    from openai import NotFoundError

    provider = OpenAIProvider(api_key="test-key", api_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    provider.set_generation_model("gemini-3.8-flash")
    provider.client.chat.completions.create = MagicMock(
        side_effect=NotFoundError("Model 'gemini-3.8-flash' not found", response=MagicMock(status_code=404), body={})
    )

    # Must return None and log error instead of crashing
    result = provider.generate_text("test prompt")
    assert result is None


def test_openai_provider_primary_success():
    from api.stores.llm.providers.OpenAIProvider import OpenAIProvider
    from unittest.mock import MagicMock

    provider = OpenAIProvider(api_key="test-key", api_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    provider.set_generation_model("gemini-3.5-flash-lite")

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Primary model response"))]
    provider.client.chat.completions.create = MagicMock(return_value=mock_resp)

    result = provider.generate_text("test prompt")
    assert result == "Primary model response"
    assert provider.client.chat.completions.create.call_count == 1
    call_args = provider.client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gemini-3.5-flash-lite"


def test_openai_provider_transient_failure_fallback_success():
    from api.stores.llm.providers.OpenAIProvider import OpenAIProvider
    from unittest.mock import MagicMock
    from openai import APIStatusError

    provider = OpenAIProvider(api_key="test-key", api_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    provider.set_generation_model("gemini-3.5-flash-lite")

    mock_fallback_resp = MagicMock()
    mock_fallback_resp.choices = [MagicMock(message=MagicMock(content="Verified fallback response"))]

    def mock_create(model, **kwargs):
        if model == "gemini-3.5-flash-lite":
            raise APIStatusError("503 High Demand", response=MagicMock(status_code=503), body={})
        return mock_fallback_resp

    provider.client.chat.completions.create = MagicMock(side_effect=mock_create)

    result = provider.generate_text("test prompt")
    assert result == "Verified fallback response"
    assert provider.client.chat.completions.create.call_count == 2
    first_call_model = provider.client.chat.completions.create.call_args_list[0][1]["model"]
    second_call_model = provider.client.chat.completions.create.call_args_list[1][1]["model"]
    assert first_call_model == "gemini-3.5-flash-lite"
    assert second_call_model == "gemini-3.5-flash"


def test_openai_provider_timeout_fallback_success():
    from api.stores.llm.providers.OpenAIProvider import OpenAIProvider
    from unittest.mock import MagicMock
    from openai import APITimeoutError

    provider = OpenAIProvider(api_key="test-key", api_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    provider.set_generation_model("gemini-3.5-flash-lite")

    mock_fallback_resp = MagicMock()
    mock_fallback_resp.choices = [MagicMock(message=MagicMock(content="Fallback response after timeout"))]

    def mock_create(model, **kwargs):
        if model == "gemini-3.5-flash-lite":
            raise APITimeoutError(request=MagicMock())
        return mock_fallback_resp

    provider.client.chat.completions.create = MagicMock(side_effect=mock_create)

    result = provider.generate_text("test prompt")
    assert result == "Fallback response after timeout"
    assert provider.client.chat.completions.create.call_count == 2


def test_openai_provider_non_retryable_client_error_no_fallback():
    from api.stores.llm.providers.OpenAIProvider import OpenAIProvider
    from unittest.mock import MagicMock
    from openai import AuthenticationError, BadRequestError, PermissionDeniedError

    provider = OpenAIProvider(api_key="test-key", api_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    provider.set_generation_model("gemini-3.5-flash-lite")

    # 400 BadRequest / INVALID_ARGUMENT must NOT call fallback
    provider.client.chat.completions.create = MagicMock(
        side_effect=BadRequestError("Please pass a valid API key (INVALID_ARGUMENT)", response=MagicMock(status_code=400), body={})
    )
    result = provider.generate_text("test prompt")
    assert result is None
    assert provider.client.chat.completions.create.call_count == 1

    # 401 AuthenticationError must NOT call fallback
    provider.client.chat.completions.create = MagicMock(
        side_effect=AuthenticationError("Invalid API Key", response=MagicMock(status_code=401), body={})
    )
    result = provider.generate_text("test prompt")
    assert result is None
    assert provider.client.chat.completions.create.call_count == 1

    # 403 PermissionDeniedError must NOT call fallback
    provider.client.chat.completions.create = MagicMock(
        side_effect=PermissionDeniedError("Project billing inactive", response=MagicMock(status_code=403), body={})
    )
    result = provider.generate_text("test prompt")
    assert result is None
    assert provider.client.chat.completions.create.call_count == 1


def test_openai_provider_both_models_unavailable():
    from api.stores.llm.providers.OpenAIProvider import OpenAIProvider
    from unittest.mock import MagicMock
    from openai import APIStatusError

    provider = OpenAIProvider(api_key="test-key", api_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    provider.set_generation_model("gemini-3.5-flash-lite")

    provider.client.chat.completions.create = MagicMock(
        side_effect=APIStatusError("High demand 503", response=MagicMock(status_code=503), body={})
    )

    result = provider.generate_text("test prompt")
    assert result is None
    assert provider.client.chat.completions.create.call_count == 2


def test_openai_provider_sanitized_error_logging():
    from api.stores.llm.providers.OpenAIProvider import _sanitize_error_message

    raw_err = "Failed with key AIzaSyTestKey1234567890abcdef and token sk-1234567890abcdefghijklmn and Authorization: Bearer secret-token-xyz"
    sanitized = _sanitize_error_message(raw_err)
    assert "AIzaSy" not in sanitized
    assert "sk-" not in sanitized
    assert "secret-token-xyz" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert "[REDACTED_TOKEN]" in sanitized


def test_provider_failure_graceful_fallback_arabic():
    from fastapi.testclient import TestClient
    from api.index import app
    from unittest.mock import MagicMock
    from helpers.cache import global_response_cache

    global_response_cache.clear()

    app.state.initialized = True
    app.vectordb_client = MagicMock()
    app.generation_client = MagicMock()
    app.embedding_client = MagicMock()
    app.template_parser = MagicMock()
    app.db_client = None

    # Simulate empty search results and provider error returning None
    app.embedding_client.embed_text.return_value = [0.1] * 768
    app.vectordb_client.search_by_vector.return_value = []
    app.template_parser.get.return_value = "Mock prompt"
    app.generation_client.generate_text.return_value = None
    app.generation_client.construct_prompt.side_effect = lambda prompt, role: {"role": role, "content": prompt}
    app.generation_client.enums.USER.value = "user"
    app.generation_client.enums.SYSTEM.value = "system"
    app.generation_client.enums.ASSISTANT.value = "assistant"

    client = TestClient(app)
    res = client.post("/api/v1/nlp/chat", json={"query": "ما هي أفضل أوقات زيارة وادي الملوك؟"})

    assert res.status_code == 503
    data = res.json()
    assert data["signal"] == "chat_error"
    assert data["error"] == "ai_generation_unavailable"
    assert "error_id" in data
    # Localized Arabic fallback message is provided
    assert "غير متاحة" in data["message"]


def test_provider_failure_graceful_fallback_english():
    from fastapi.testclient import TestClient
    from api.index import app
    from unittest.mock import MagicMock
    from helpers.cache import global_response_cache

    global_response_cache.clear()

    app.state.initialized = True
    app.vectordb_client = MagicMock()
    app.generation_client = MagicMock()
    app.embedding_client = MagicMock()
    app.template_parser = MagicMock()
    app.db_client = None

    # Simulate empty search results and provider error returning None
    app.embedding_client.embed_text.return_value = [0.1] * 768
    app.vectordb_client.search_by_vector.return_value = []
    app.template_parser.get.return_value = "Mock prompt"
    app.generation_client.generate_text.return_value = None
    app.generation_client.construct_prompt.side_effect = lambda prompt, role: {"role": role, "content": prompt}
    app.generation_client.enums.USER.value = "user"
    app.generation_client.enums.SYSTEM.value = "system"
    app.generation_client.enums.ASSISTANT.value = "assistant"

    client = TestClient(app)
    res = client.post("/api/v1/nlp/chat", json={"query": "What is the best time to visit Karnak Temple?"})

    assert res.status_code == 503
    data = res.json()
    assert data["signal"] == "chat_error"
    assert data["error"] == "ai_generation_unavailable"
    assert "error_id" in data
    # Localized English fallback message is provided
    assert "temporarily unavailable" in data["message"]
