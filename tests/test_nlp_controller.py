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
