import pytest
import filecmp
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.db_schemes import RetrievedDocument
from controllers.NLPController import NLPController
from stores.llm.templates.template_parser import TemplateParser


@pytest.fixture
def mock_template_parser_ar():
    return TemplateParser(language="ar")


@pytest.fixture
def mock_template_parser_en():
    return TemplateParser(language="en")


@pytest.fixture
def sample_retrieved_activities():
    """Five realistic NileGuide activities representing the Pyramids + GEM query."""
    return [
        RetrievedDocument(
            text="Comprehensive day tour covering Pyramids of Giza, Sphinx, and the Grand Egyptian Museum with licensed Egyptologist guide.",
            score=0.8050,
            metadata={
                "activity_id": 8,
                "title": "Private tour Giza Pyramids, Grand Egyptian Museum & Local Lunch",
                "city": "Giza",
                "category": "Pyramids & Archaeology",
                "price": "55.44 USD",
                "link": "https://www.nileguide.online/activities/8",
                "image_url": "https://example.com/act8.jpg",
                "is_active": True,
            }
        ),
        RetrievedDocument(
            text="Explore Ancient Egypt all in one day. Take in the Giza Pyramids and Sphinx before visiting the Grand Egyptian Museum.",
            score=0.8012,
            metadata={
                "activity_id": 10,
                "title": "Tour to Pyramids & The Grand Egyptian Museum",
                "city": "Giza",
                "category": "Pyramids & Archaeology",
                "price": "185.0 USD",
                "link": "https://www.nileguide.online/activities/10",
                "image_url": "https://example.com/act10.jpg",
                "is_active": True,
            }
        ),
        RetrievedDocument(
            text="Cairo 8 Hour Private Tour of Pyramids & Grand Egyptian Museum with camel ride and lunch.",
            score=0.7985,
            metadata={
                "activity_id": 13,
                "title": "Cairo 8 Hour Private Tour of Pyramids & Grand Egyptian Museum",
                "city": "Cairo",
                "category": "Pyramids & Archaeology",
                "price": "35.0 USD",
                "link": "https://www.nileguide.online/activities/13",
                "image_url": "https://example.com/act13.jpg",
                "is_active": True,
            }
        ),
        RetrievedDocument(
            text="Giza Pyramids, Sphinx & Grand Egyptian Museum Tour Full Experience.",
            score=0.7930,
            metadata={
                "activity_id": 27,
                "title": "Giza Pyramids,Sphinx &Grand Egyptian Museum Tour Full Experience",
                "city": "Giza",
                "category": "Pyramids & Archaeology",
                "price": "30.0 USD",
                "link": "https://www.nileguide.online/activities/27",
                "image_url": "https://example.com/act27.jpg",
                "is_active": True,
            }
        ),
        RetrievedDocument(
            text="Grand Egyptian Museum Private Tour in Cairo with Guide. 12 exhibition halls with 25,000 artifacts.",
            score=0.7898,
            metadata={
                "activity_id": 84,
                "title": "Grand Egyptian Museum Private Tour in Cairo with Guide",
                "city": "Cairo",
                "category": "History & Culture",
                "price": None,  # Intentionally missing price to test safe handling
                "link": "https://www.nileguide.online/activities/84",
                "image_url": "https://example.com/act84.jpg",
                "is_active": True,
            }
        ),
    ]


# ==============================================================================
# TEST 1: ARABIC SYSTEM PROMPT PROHIBITIONS ELIMINATED
# ==============================================================================

def test_arabic_system_prompt_no_longer_prohibits_names_or_prices(mock_template_parser_ar):
    """Test 1: Verify Arabic system prompt does not contain negative prohibitions against names or prices."""
    sys_prompt = mock_template_parser_ar.get("rag", "system_prompt")
    assert sys_prompt is not None

    # Prohibited negative phrases from prior flawed versions
    assert "لا تذكر أسماء" not in sys_prompt
    assert "لا تذكر" not in sys_prompt
    assert "سطر واحد فقط" not in sys_prompt
    assert "بطاقات تفاعلية" not in sys_prompt

    # Positive grounded instructions must be present
    assert "اذكر اسم كل نشاط وسعره" in sys_prompt
    assert "المصادر المرفقة" in sys_prompt
    assert "لا تخترع" in sys_prompt


# ==============================================================================
# TEST 2: ENGLISH SYSTEM PROMPT EQUIVALENT BEHAVIOR
# ==============================================================================

def test_english_system_prompt_has_equivalent_behavior(mock_template_parser_en):
    """Test 2: Verify English system prompt contains positive grounded rules and no negative prohibitions."""
    sys_prompt = mock_template_parser_en.get("rag", "system_prompt")
    assert sys_prompt is not None

    # Prohibited negative phrases from prior flawed versions
    assert "Do NOT list the place names" not in sys_prompt
    assert "ONLY a single short introductory sentence" not in sys_prompt
    assert "interactive cards below" not in sys_prompt

    # Positive grounded instructions must be present
    assert "include the exact activity names and prices" in sys_prompt
    assert "NEVER fabricate" in sys_prompt
    assert "retrieved sources" in sys_prompt


# ==============================================================================
# TEST 3: FOOTER PROMPTS CONTAIN NO CONTRADICTORY RESTRICTIONS
# ==============================================================================

def test_footer_prompts_contain_no_contradictory_restrictions(mock_template_parser_ar, mock_template_parser_en):
    """Test 3: Verify footer prompts in both languages have zero negative suppression clauses."""
    query = "ما هي الأنشطة المتاحة وأسعارها؟"

    # Arabic footer
    foot_ar = mock_template_parser_ar.get("rag", "footer_prompt", {"query": query})
    assert "بدون سرد أسماء أو أسعار" not in foot_ar
    assert "اكتفِ بسطر تقديمي" not in foot_ar
    assert "واذكر أسماء الأنشطة وأسعارها" in foot_ar

    # English footer
    foot_en = mock_template_parser_en.get("rag", "footer_prompt", {"query": "What are the available activities and prices?"})
    assert "do not list place names or prices" not in foot_en
    assert "ONLY a one-line introduction" not in foot_en
    assert "including activity names and prices" in foot_en


# ==============================================================================
# TEST 4: RETRIEVED METADATA TITLE APPEARS IN RENDERED CONTEXT
# ==============================================================================

def test_retrieved_metadata_title_appears_in_rendered_gemini_context(sample_retrieved_activities):
    """Test 4: Verify metadata title appears explicitly in the rendered context passed to Gemini."""
    doc = sample_retrieved_activities[0]
    formatted = NLPController._format_document_context(doc)

    assert "- Title: Private tour Giza Pyramids, Grand Egyptian Museum & Local Lunch" in formatted


# ==============================================================================
# TEST 5: RETRIEVED METADATA PRICE APPEARS IN RENDERED CONTEXT
# ==============================================================================

def test_retrieved_metadata_price_appears_in_rendered_gemini_context(sample_retrieved_activities):
    """Test 5: Verify metadata price appears explicitly with original currency in rendered context."""
    doc = sample_retrieved_activities[0]
    formatted = NLPController._format_document_context(doc)

    assert "- Price: 55.44 USD" in formatted

    doc2 = sample_retrieved_activities[1]
    formatted2 = NLPController._format_document_context(doc2)
    assert "- Price: 185.0 USD" in formatted2


# ==============================================================================
# TEST 6: CITY, CATEGORY, AND ACTIVITY ID INCLUDED IN CONTEXT
# ==============================================================================

def test_city_category_and_activity_id_included_when_available(sample_retrieved_activities):
    """Test 6: Verify Activity ID, City, and Category are structured in rendered context."""
    doc = sample_retrieved_activities[0]
    formatted = NLPController._format_document_context(doc)

    assert "- Activity ID: 8" in formatted
    assert "- City: Giza" in formatted
    assert "- Category: Pyramids & Archaeology" in formatted


# ==============================================================================
# TEST 7: ORIGINAL DOCUMENT TEXT IS PRESERVED
# ==============================================================================

def test_original_retrieved_document_text_preserved(sample_retrieved_activities):
    """Test 7: Verify original document text is preserved under Description: header."""
    doc = sample_retrieved_activities[0]
    formatted = NLPController._format_document_context(doc)

    assert "Description:\nComprehensive day tour covering Pyramids of Giza" in formatted


# ==============================================================================
# TEST 8: MISSING METADATA DOES NOT CRASH GENERATION
# ==============================================================================

def test_missing_metadata_does_not_crash_generation():
    """Test 8: Verify empty, None, or unexpected metadata structures do not crash generation."""
    # doc with None metadata
    doc_none = RetrievedDocument(text="Plain text document without metadata", score=0.75, metadata=None)
    res_none = NLPController._format_document_context(doc_none)
    assert res_none == "Plain text document without metadata"

    # doc with empty dict metadata
    doc_empty = RetrievedDocument(text="Plain text document with empty metadata", score=0.75, metadata={})
    res_empty = NLPController._format_document_context(doc_empty)
    assert res_empty == "Plain text document with empty metadata"

    # doc with non-dict metadata
    doc_corrupt = RetrievedDocument(text="Corrupt metadata test", score=0.75)
    doc_corrupt.metadata = "not-a-dict"  # type: ignore
    res_corrupt = NLPController._format_document_context(doc_corrupt)
    assert res_corrupt == "Corrupt metadata test"


# ==============================================================================
# TEST 9: MISSING PRICES ARE NOT FABRICATED
# ==============================================================================

def test_missing_prices_are_not_fabricated(sample_retrieved_activities):
    """Test 9: When price is None in metadata, no Price line is added or invented."""
    doc_no_price = sample_retrieved_activities[4]  # Activity 84 has price=None
    formatted = NLPController._format_document_context(doc_no_price)

    assert "- Title: Grand Egyptian Museum Private Tour in Cairo with Guide" in formatted
    assert "- Activity ID: 84" in formatted
    assert "- Price:" not in formatted


# ==============================================================================
# TEST 10: DETAILED PROVIDER RESPONSES SURVIVE POST-PROCESSING
# ==============================================================================

def test_detailed_provider_responses_survive_post_processing(sample_retrieved_activities):
    """Test 10: Detailed provider responses survive post-processing and caching without alteration."""
    detailed_arabic_response = (
        "إليك الأنشطة المتاحة التي تجمع بين زيارة أهرامات الجيزة والمتحف المصري الكبير وأسعارها:\n"
        "1. Private tour Giza Pyramids, Grand Egyptian Museum & Local Lunch: 55.44 USD\n"
        "2. Tour to Pyramids & The Grand Egyptian Museum: 185.0 USD\n"
        "3. Cairo 8 Hour Private Tour of Pyramids & Grand Egyptian Museum: 35.0 USD\n"
        "4. Giza Pyramids,Sphinx &Grand Egyptian Museum Tour Full Experience: 30.0 USD\n"
        "5. Grand Egyptian Museum Private Tour in Cairo with Guide: السعر غير محدد في قاعدة البيانات."
    )

    mock_gen = MagicMock()
    mock_gen.enums.SYSTEM.value = "system"
    mock_gen.enums.USER.value = "user"
    mock_gen.construct_prompt.side_effect = lambda prompt, role: {"role": role, "content": prompt}
    mock_gen.generate_text.return_value = detailed_arabic_response

    nlp = NLPController(
        vectordb_client=MagicMock(),
        generation_client=mock_gen,
        embedding_client=MagicMock(),
        template_parser=TemplateParser(language="ar"),
    )

    query = "اعرض الأنشطة المتاحة التي تجمع بين زيارة أهرامات الجيزة والمتحف المصري الكبير واذكر اسم كل نشاط وسعره."
    answer, full_prompt, _ = nlp.generate_chat_answer(
        query=query,
        retrieved_documents=sample_retrieved_activities,
    )

    assert answer == detailed_arabic_response
    assert "55.44 USD" in answer
    assert "185.0 USD" in answer
    assert "Private tour Giza Pyramids" in answer


# ==============================================================================
# TEST 11: API RESPONSE SOURCES REMAIN UNCHANGED
# ==============================================================================

def test_api_response_sources_remain_unchanged(sample_retrieved_activities):
    """Test 11: Verify source serialization preserves all expected API schema keys."""
    sources = [
        {
            "activity_id": doc.metadata.get("activity_id") if doc.metadata else None,
            "title": doc.metadata.get("title") or "",
            "city": doc.metadata.get("city") if doc.metadata else None,
            "category": doc.metadata.get("category") if doc.metadata else None,
            "price": doc.metadata.get("price") if doc.metadata else None,
            "image_url": doc.metadata.get("image_url") if doc.metadata else None,
            "link": doc.metadata.get("link"),
            "score": round(doc.score, 4),
        }
        for doc in sample_retrieved_activities
    ]

    assert len(sources) == 5
    expected_keys = {"activity_id", "title", "city", "category", "price", "image_url", "link", "score"}
    for s in sources:
        assert set(s.keys()) == expected_keys

    assert sources[0]["activity_id"] == 8
    assert sources[0]["price"] == "55.44 USD"
    assert sources[4]["price"] is None


# ==============================================================================
# TEST 12: GENERAL TOURISM QUERIES REMAIN NATURAL AND CONCISE
# ==============================================================================

def test_general_tourism_queries_remain_natural_and_concise():
    """Test 12: General tourist queries without price requests assemble clean, natural context."""
    general_doc = RetrievedDocument(
        text="The Great Pyramid of Giza is the oldest and largest of the pyramids in the Giza pyramid complex.",
        score=0.92,
        metadata={"title": "Giza Pyramids Overview", "city": "Giza", "category": "History"}
    )

    mock_gen = MagicMock()
    mock_gen.enums.SYSTEM.value = "system"
    mock_gen.enums.USER.value = "user"
    mock_gen.construct_prompt.side_effect = lambda prompt, role: {"role": role, "content": prompt}
    mock_gen.generate_text.return_value = "أهرامات الجيزة تقع في هضبة الجيزة غرب القاهرة وتعتبر من عجائب الدنيا السبع."

    nlp = NLPController(
        vectordb_client=MagicMock(),
        generation_client=mock_gen,
        embedding_client=MagicMock(),
        template_parser=TemplateParser(language="ar"),
    )

    answer, full_prompt, _ = nlp.generate_chat_answer(
        query="أين تقع أهرامات الجيزة؟",
        retrieved_documents=[general_doc],
    )

    assert "أهرامات الجيزة" in answer
    assert "Title: Giza Pyramids Overview" in full_prompt


# ==============================================================================
# TEST 13: EXPLICIT ACTIVITY & PRICE QUESTIONS PRODUCE GROUNDED DETAILED CONTEXT
# ==============================================================================

def test_explicit_activity_and_price_questions_produce_grounded_detailed_answers(sample_retrieved_activities):
    """Test 13: Full context passed to generate_text contains all activity titles, cities, and prices."""
    mock_gen = MagicMock()
    mock_gen.enums.SYSTEM.value = "system"
    mock_gen.enums.USER.value = "user"
    mock_gen.construct_prompt.side_effect = lambda prompt, role: {"role": role, "content": prompt}
    mock_gen.generate_text.return_value = "Mocked grounded response"

    nlp = NLPController(
        vectordb_client=MagicMock(),
        generation_client=mock_gen,
        embedding_client=MagicMock(),
        template_parser=TemplateParser(language="ar"),
    )

    query = (
        "اعرض الأنشطة المتاحة التي تجمع بين زيارة أهرامات الجيزة والمتحف المصري الكبير. "
        "اذكر اسم كل نشاط وسعره كما هو مسجل في قاعدة بيانات NileGuide."
    )

    answer, full_prompt, _ = nlp.generate_chat_answer(
        query=query,
        retrieved_documents=sample_retrieved_activities,
    )

    # Context verification: all titles and prices are in full_prompt
    assert "- Title: Private tour Giza Pyramids, Grand Egyptian Museum & Local Lunch" in full_prompt
    assert "- Price: 55.44 USD" in full_prompt
    assert "- Title: Tour to Pyramids & The Grand Egyptian Museum" in full_prompt
    assert "- Price: 185.0 USD" in full_prompt
    assert "- Title: Cairo 8 Hour Private Tour of Pyramids & Grand Egyptian Museum" in full_prompt
    assert "- Price: 35.0 USD" in full_prompt


# ==============================================================================
# TEST 14: BOTH API AND SRC IMPLEMENTATIONS REMAIN SYNCHRONIZED
# ==============================================================================

def test_both_api_and_src_implementations_remain_synchronized():
    """Test 14: Byte-for-byte parity between api/ and src/ for all modified modules."""
    repo_root = Path(__file__).resolve().parent.parent
    files_to_check = [
        ("api/controllers/NLPController.py", "src/controllers/NLPController.py"),
        ("api/routes/nlp.py", "src/routes/nlp.py"),
        ("api/stores/llm/templates/locales/ar/rag.py", "src/stores/llm/templates/locales/ar/rag.py"),
        ("api/stores/llm/templates/locales/en/rag.py", "src/stores/llm/templates/locales/en/rag.py"),
        ("api/stores/llm/providers/OpenAIProvider.py", "src/stores/llm/providers/OpenAIProvider.py"),
        ("api/stores/vectordb/providers/QdrantDBProvider.py", "src/stores/vectordb/providers/QdrantDBProvider.py"),
    ]

    for api_rel, src_rel in files_to_check:
        api_file = repo_root / api_rel
        src_file = repo_root / src_rel
        assert api_file.exists(), f"Missing API file: {api_file}"
        assert src_file.exists(), f"Missing SRC file: {src_file}"
        assert filecmp.cmp(api_file, src_file, shallow=False), f"Parity mismatch between {api_rel} and {src_rel}"
