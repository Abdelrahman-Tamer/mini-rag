import pytest
import filecmp
from unittest.mock import MagicMock
from api.controllers.NLPController import NLPController


class MockDocument:
    def __init__(self, doc_id: int, title: str, price: any, score: float, city: str = "Giza", category: str = "Tours"):
        self.doc_id = doc_id
        self.text = f"Activity: {title}\nPrice: {price}\nCity: {city}"
        self.metadata = {
            "activity_id": doc_id,
            "title": title,
            "price": price,
            "city": city,
            "category": category,
        }
        self.score = score

    def __repr__(self):
        return f"<MockDoc {self.doc_id} {self.metadata['title']} price={self.metadata['price']} score={self.score}>"


def test_scenario_1_ascending_price_ranking():
    """
    Scenario 1: Scenario 3 from audit
    User asks for cheapest 3 activities.
    Vector DB returns 5 activities ($15, $35, $20, $25, $40).
    Backend must deterministically rank them ascending ($15, $20, $25)
    and include Act 58 ($20) which was previously omitted.
    """
    docs = [
        MockDocument(28, "جولة الأهرامات والمتحف السريعة", "15.0 USD", 0.805),
        MockDocument(65, "جولة VIP للأهرامات والمتحف", "35.0 USD", 0.801),
        MockDocument(58, "جولة استكشافية للأهرامات والمتحف", "20.0 USD", 0.798),
        MockDocument(53, "رحلة اليوم الكامل أهرامات ومتحف", "25.0 USD", 0.792),
        MockDocument(43, "جولة مسائية بانورامية", "40.0 USD", 0.789),
    ]

    query = "اعرض أرخص 3 أنشطة تجمع بين الأهرامات والمتحف المصري الكبير"
    ranked = NLPController.apply_price_ranking_if_requested(query, docs, default_limit=5)

    assert len(ranked) == 3
    assert [d.doc_id for d in ranked] == [28, 58, 53]
    prices = [NLPController.extract_numeric_price(d)[0] for d in ranked]
    assert prices == [15.0, 20.0, 25.0]


def test_scenario_2_descending_price_ranking():
    """
    Scenario 2: Descending price ranking (most expensive).
    User asks for most expensive 2 activities.
    Candidates ($15, $35, $20, $25, $40) -> Top 2: $40, $35.
    """
    docs = [
        MockDocument(28, "جولة سريعة", "15.0 USD", 0.805),
        MockDocument(65, "جولة VIP", "35.0 USD", 0.801),
        MockDocument(58, "جولة استكشافية", "20.0 USD", 0.798),
        MockDocument(53, "رحلة اليوم الكامل", "25.0 USD", 0.792),
        MockDocument(43, "جولة مسائية", "40.0 USD", 0.789),
    ]

    query = "اعرض أغلى نشاطين من النتائج المتاحة لزيارة الأهرامات والمتحف"
    ranked = NLPController.apply_price_ranking_if_requested(query, docs, default_limit=5)

    assert len(ranked) == 2
    assert [d.doc_id for d in ranked] == [43, 65]
    prices = [NLPController.extract_numeric_price(d)[0] for d in ranked]
    assert prices == [40.0, 35.0]


def test_scenario_3_count_preservation_and_insufficient_candidates():
    """
    Scenario 3: Requested count N is respected when >= N valid candidates exist,
    and never padded/hallucinated when fewer candidates exist.
    """
    docs = [
        MockDocument(1, "نشاط 1", "10 USD", 0.9),
        MockDocument(2, "نشاط 2", "20 USD", 0.85),
    ]

    # User asks for 3, but only 2 exist
    query = "أرخص 3 أنشطة"
    ranked = NLPController.apply_price_ranking_if_requested(query, docs, default_limit=5)
    assert len(ranked) == 2
    assert [d.doc_id for d in ranked] == [1, 2]

    # User asks for 1 (singular 'أرخص نشاط')
    query_single = "أرخص نشاط متاح"
    ranked_single = NLPController.apply_price_ranking_if_requested(query_single, docs, default_limit=5)
    assert len(ranked_single) == 1
    assert ranked_single[0].doc_id == 1


def test_scenario_4_missing_and_invalid_price_handling():
    """
    Scenario 4: Distinguish explicit zero-price / genuine free activities
    from unconfirmed / missing prices.
    - Confirmed free activities ('Free', '0.0', 'مجاني') rank as cheapest (0.0).
    - Missing / unconfirmed prices (None, 'Contact us', 'غير محدد') are excluded.
    """
    docs = [
        MockDocument(1, "نشاط بسعر عادي", "30.0 USD", 0.9),
        MockDocument(2, "نشاط بدون سعر", None, 0.85),
        MockDocument(3, "نشاط مجاني مؤكد", "Free", 0.82),
        MockDocument(4, "نشاط مخفض", "10.0 USD", 0.80),
        MockDocument(5, "نشاط بسعر غير محدد", "Contact us", 0.75),
    ]

    # User asks for cheapest 2 activities:
    # Ranked ascending: Doc 3 (Free / 0.0 USD) first, Doc 4 (10.0 USD) second.
    # Doc 1 (30.0 USD) is 3rd; Doc 2 (None) and Doc 5 (Contact us) are excluded.
    query = "اعرض أرخص نشاطين"
    ranked = NLPController.apply_price_ranking_if_requested(query, docs, default_limit=5)

    assert len(ranked) == 2
    assert [d.doc_id for d in ranked] == [3, 4]
    assert [NLPController.extract_numeric_price(d)[0] for d in ranked] == [0.0, 10.0]


def test_scenario_5_mixed_currency_safety():
    """
    Scenario 5: Mixed currencies guardrail.
    If candidates have conflicting currencies without a trusted conversion rate,
    never present raw vector similarity order as cheapest-to-most-expensive.
    Group candidates by currency and sort deterministically within each currency group.
    """
    docs = [
        MockDocument(1, "جولة بالدولار A", "50 USD", 0.95),
        MockDocument(2, "جولة بالجنيه المصري", "200 EGP", 0.90),
        MockDocument(3, "جولة بالدولار B", "20 USD", 0.85),
    ]

    query = "أرخص 3 أنشطة"
    ranked = NLPController.apply_price_ranking_if_requested(query, docs, default_limit=3)

    # In USD group: Doc 3 ($20 USD) comes before Doc 1 ($50 USD).
    # In EGP group: Doc 2 (200 EGP).
    # Results are grouped and sorted by currency, NEVER in raw similarity order [1, 2, 3].
    assert len(ranked) == 3
    assert [d.doc_id for d in ranked] == [3, 1, 2]
    assert NLPController.extract_numeric_price(ranked[0])[0] == 20.0
    assert NLPController.extract_numeric_price(ranked[1])[0] == 50.0
    assert NLPController.extract_numeric_price(ranked[2])[0] == 200.0


def test_scenario_6_relevance_filtering_preserved():
    """
    Scenario 6: Relevance filtering is strictly preserved.
    Unrelated or below-threshold documents must never be injected to fill N.
    """
    # Simulate vector search where low score items (< 0.75) are already filtered out
    relevant_docs = [
        MockDocument(1, "جولة الأهرامات", "25 USD", 0.82),
    ]
    # User asks for 3, but only 1 relevant document exists
    query = "اعرض أرخص 3 أنشطة للأهرامات"
    ranked = NLPController.apply_price_ranking_if_requested(query, relevant_docs, default_limit=3)

    assert len(ranked) == 1
    assert ranked[0].doc_id == 1


def test_scenario_7_source_and_context_synchronization():
    """
    Scenario 7: Full pipeline test verifying that retrieved_documents
    used in prompt generation match the exact price-ranked order.
    """
    docs = [
        MockDocument(1, "Tour A", "30 USD", 0.85),
        MockDocument(2, "Tour B", "10 USD", 0.84),
        MockDocument(3, "Tour C", "20 USD", 0.83),
    ]

    # Initialize a lightweight controller mock
    mock_vectordb = MagicMock()
    mock_gen = MagicMock()
    mock_gen.enums.SYSTEM.value = "system"
    mock_gen.enums.USER.value = "user"
    mock_gen.enums.ASSISTANT.value = "assistant"
    mock_gen.construct_prompt.side_effect = lambda prompt, role: {"role": role, "content": prompt}
    mock_gen.generate_text.return_value = "Here are the cheapest 3 tours in order: Tour B ($10), Tour C ($20), Tour A ($30)."

    controller = NLPController(
        vectordb_client=mock_vectordb,
        generation_client=mock_gen,
        embedding_client=MagicMock(),
        template_parser=MagicMock(),
    )
    controller.app_settings = MagicMock()
    controller.app_settings.INPUT_DEFAULT_MAX_CHARACTERS = 4000
    controller.app_settings.GENERATION_DEFAULT_MAX_TOKENS = 500
    controller.app_settings.GENERATION_DEFAULT_TEMPERATURE = 0.2

    # Mock template parser to return document string
    controller.template_parser.get.side_effect = lambda cat, key, vars=None: (
        f"Doc {vars['doc_num']}: {vars['chunk_text']}" if vars and 'doc_num' in vars else "Prompt"
    )

    query = "cheapest 3 tours"
    answer, prompt, cached = controller.generate_chat_answer(
        query=query,
        retrieved_documents=docs,
    )

    # Prompt must contain Tour B (10 USD) first, Tour C (20 USD) second, Tour A (30 USD) third
    pos_b = prompt.find("Tour B")
    pos_c = prompt.find("Tour C")
    pos_a = prompt.find("Tour A")
    assert pos_b != -1 and pos_c != -1 and pos_a != -1
    assert pos_b < pos_c < pos_a


def test_scenario_8_arabic_tourism_terminology_normalization():
    """
    Scenario 8: Arabic tourism terminology normalization.
    Replaces accidental typos (مرشد سياسي -> مرشد سياحي, جولة سياسية -> جولة سياحية)
    while strictly preserving unrelated political terms.
    """
    # 1. Corrections that MUST occur
    corrections = [
        ("يتضمن البرنامج مرشد سياسي معتمد.", "يتضمن البرنامج مرشد سياحي معتمد."),
        ("سيكون المرشد السياسي بانتظاركم في الفندق.", "سيكون المرشد السياحي بانتظاركم في الفندق."),
        ("نوفر مرشدين سياسيين ذوي خبرة.", "نوفر مرشدين سياحيين ذوي خبرة."),
        ("حضر مرشدون سياسيون للموقع.", "حضر مرشدون سياحيون للموقع."),
        ("رافق المجموعة مرشدة سياسية محترفة.", "رافق المجموعة مرشدة سياحية محترفة."),
        ("لدينا مرشدات سياسيات متميزات.", "لدينا مرشدات سياحيات متميزات."),
        ("استعن بمرشداً سياسياً في جولتك.", "استعن بمرشداً سياحياً في جولتك."),
        ("استمتع بأجمل جولة سياسية في خان الخليلي.", "استمتع بأجمل جولة سياحية في خان الخليلي."),
        ("هذه الجولة السياسية ممتعة جداً.", "هذه الجولة السياحية ممتعة جداً."),
        ("تنظيم جولات سياسية ممتعة.", "تنظيم جولات سياحية ممتعة."),
        ("حجزنا جولتين سياسيتين في أسوان.", "حجزنا جولتين سياحيتين في أسوان."),
        ("الاستمتاع برحلة مع مرشدنا السياسي.", "الاستمتاع برحلة مع مرشدنا السياحي."),
    ]
    for inp, expected in corrections:
        assert NLPController._clean_tourism_terminology(inp) == expected

    # 2. Political contexts that MUST NOT change
    political = [
        "قال محلل سياسي في التلفزيون.",
        "هذا قرار سياسي هام للحكومة.",
        "أزمة سياسية تعصف بالمنطقة.",
        "انضم إلى حزب سياسي جديد.",
        "يدرس في كلية علوم سياسية.",
        "الوضع السياسي في الدولة مستقر.",
    ]
    for text in political:
        assert NLPController._clean_tourism_terminology(text) == text


def test_scenario_9_english_ranking_and_general_query_neutrality():
    """
    Scenario 9: English ranking intents and non-ranking query neutrality.
    """
    docs = [
        MockDocument(1, "Pyramids Tour", "25 USD", 0.88),
        MockDocument(2, "Museum Tour", "15 USD", 0.87),
        MockDocument(3, "Nile Cruise", "45 USD", 0.86),
    ]

    # English ascending
    ranked_en_asc = NLPController.apply_price_ranking_if_requested("cheapest 2 activities", docs, default_limit=3)
    assert len(ranked_en_asc) == 2
    assert [d.doc_id for d in ranked_en_asc] == [2, 1]

    # English descending
    ranked_en_desc = NLPController.apply_price_ranking_if_requested("most expensive activity", docs, default_limit=3)
    assert len(ranked_en_desc) == 1
    assert ranked_en_desc[0].doc_id == 3

    # General factual query (no ranking intent) -> preserves original relevance order
    neutral_query = "أين تقع أهرامات الجيزة؟"
    ranked_neutral = NLPController.apply_price_ranking_if_requested(neutral_query, docs, default_limit=3)
    assert [d.doc_id for d in ranked_neutral] == [1, 2, 3]

    neutral_en = "What are the opening hours of the Grand Egyptian Museum?"
    ranked_neutral_en = NLPController.apply_price_ranking_if_requested(neutral_en, docs, default_limit=3)
    assert [d.doc_id for d in ranked_neutral_en] == [1, 2, 3]


def test_scenario_10_byte_for_byte_parity():
    """
    Scenario 10: Byte-for-byte parity between api/ and src/ directories.
    """
    files_to_check = [
        ("api/controllers/NLPController.py", "src/controllers/NLPController.py"),
        ("api/routes/nlp.py", "src/routes/nlp.py"),
        ("api/stores/llm/templates/locales/ar/rag.py", "src/stores/llm/templates/locales/ar/rag.py"),
        ("api/stores/llm/templates/locales/en/rag.py", "src/stores/llm/templates/locales/en/rag.py"),
        ("api/stores/llm/providers/OpenAIProvider.py", "src/stores/llm/providers/OpenAIProvider.py"),
        ("api/stores/llm/LLMProviderFactory.py", "src/stores/llm/LLMProviderFactory.py"),
        ("api/helpers/config.py", "src/helpers/config.py"),
    ]

    for api_file, src_file in files_to_check:
        assert filecmp.cmp(api_file, src_file, shallow=False), f"Parity mismatch between {api_file} and {src_file}"


def test_scenario_11_explicit_zero_price_and_free_activities():
    """
    Scenario 11: Explicit zero-price and genuine free activities.
    - 'Free', 'مجاني', 'دخول مجاني', 0, 0.0, '0 USD' extract as price 0.0.
    - Missing prices (None, 'Contact for price', 'غير محدد') extract as None.
    - Free activities rank as the cheapest in ascending order.
    - Free activities rank as lowest in descending order.
    """
    # 1. Extraction checks
    free_cases = [
        ({"price": "Free"}, 0.0, "USD"),
        ({"price": "مجاني"}, 0.0, "USD"),
        ({"price": "دخول مجاني"}, 0.0, "USD"),
        ({"price": 0}, 0.0, "USD"),
        ({"price": 0.0}, 0.0, "USD"),
        ({"price": "0.0 USD"}, 0.0, "USD"),
        ({"price": "0 EGP"}, 0.0, "EGP"),
    ]
    for meta, exp_p, exp_c in free_cases:
        doc = MockDocument(1, "Free Place", meta["price"], 0.9)
        doc.metadata = meta
        extracted = NLPController.extract_numeric_price(doc)
        assert extracted is not None, f"Failed for {meta}"
        assert extracted[0] == exp_p
        assert extracted[1] == exp_c

    missing_cases = [
        {"price": None},
        {"price": ""},
        {"price": "Contact for price"},
        {"price": "Inquire"},
        {"price": "غير محدد"},
        {"price": "اتصل بنا لمعرفة السعر"},
    ]
    for meta in missing_cases:
        doc = MockDocument(2, "Missing Price Place", meta["price"], 0.9)
        doc.metadata = meta
        assert NLPController.extract_numeric_price(doc) is None, f"Expected None for {meta}"

    # 2. Ranking checks (Ascending and Descending)
    candidates = [
        MockDocument(1, "Paid Tour High", "50.0 USD", 0.85),
        MockDocument(2, "Free Museum Walk", "Free", 0.90),
        MockDocument(3, "Paid Tour Low", "15.0 USD", 0.88),
        MockDocument(4, "Unknown Price Tour", "Contact us", 0.80),
    ]

    # Ascending (cheapest 2): Free (0.0) first, Paid Tour Low (15.0) second
    asc_ranked = NLPController.apply_price_ranking_if_requested("أرخص نشاطين", candidates, default_limit=4)
    assert len(asc_ranked) == 2
    assert [d.doc_id for d in asc_ranked] == [2, 3]

    # Descending (most expensive 3): 50.0, 15.0, 0.0 (Free is last)
    desc_ranked = NLPController.apply_price_ranking_if_requested("أغلى 3 أنشطة", candidates, default_limit=4)
    assert len(desc_ranked) == 3
    assert [d.doc_id for d in desc_ranked] == [1, 3, 2]
