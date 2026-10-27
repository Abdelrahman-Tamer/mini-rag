from .BaseController import BaseController
from stores.llm.LLMEnums import DocumentTypeEnum
from helpers.cache import global_response_cache
import logging

logger = logging.getLogger('uvicorn.error')

MAX_HISTORY_TURNS = 5

CITY_MAP = {
    "cairo": "Cairo", "القاهرة": "Cairo", "قاهرة": "Cairo",
    "luxor": "Luxor", "الاقصر": "Luxor", "الأقصر": "Luxor", "أقصر": "Luxor",
    "aswan": "Aswan", "اسوان": "Aswan", "أسوان": "Aswan",
    "alexandria": "Alexandria", "الاسكندرية": "Alexandria", "الإسكندرية": "Alexandria", "اسكندرية": "Alexandria",
    "hurghada": "Hurghada", "الغردقة": "Hurghada", "غردقة": "Hurghada",
    "sharm": "Sharm El Sheikh", "شرم الشيخ": "Sharm El Sheikh", "شرم": "Sharm El Sheikh",
    "giza": "Giza", "الجيزة": "Giza", "جيزة": "Giza",
    "dahab": "Dahab", "دهب": "Dahab",
    "siwa": "Siwa", "سيوة": "Siwa",
}

class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client,
                embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.collection_name = self.app_settings.VECTOR_DB_COLLECTION_NAME
        self.score_threshold = self.app_settings.SCORE_THRESHOLD

    def extract_city(self, text: str):
        if not text:
            return None
        text_lower = text.lower()
        for token, city_name in CITY_MAP.items():
            if token in text_lower:
                return city_name
        return None

    def needs_query_rewriting(self, query: str, history: list) -> bool:
        """
        Determines whether a query needs conversational rewriting.
        Independent/standalone queries bypass the extra LLM call to save 1.5-2.0s of latency.
        """
        if not history or len(history) == 0:
            return False

        q = query.strip().lower()
        words = q.split()

        # Context-dependent markers in Arabic & English
        context_markers = {
            "ده", "دي", "دول", "هناك", "بكام", "سعره", "سعرها", "فين", "طب", "و ده", "و دي",
            "قريب", "منها", "منه", "احجزلي", "احجز", "ضيفه", "ضيفها", "بالنسبة", "تاني",
            "it", "this", "that", "these", "those", "there", "them", "cost", "price",
            "nearby", "how much", "what about", "is it", "and", "too", "also", "add it",
        }

        if len(words) <= 4:
            return True

        for marker in context_markers:
            if marker in q:
                return True

        return False

    def rewrite_query(self, query: str, history: list) -> str:
        if not self.needs_query_rewriting(query, history):
            return query

        context_lines = []
        for msg in history[-MAX_HISTORY_TURNS*2:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_lines.append(f"{role}: {content}")
        context = "\n".join(context_lines)

        rewrite_prompt = (
            "You are a search query rewriter. Given the conversation history "
            "and the latest user query, rewrite the latest query as a standalone, "
            "detailed search query. Return ONLY the rewritten query, preserving any explicit city, category, or budget.\n\n"
            f"Conversation:\n{context}\n\n"
            f"Latest query: {query}\n\n"
            f"Standalone query:"
        )

        try:
            rewritten = self.generation_client.generate_text(
                prompt=rewrite_prompt,
                max_output_tokens=100,
                temperature=0.0,
            )
            return (rewritten or query).strip().strip('"')
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}. Falling back to original query.")
            return query

    def is_greeting_query(self, query: str) -> bool:
        query_clean = query.strip().lower().rstrip("?!.,;:")
        greetings = {
            "hi", "hello", "hey", "howdy", "greetings",
            "good morning", "good afternoon", "good evening", "good day",
            "what's up", "sup", "yo",
            "مرحبا", "اهلا", "اهلاً", "السلام عليكم", "سلام",
            "bonjour", "salut", "bonsoir",
            "hola", "buenos días", "buenas tardes",
        }
        return query_clean in greetings

    def chat(self, query: str, limit: int = 5, prior_history: list = None, city: str = None, category: str = None):
        if not query or not query.strip():
            return None

        # Dynamically detect language
        is_ar = any('\u0600' <= char <= '\u06ff' for char in query)
        detected_lang = "ar" if is_ar else "en"
        self.template_parser.set_language(detected_lang)

        # Extract city if not passed
        target_city = city or self.extract_city(query)

        search_query = self.rewrite_query(query, prior_history)
        vector = self.embedding_client.embed_text(
            text=search_query,
            document_type=DocumentTypeEnum.QUERY.value
        )

        if not vector or len(vector) == 0:
            return None

        # 1. Try search with city filter if detected
        results = None
        if target_city and hasattr(self.vectordb_client, "search_by_vector"):
            filter_payload = {"city": target_city}
            if category:
                filter_payload["category"] = category
            results = self.vectordb_client.search_by_vector(
                collection_name=self.collection_name,
                vector=vector,
                limit=limit,
                filter=filter_payload,
            )

        # 2. Fallback to general vector search
        if not results:
            results = self.vectordb_client.search_by_vector(
                collection_name=self.collection_name,
                vector=vector,
                limit=limit,
            )

        if not results:
            return None

        results = [doc for doc in results if doc.score >= self.score_threshold]
        return results if results else None

    def generate_chat_answer(self, query: str, retrieved_documents: list, prior_history: list = None, system_prompt: str = None):
        is_ar = any('\u0600' <= char <= '\u06ff' for char in query)
        detected_lang = "ar" if is_ar else "en"
        self.template_parser.set_language(detected_lang)

        # Check response cache if standalone query
        is_standalone = not prior_history or len(prior_history) == 0
        cache_key = None
        if is_standalone:
            detected_city = self.extract_city(query)
            cache_key = global_response_cache.generate_key(
                query=query,
                language=detected_lang,
                city=detected_city
            )
            cached_data = global_response_cache.get(cache_key)
            if cached_data:
                return cached_data["answer"], cached_data.get("full_prompt", ""), True

        if system_prompt is None:
            system_prompt = self.template_parser.get("rag", "system_prompt")

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        max_input_chars = self.app_settings.INPUT_DEFAULT_MAX_CHARACTERS
        footer_len = len(footer_prompt) if footer_prompt else 0
        doc_budget = max_input_chars - footer_len

        documents_prompts = ""
        for idx, doc in enumerate(retrieved_documents):
            doc_prompt = self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": doc.text,
                "score": round(doc.score, 3),
            })
            if not doc_prompt:
                continue
            separator = "\n\n" if documents_prompts else ""
            if len(documents_prompts) + len(separator) + len(doc_prompt) > doc_budget:
                break
            documents_prompts += separator + doc_prompt

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        if prior_history:
            for msg in prior_history[-MAX_HISTORY_TURNS*2:]:
                role_value = msg.get("role", "user")
                if role_value == "assistant":
                    mapped_role = self.generation_client.enums.ASSISTANT.value
                else:
                    mapped_role = self.generation_client.enums.USER.value
                chat_history.append(
                    self.generation_client.construct_prompt(
                        prompt=msg.get("content", ""),
                        role=mapped_role,
                    )
                )

        full_prompt = (
            "\n\n".join([documents_prompts, footer_prompt])
            if documents_prompts
            else footer_prompt
        )

        chat_history.append(
            self.generation_client.construct_prompt(
                prompt=full_prompt,
                role=self.generation_client.enums.USER.value,
            )
        )

        answer = self.generation_client.generate_text(
            prompt=chat_history,
            max_output_tokens=self.app_settings.GENERATION_DEFAULT_MAX_TOKENS,
            temperature=self.app_settings.GENERATION_DEFAULT_TEMPERATURE,
        )

        # Cache standalone results
        if is_standalone and cache_key and answer:
            global_response_cache.set(cache_key, {
                "answer": answer,
                "full_prompt": full_prompt,
            })

        return answer, full_prompt, False

    def sync_activity(self, activity: dict) -> dict:
        """
        Idempotently inserts or updates an activity in Qdrant with embeddings.
        """
        act_id = activity.get("activity_id")
        title = activity.get("title", "")
        city = activity.get("city", "Egypt")
        category = activity.get("category", "General")
        price = activity.get("price", "0.0 USD")
        image_url = activity.get("image_url", "")
        link = activity.get("link") or f"https://www.nileguide.online/activities/{act_id}"
        desc = activity.get("description", "")

        text_content = (
            f"Activity: {title}\n"
            f"City: {city}\n"
            f"Category: {category}\n"
            f"Price: {price}\n"
            f"Link: {link}\n"
            f"Image: {image_url}\n"
            f"Details: {desc}"
        )

        vector = self.embedding_client.embed_text(
            text=text_content,
            document_type=DocumentTypeEnum.DOCUMENT.value
        )

        metadata = {
            "activity_id": act_id,
            "title": title,
            "city": city,
            "category": category,
            "price": price,
            "image_url": image_url,
            "link": link,
            "is_active": True,
        }

        # Upsert point to Qdrant
        if hasattr(self.vectordb_client, "insert_one"):
            self.vectordb_client.insert_one(
                collection_name=self.collection_name,
                text=text_content,
                vector=vector,
                metadata=metadata,
                record_id=str(act_id)
            )

        # Invalidate cache
        global_response_cache.clear()
        return {"status": "upserted", "activity_id": act_id}

    def delete_activity(self, activity_id: int) -> dict:
        """
        Removes an activity from Qdrant.
        """
        if hasattr(self.vectordb_client, "delete_by_filter"):
            self.vectordb_client.delete_by_filter(
                collection_name=self.collection_name,
                filter={"activity_id": activity_id}
            )

        global_response_cache.clear()
        return {"status": "deleted", "activity_id": activity_id}
