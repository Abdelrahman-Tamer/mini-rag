from .BaseController import BaseController
from stores.llm.LLMEnums import DocumentTypeEnum
from helpers.cache import global_response_cache
import logging
import uuid
import os
import threading

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
    _asset_locks = {}
    _asset_locks_guard = threading.Lock()

    @classmethod
    def _get_asset_lock(cls, asset_id: str) -> threading.Lock:
        with cls._asset_locks_guard:
            if asset_id not in cls._asset_locks:
                cls._asset_locks[asset_id] = threading.Lock()
            return cls._asset_locks[asset_id]

    def _deduplicate_search_results(self, documents: list) -> list:
        """
        Deduplicates vector search results to prevent duplicate chunks
        if an old generation cleanup failed in Qdrant (Scenario C).
        Prefers the first (highest scoring) chunk for any (asset_id, chunk_order).
        """
        if not documents:
            return []
        seen_keys = set()
        deduped = []
        for doc in documents:
            meta = getattr(doc, "metadata", {}) or {}
            asset_id = meta.get("asset_id")
            chunk_order = meta.get("chunk_order")
            if asset_id and chunk_order is not None:
                dedup_key = f"{asset_id}:{chunk_order}"
                if dedup_key in seen_keys:
                    logger.debug(f"Filtered duplicate chunk from obsolete generation: {dedup_key}")
                    continue
                seen_keys.add(dedup_key)
            deduped.append(doc)
        return deduped


    def __init__(self, vectordb_client, generation_client,
                embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.collection_name = self.app_settings.VECTOR_DB_COLLECTION_NAME
        self.score_threshold = self.app_settings.SCORE_THRESHOLD
        self.logger = logger

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
        fetch_limit = max(limit * 2, limit + 5)
        results = None
        if target_city and hasattr(self.vectordb_client, "search_by_vector"):
            filter_payload = {"city": target_city}
            if category:
                filter_payload["category"] = category
            results = self.vectordb_client.search_by_vector(
                collection_name=self.collection_name,
                vector=vector,
                limit=fetch_limit,
                filter=filter_payload,
            )

        # 2. Fallback to general vector search
        if not results:
            results = self.vectordb_client.search_by_vector(
                collection_name=self.collection_name,
                vector=vector,
                limit=fetch_limit,
            )

        if not results:
            return None

        results = [doc for doc in results if doc.score >= self.score_threshold]
        results = self._deduplicate_search_results(results)[:limit]
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
        if hasattr(self.vectordb_client, "delete_by_id"):
            self.vectordb_client.delete_by_id(
                collection_name=self.collection_name,
                record_id=activity_id
            )

        if hasattr(self.vectordb_client, "delete_by_filter"):
            self.vectordb_client.delete_by_filter(
                collection_name=self.collection_name,
                filter={"activity_id": activity_id}
            )

        global_response_cache.clear()
        return {"status": "deleted", "activity_id": activity_id}

    def index_chunks(self, chunks: list) -> dict:
        """
        Generates dense vector embeddings and indexes document chunks into Qdrant.
        Uses deterministic UUIDv5 point IDs to avoid collisions and enable idempotent retries.
        """
        if not chunks:
            return {"status": "no_chunks", "indexed": 0}

        if not self.vectordb_client:
            logger.warning("Vector DB client not available for indexing chunks.")
            return {"status": "vectordb_unavailable", "indexed": 0}

        texts = []
        metadata = []
        record_ids = []
        target_asset_id = None

        for i, chunk in enumerate(chunks):
            if hasattr(chunk, "chunk_text"):
                text = chunk.chunk_text
                meta = dict(chunk.chunk_metadata) if getattr(chunk, "chunk_metadata", None) else {}
                order = getattr(chunk, "chunk_order", i + 1)
            elif isinstance(chunk, dict):
                text = chunk.get("chunk_text") or chunk.get("text") or ""
                meta = dict(chunk.get("chunk_metadata") or chunk.get("metadata") or {})
                order = chunk.get("chunk_order", i + 1)
            else:
                continue

            if not text or not str(text).strip():
                continue

            asset_id = str(meta.get("asset_id", "")).strip()
            if asset_id:
                target_asset_id = asset_id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nileguide_asset_{asset_id}_chunk_{order}"))
            else:
                point_id = str(uuid.uuid4())

            meta["asset_id"] = asset_id
            meta["chunk_order"] = order

            texts.append(str(text))
            metadata.append(meta)
            record_ids.append(point_id)

        if not texts:
            return {"status": "empty", "indexed": 0}

        # Generate embeddings
        vectors = []
        for t in texts:
            vec = self.embedding_client.embed_text(
                text=t,
                document_type=DocumentTypeEnum.DOCUMENT.value
            )
            if not vec:
                logger.error("Failed to generate embedding during chunk indexing.")
                return {"status": "embedding_failed", "indexed": 0}
            vectors.append(vec)

        # Batch insert into Qdrant
        if hasattr(self.vectordb_client, "insert_many"):
            success = self.vectordb_client.insert_many(
                collection_name=self.collection_name,
                texts=texts,
                vectors=vectors,
                metadata=metadata,
                record_ids=record_ids,
                batch_size=50
            )
        else:
            success = False
            for idx in range(len(texts)):
                ok = self.vectordb_client.insert_one(
                    collection_name=self.collection_name,
                    text=texts[idx],
                    vector=vectors[idx],
                    metadata=metadata[idx],
                    record_id=record_ids[idx]
                )
                if ok:
                    success = True

        global_response_cache.clear()
        return {
            "status": "indexed" if success else "failed",
            "indexed": len(texts) if success else 0,
            "asset_id": target_asset_id
        }

    def safe_reindex_chunks(
        self,
        asset_id: str,
        chunks: Optional[list],
        old_chunk_count: Optional[int] = None,
        old_generation_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        allow_empty: bool = True
    ) -> dict:
        """
        Failure-safe recoverable document re-indexing for Qdrant.
        Serializes concurrent indexing per-asset using per-asset lock (Scenario D).
        """
        if not asset_id or not str(asset_id).strip():
            return {"status": "invalid_asset_id", "indexed": 0}

        clean_asset_id = str(asset_id).strip()

        with self._get_asset_lock(clean_asset_id):
            return self._safe_reindex_chunks_locked(
                clean_asset_id=clean_asset_id,
                chunks=chunks,
                old_chunk_count=old_chunk_count,
                old_generation_id=old_generation_id,
                generation_id=generation_id,
                allow_empty=allow_empty
            )

    def _safe_reindex_chunks_locked(
        self,
        clean_asset_id: str,
        chunks: Optional[list],
        old_chunk_count: Optional[int] = None,
        old_generation_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        allow_empty: bool = True
    ) -> dict:

        # Protection against malformed/null extraction
        if chunks is None:
            logger.warning(f"Null chunk list received for asset '{clean_asset_id}'. Refusing to modify vector store.")
            return {
                "status": "extraction_failed",
                "indexed": 0,
                "preserved": True,
                "detail": "Null chunk extraction indicates upstream processing failure. Existing vectors preserved."
            }

        # Handling empty chunk list: distinguish intentional deletion from accidental empty extraction
        if not chunks:
            if not allow_empty:
                logger.warning(f"Empty chunk list received for asset '{clean_asset_id}' with allow_empty=False. Preserving existing vectors.")
                return {
                    "status": "empty_chunks_rejected",
                    "indexed": 0,
                    "preserved": True,
                    "detail": "Empty chunk extraction rejected to prevent accidental document erasure. Pass allow_empty=True if intentional."
                }
            del_res = self.delete_file_vectors(clean_asset_id)
            return {"status": "indexed", "indexed": 0, "pruned": del_res.get("deleted", False)}

        if not self.vectordb_client:
            logger.error("Vector DB client not available for safe re-indexing.")
            return {"status": "vectordb_unavailable", "indexed": 0}

        texts = []
        metas = []
        orders = []

        for i, chunk in enumerate(chunks):
            if hasattr(chunk, "chunk_text"):
                text = chunk.chunk_text
                meta = dict(chunk.chunk_metadata) if getattr(chunk, "chunk_metadata", None) else {}
                order = getattr(chunk, "chunk_order", i + 1)
            elif isinstance(chunk, dict):
                text = chunk.get("chunk_text") or chunk.get("text") or ""
                meta = dict(chunk.get("chunk_metadata") or chunk.get("metadata") or {})
                order = chunk.get("chunk_order", i + 1)
            else:
                continue

            if not text or not str(text).strip():
                continue

            texts.append(str(text))
            metas.append(meta)
            orders.append(order)

        if not texts:
            if not allow_empty:
                return {"status": "empty_chunks_rejected", "indexed": 0, "preserved": True}
            return {"status": "empty", "indexed": 0}

        # Step 2: Generate and validate embeddings BEFORE touching Qdrant
        vectors = []
        expected_dim = None
        for t in texts:
            try:
                vec = self.embedding_client.embed_text(
                    text=t,
                    document_type=DocumentTypeEnum.DOCUMENT.value
                )
            except Exception as e:
                logger.error(f"Embedding generation threw exception for chunk: {e}")
                return {"status": "embedding_failed", "indexed": 0, "error": str(e), "preserved": True}

            if not vec or not isinstance(vec, list) or len(vec) == 0:
                logger.error("Embedding generation returned empty or invalid vector.")
                return {"status": "embedding_failed", "indexed": 0, "preserved": True}

            if expected_dim is None:
                expected_dim = len(vec)
            elif len(vec) != expected_dim:
                logger.error(f"Embedding dimension mismatch: expected {expected_dim}, got {len(vec)}")
                return {"status": "dimension_mismatch", "indexed": 0, "preserved": True}

            vectors.append(vec)

        # Step 3: Assign point IDs and generation metadata
        record_ids = []
        metadata = []
        for i, text in enumerate(texts):
            order = orders[i]
            meta = dict(metas[i])
            meta["asset_id"] = clean_asset_id
            meta["chunk_order"] = order

            if generation_id:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nileguide_asset_{clean_asset_id}_gen_{generation_id}_chunk_{order}"))
                meta["generation_id"] = generation_id
                meta["is_active"] = False  # Staged
            else:
                # Phase 1.1 deterministic ID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nileguide_asset_{clean_asset_id}_chunk_{order}"))
                meta["is_active"] = True

            record_ids.append(point_id)
            metadata.append(meta)

        # Step 4: Batch upsert with partial-write rollback on failure
        inserted_point_ids = []
        batch_size = 50
        write_failed = False
        write_error = None

        for b_start in range(0, len(texts), batch_size):
            b_end = b_start + batch_size
            b_texts = texts[b_start:b_end]
            b_vectors = vectors[b_start:b_end]
            b_meta = metadata[b_start:b_end]
            b_ids = record_ids[b_start:b_end]

            try:
                if hasattr(self.vectordb_client, "insert_many"):
                    ok = self.vectordb_client.insert_many(
                        collection_name=self.collection_name,
                        texts=b_texts,
                        vectors=b_vectors,
                        metadata=b_meta,
                        record_ids=b_ids,
                        batch_size=batch_size
                    )
                else:
                    ok = True
                    for idx in range(len(b_texts)):
                        single_ok = self.vectordb_client.insert_one(
                            collection_name=self.collection_name,
                            text=b_texts[idx],
                            vector=b_vectors[idx],
                            metadata=b_meta[idx],
                            record_id=b_ids[idx]
                        )
                        if not single_ok:
                            ok = False
                            break

                if ok:
                    inserted_point_ids.extend(b_ids)
                else:
                    write_failed = True
                    break
            except Exception as e:
                write_failed = True
                write_error = str(e)
                break

        # Rollback partial batch write on failure to guarantee Qdrant integrity
        if write_failed:
            logger.error(
                f"Batch write failed during safe re-index for asset {clean_asset_id} "
                f"(generation={generation_id}). Rolling back {len(inserted_point_ids)} partial points."
            )
            try:
                if inserted_point_ids and hasattr(self.vectordb_client, "delete_by_ids"):
                    self.vectordb_client.delete_by_ids(self.collection_name, inserted_point_ids)
                elif hasattr(self.vectordb_client, "delete_by_filter"):
                    rollback_filter = {"asset_id": clean_asset_id}
                    if generation_id:
                        rollback_filter["generation_id"] = generation_id
                    self.vectordb_client.delete_by_filter(self.collection_name, rollback_filter)
            except Exception as rb_err:
                logger.error(f"Rollback cleanup error: {rb_err}")

            return {
                "status": "vectordb_failed",
                "indexed": 0,
                "generation_id": generation_id,
                "rolled_back": True,
                "preserved": True,
                "error": write_error or "Batch insertion failure"
            }

        # Step 5: Promotion & obsolete cleanup
        obsolete_pruned = 0
        new_count = len(texts)

        # 5a. If versioned, activate new generation
        if generation_id and hasattr(self.vectordb_client, "set_payload"):
            try:
                self.vectordb_client.set_payload(
                    collection_name=self.collection_name,
                    payload={"metadata": {"is_active": True}},
                    record_ids=record_ids
                )
            except Exception as e:
                logger.warning(f"Could not activate generation {generation_id}: {e}")

        # 5b. Prune previous active generation with error handling
        cleanup_failed = False
        if old_generation_id and old_generation_id != generation_id:
            if hasattr(self.vectordb_client, "delete_by_filter"):
                try:
                    del_ok = self.vectordb_client.delete_by_filter(
                        self.collection_name,
                        {"asset_id": clean_asset_id, "generation_id": old_generation_id}
                    )
                    if del_ok is False:
                        cleanup_failed = True
                        logger.warning(f"Cleanup of obsolete generation {old_generation_id} returned False for asset {clean_asset_id}. Search deduplication will prevent duplicate chunk exposure.")
                except Exception as del_err:
                    cleanup_failed = True
                    logger.warning(f"Could not prune obsolete generation {old_generation_id} for asset {clean_asset_id}: {del_err}. Search deduplication will prevent duplicate chunk exposure.")

        # 5c. Prune legacy un-versioned points if old_chunk_count > new_count
        if old_chunk_count and old_chunk_count > new_count and not generation_id:
            obsolete_ids = [
                str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nileguide_asset_{clean_asset_id}_chunk_{o}"))
                for o in range(new_count + 1, old_chunk_count + 1)
            ]
            if hasattr(self.vectordb_client, "delete_by_ids"):
                self.vectordb_client.delete_by_ids(
                    collection_name=self.collection_name,
                    record_ids=obsolete_ids
                )
                obsolete_pruned = len(obsolete_ids)
            elif hasattr(self.vectordb_client, "delete_by_id"):
                for oid in obsolete_ids:
                    self.vectordb_client.delete_by_id(
                        collection_name=self.collection_name,
                        record_id=oid
                    )
                obsolete_pruned = len(obsolete_ids)

        # 5d. Also prune legacy points if migrating to versioned generation
        if generation_id and old_chunk_count and old_chunk_count > 0:
            legacy_ids = [
                str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nileguide_asset_{clean_asset_id}_chunk_{o}"))
                for o in range(1, old_chunk_count + 1)
            ]
            if hasattr(self.vectordb_client, "delete_by_ids"):
                self.vectordb_client.delete_by_ids(self.collection_name, legacy_ids)

        # 5e. Garbage collect orphaned inactive generations
        if hasattr(self.vectordb_client, "delete_by_filter"):
            try:
                self.vectordb_client.delete_by_filter(
                    self.collection_name,
                    {"asset_id": clean_asset_id, "is_active": False}
                )
            except Exception:
                pass

        global_response_cache.clear()
        res = {
            "status": "indexed",
            "indexed": new_count,
            "obsolete_pruned": obsolete_pruned,
            "generation_id": generation_id,
            "asset_id": clean_asset_id,
            "promoted": True
        }
        if cleanup_failed:
            res["cleanup_failed"] = True
        return res

    def delete_file_vectors(self, asset_id: str) -> dict:
        """
        Removes all vectors belonging to a specific asset from Qdrant.
        """
        if not asset_id or not str(asset_id).strip():
            return {"status": "invalid_asset_id", "deleted": False}

        clean_asset_id = str(asset_id).strip()

        if not self.vectordb_client:
            logger.warning("Vector DB client not available for deleting file vectors.")
            return {"status": "vectordb_unavailable", "deleted": False}

        success = False
        if hasattr(self.vectordb_client, "delete_by_filter"):
            success = self.vectordb_client.delete_by_filter(
                collection_name=self.collection_name,
                filter={"asset_id": clean_asset_id}
            )

        global_response_cache.clear()
        return {
            "status": "deleted" if success else "failed",
            "deleted": bool(success),
            "asset_id": clean_asset_id
        }
