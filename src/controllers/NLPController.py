from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Optional
import json
import functools
import os
@functools.lru_cache(maxsize=1)
def load_city_mappings(path: str = "city_mappings.json"):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client,
                embedding_client, template_parser,
                rerank_client=None, rerank_model_id=None):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.rerank_client = rerank_client
        self.rerank_model_id = rerank_model_id
        self.collection_name = self.app_settings.VECTOR_DB_COLLECTION_NAME
        self.score_threshold = self.app_settings.SCORE_THRESHOLD

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

    def detect_city_from_query(self, query: str, known_cities: list) -> Optional[str]:
        query_lower = query.lower()
        matched_cities = [c for c in known_cities if c and c.lower() in query_lower]
        if len(matched_cities) == 1:
            return matched_cities[0]
        if not matched_cities:
            mappings = load_city_mappings(self.app_settings.CITY_MAPPINGS_PATH)
            matched_countries = [c for c in mappings if c in query_lower]
            if len(matched_countries) == 1:
                return mappings[matched_countries[0]]
        return None

    def reset_vector_db_collection(self, project: Project):
        return self.vectordb_client.delete_by_filter(
            collection_name=self.collection_name,
            filter={"project_id": project.project_id}
        )

    def get_vector_db_collection_info(self, project: Project):
        collection_info = self.vectordb_client.get_collection_info(collection_name=self.collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )

    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                chunks_ids: List[int],
                                do_reset: bool = False):

        if do_reset:
            _ = self.reset_vector_db_collection(project=project)

        texts = [c.chunk_text for c in chunks]
        metadata = []
        for c in chunks:
            chunk_meta = dict(c.chunk_metadata) if c.chunk_metadata else {}
            chunk_meta["project_id"] = project.project_id
            metadata.append(chunk_meta)

        vectors = [
            self.embedding_client.embed_text(text=text,
                                            document_type=DocumentTypeEnum.DOCUMENT.value)
            for text in texts
        ]

        _ = self.vectordb_client.create_collection(
            collection_name=self.collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=False,
        )

        _ = self.vectordb_client.insert_many(
            collection_name=self.collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return True

    def answer_rag_question(self, project: Project, query: str, limit: int = 10):

        answer, full_prompt, chat_history = None, None, None

        retrieved_documents = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history

        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": doc.text,
                "score": round(doc.score, 3),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history

    def chat(self, query: str, limit: int = 5, city: str = None, doc_type: str = None):

        if not query or not query.strip():
            return None

        vector = self.embedding_client.embed_text(text=query,
                                                document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return None

        search_filter = {}
        if city:
            search_filter["city"] = city
        if doc_type:
            search_filter["doc_type"] = doc_type

        # retrieve more for reranking
        retrieve_limit = 20 if self.rerank_client else limit

        results = self.vectordb_client.search_by_vector(
            collection_name=self.collection_name,
            vector=vector,
            limit=retrieve_limit,
            filter=search_filter if search_filter else None
        )

        if not results:
            return None

        # rerank if available
        reranked_successfully = False
        if self.rerank_client and len(results) > 1:
            reranked = self.rerank_client.rerank(
                query=query,
                documents=[doc.text for doc in results],
                top_k=limit,
                model=self.rerank_model_id
            )
            if reranked:
                reranked_successfully = True
                reranked_docs = results
                results = []
                for idx, score in reranked:
                    doc = reranked_docs[idx]
                    doc.score = score
                    results.append(doc)

        # filter by score threshold
        if reranked_successfully:
            results = [doc for doc in results if doc.score >= 0.0]
        else:
            results = [doc for doc in results if doc.score >= self.score_threshold]

        if not results:
            return None

        return results[:limit]

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        vector = self.embedding_client.embed_text(text=text,
                                                document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        # retrieve more for reranking
        retrieve_limit = 20 if self.rerank_client else limit

        results = self.vectordb_client.search_by_vector(
            collection_name=self.collection_name,
            vector=vector,
            limit=retrieve_limit,
            filter={"project_id": project.project_id}
        )

        if not results:
            return False

        # rerank if available
        reranked_successfully = False
        if self.rerank_client and len(results) > 1:
            reranked = self.rerank_client.rerank(
                query=text,
                documents=[doc.text for doc in results],
                top_k=limit,
                model=self.rerank_model_id
            )
            if reranked:
                reranked_successfully = True
                reranked_docs = results
                results = []
                for idx, score in reranked:
                    doc = reranked_docs[idx]
                    doc.score = score
                    results.append(doc)

        # filter by score threshold
        if reranked_successfully:
            results = [doc for doc in results if doc.score >= 0.0]
        else:
            results = [doc for doc in results if doc.score >= self.score_threshold]

        if not results:
            return False

        return results[:limit]

    def generate_chat_answer(self, query: str, retrieved_documents: list, prior_history: list = None, system_prompt: str = None):

        if system_prompt is None:
            system_prompt = self.template_parser.get("rag", "system_prompt")

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        # prevent prompt truncation by limiting document budget
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
            for msg in prior_history[-6:]:
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

        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history

    def generate_related_questions(self, query: str, answer: str):

        related_prompt = self.template_parser.get("rag", "related_prompt", {
            "query": query,
            "answer": answer
        })

        result = self.generation_client.generate_text(
            prompt=related_prompt,
            chat_history=[]
        )

        if not result:
            return []

        questions = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                clean = line.lstrip("0123456789.-*() ")
                if clean:
                    questions.append(clean)

        return questions[:5]

    def delete_project_vectors(self, project_id: str):
        return self.vectordb_client.delete_by_filter(
            collection_name=self.collection_name,
            filter={"project_id": project_id}
        )

    def delete_file_vectors(self, asset_id: str):
        return self.vectordb_client.delete_by_filter(
            collection_name=self.collection_name,
            filter={"asset_id": asset_id}
        )