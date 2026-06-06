from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Optional
import json


class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client,
                embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.collection_name = self.app_settings.VECTOR_DB_COLLECTION_NAME

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

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        vector = self.embedding_client.embed_text(text=text,
                                                document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        results = self.vectordb_client.search_by_vector(
            collection_name=self.collection_name,
            vector=vector,
            limit=limit,
            filter={"project_id": project.project_id}
        )

        if not results:
            return False

        return results

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

        vector = self.embedding_client.embed_text(text=query,
                                                document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return None

        search_filter = {}
        if city:
            search_filter["city"] = city
        if doc_type:
            search_filter["doc_type"] = doc_type

        results = self.vectordb_client.search_by_vector(
            collection_name=self.collection_name,
            vector=vector,
            limit=limit,
            filter=search_filter if search_filter else None
        )

        if not results:
            return None

        return results

    def generate_chat_answer(self, query: str, retrieved_documents: list):

        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": doc.text,
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

    def update_asset_payload(self, asset_id: str, city: str, doc_type: str = None):
        old_filter = {"asset_id": asset_id}

        update_filter = {"asset_id": asset_id}
        if city:
            update_filter["city"] = city
        if doc_type:
            update_filter["doc_type"] = doc_type

        points = self.vectordb_client.search_by_vector(
            collection_name=self.collection_name,
            vector=[0.0] * self.embedding_client.embedding_size,
            limit=100,
            filter=old_filter
        )

        if points:
            _ = self.vectordb_client.delete_by_filter(
                collection_name=self.collection_name,
                filter=old_filter
            )
            return True

        return True