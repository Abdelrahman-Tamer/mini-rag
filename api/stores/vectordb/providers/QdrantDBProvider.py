from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
import uuid
from typing import List, Optional, Any
from models.db_schemes import RetrievedDocument


class QdrantDBProvider(VectorDBInterface):

    def __init__(self, qdrant_db_url: str, qdrant_db_api_key: str, distance_method: str):

        self.client = None
        self.qdrant_db_url = qdrant_db_url
        self.qdrant_db_api_key = qdrant_db_api_key
        self.qdrant_db_distance_method = distance_method

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(url=self.qdrant_db_url, api_key=self.qdrant_db_api_key)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)

    def list_all_collections(self) -> List:
        return self.client.get_collections()

    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)

    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            return self.client.delete_collection(collection_name=collection_name)

    def create_collection(self, collection_name: str,
                                embedding_size: int,
                                do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)

        if not self.is_collection_existed(collection_name):
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )

            for field in ["project_id", "city", "doc_type", "asset_id", "generation_id", "is_active"]:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=f"metadata.{field}",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as e:
                    self.logger.warning(f"Could not create index on metadata.{field}: {e}")

            return True

        return False

    def _build_filter(self, filter: Optional[dict] = None, exclude_inactive: bool = False) -> Optional[models.Filter]:
        must_conditions = []
        must_not_conditions = []

        if filter:
            for key, value in filter.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value)
                    )
                )

        if exclude_inactive:
            must_not_conditions.append(
                models.FieldCondition(
                    key="metadata.is_active",
                    match=models.MatchValue(value=False)
                )
            )

        if not must_conditions and not must_not_conditions:
            return None

        return models.Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None
        )

    def insert_one(self, collection_name: str, text: str, vector: list,
                        metadata: dict = None,
                        record_id: str = None):

        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False

        point_id = int(record_id) if str(record_id).isdigit() else record_id
        try:
            _ = self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": text, "metadata": metadata
                        }
                    )
                ]
            )
        except Exception as e:
            self.logger.error(f"Error while inserting record: {e}")
            return False

        return True

    def insert_many(self, collection_name: str, texts: list,
                        vectors: list, metadata: list = None,
                        record_ids: list = None, batch_size: int = 50):

        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = [str(uuid.uuid4()) for _ in texts]
        else:
            record_ids = [
                int(rid) if str(rid).isdigit() else str(rid)
                for rid in record_ids
            ]

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_records = [
                models.Record(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={
                        "text": batch_texts[x], "metadata": batch_metadata[x]
                    }
                )

                for x in range(len(batch_texts))
            ]

            try:
                _ = self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records,
                )
            except Exception as e:
                self.logger.error(f"Error while inserting batch: {e}")
                return False

        return True

    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5, filter: Optional[dict] = None):
        if not self.is_collection_existed(collection_name):
            return None

        query_filter = self._build_filter(filter, exclude_inactive=True)

        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    limit=limit,
                    query_filter=query_filter,
                )
                results = response.points if response else []
            else:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    limit=limit,
                    query_filter=query_filter,
                )
        except Exception as e:
            self.logger.error(f"Error during vector search: {e}")
            return None

        if not results or len(results) == 0:
            return None
        return [
            RetrievedDocument(**{
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {}),
            })
            for result in results
        ]

    def delete_by_filter(self, collection_name: str, filter: dict):
        if not self.is_collection_existed(collection_name):
            return False

        query_filter = self._build_filter(filter)
        if not query_filter:
            return False

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=query_filter
                )
            )
            return True
        except Exception as e:
            self.logger.error(f"Error during delete_by_filter: {e}")
            return False

    def delete_by_id(self, collection_name: str, record_id: Any):
        if not self.is_collection_existed(collection_name):
            return False
        point_id = int(record_id) if str(record_id).isdigit() else record_id
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            return True
        except Exception as e:
            self.logger.error(f"Error during delete_by_id: {e}")
            return False

    def delete_by_ids(self, collection_name: str, record_ids: list) -> bool:
        if not self.is_collection_existed(collection_name) or not record_ids:
            return False
        point_ids = [int(rid) if str(rid).isdigit() else str(rid) for rid in record_ids]
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            return True
        except Exception as e:
            self.logger.error(f"Error during delete_by_ids: {e}")
            return False

    def set_payload(self, collection_name: str, payload: dict, record_ids: list = None) -> bool:
        if not self.is_collection_existed(collection_name) or not record_ids:
            return False
        point_ids = [int(rid) if str(rid).isdigit() else str(rid) for rid in record_ids]
        try:
            self.client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=point_ids
            )
            return True
        except Exception as e:
            self.logger.error(f"Error during set_payload: {e}")
            return False
