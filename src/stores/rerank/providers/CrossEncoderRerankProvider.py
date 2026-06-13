import logging
from typing import List, Tuple
from ..RerankInterface import RerankInterface


class CrossEncoderRerankProvider(RerankInterface):

    def __init__(self, api_url: str = None, api_key: str = None):
        self.logger = logging.getLogger(__name__)
        self._model = None

    def _get_model(self, model_id: str):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self.logger.info(f"Loading cross-encoder model: {model_id}")
            self._model = CrossEncoder(model_id)
        return self._model

    def rerank(self, query: str, documents: List[str], top_k: int = 5, model: str = None) -> List[Tuple[int, float]]:

        if not documents:
            return []

        model_id = model or "BAAI/bge-reranker-v2-m3"
        cross_encoder = self._get_model(model_id)

        try:
            pairs = [[query, doc] for doc in documents]
            scores = cross_encoder.predict(pairs, show_progress_bar=False)
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return [(int(idx), float(score)) for idx, score in indexed[:top_k]]
        except Exception as e:
            self.logger.warning(f"Cross-encoder rerank failed: {e}")
            return []
