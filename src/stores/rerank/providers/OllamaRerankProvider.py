import logging
from typing import List, Tuple
from ..RerankInterface import RerankInterface


class OllamaRerankProvider(RerankInterface):

    def __init__(self, api_url: str, api_key: str = None):
        api_url_clean = api_url.rstrip("/v1").rstrip("/") if api_url else "http://localhost:11434"
        self.api_url = api_url_clean
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

    def rerank(self, query: str, documents: List[str], top_k: int = 5, model: str = None) -> List[Tuple[int, float]]:

        if not documents or not model:
            return []

        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        try:
            import httpx
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.api_url}/api/rerank",
                    json=payload,
                )
                if response.status_code == 404:
                    self.logger.warning("Ollama rerank API not available (got 404)")
                    return []
                response.raise_for_status()
                result = response.json()
        except Exception as e:
            self.logger.warning(f"Ollama rerank failed: {e}")
            return []

        if not result or "results" not in result:
            return []

        return [
            (item["index"], item["relevance_score"])
            for item in result["results"]
        ]
