from abc import ABC, abstractmethod
from typing import List, Tuple


class RerankInterface(ABC):

    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = 5, model: str = None) -> List[Tuple[int, float]]:
        pass
