from abc import ABC, abstractmethod

from backend.app.models.domain import RetrievalResult


class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...
