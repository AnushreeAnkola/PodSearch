from abc import ABC, abstractmethod

from backend.app.models.domain import Chunk, RetrievalResult


class VectorStore(ABC):
    @abstractmethod
    async def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        ...

    @abstractmethod
    async def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> None:
        ...

    @abstractmethod
    async def fetch_all(self) -> list[Chunk]:
        ...
