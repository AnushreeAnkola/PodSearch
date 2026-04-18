import pytest

from backend.app.models.domain import Chunk, RetrievalResult
from backend.app.providers.embeddings.base import EmbeddingProvider
from backend.app.providers.vector_store.base import VectorStore
from backend.app.services.retrieval.semantic import SemanticRetriever


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    @property
    def dim(self) -> int:
        return 3


class FakeVectorStore(VectorStore):
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    async def upsert(self, chunks, embeddings) -> None:
        pass

    async def query(self, embedding, top_k=5) -> list[RetrievalResult]:
        return self._results[:top_k]

    async def count(self) -> int:
        return len(self._results)

    async def delete(self, chunk_ids) -> None:
        pass


@pytest.mark.asyncio
async def test_semantic_retriever_returns_results():
    fake_results = [
        RetrievalResult(
            chunk=Chunk(
                chunk_id="c1",
                text="[00:00] Hello",
                episode_id="ep1",
                start_seconds=0,
                end_seconds=15,
            ),
            score=0.95,
        ),
    ]
    retriever = SemanticRetriever(FakeEmbeddingProvider(), FakeVectorStore(fake_results))
    results = await retriever.retrieve("hello", top_k=5)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score == 0.95


@pytest.mark.asyncio
async def test_semantic_retriever_respects_top_k():
    fake_results = [
        RetrievalResult(
            chunk=Chunk(
                chunk_id=f"c{i}",
                text=f"chunk {i}",
                episode_id="ep1",
                start_seconds=i * 10,
                end_seconds=(i + 1) * 10,
            ),
            score=0.9 - i * 0.1,
        )
        for i in range(5)
    ]
    retriever = SemanticRetriever(FakeEmbeddingProvider(), FakeVectorStore(fake_results))
    results = await retriever.retrieve("test", top_k=2)
    assert len(results) == 2
