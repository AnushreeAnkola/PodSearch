import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models.domain import Chunk, LLMResponse, RetrievalResult
from backend.app.providers.embeddings.base import EmbeddingProvider
from backend.app.providers.llm.base import LLMProvider
from backend.app.providers.vector_store.base import VectorStore
from backend.app.services.retrieval.semantic import SemanticRetriever


class FakeEmbedder(EmbeddingProvider):
    async def embed(self, texts):
        return [[0.1] * 3 for _ in texts]

    async def embed_query(self, text):
        return [0.1] * 3

    @property
    def dim(self):
        return 3


class FakeStore(VectorStore):
    def __init__(self):
        self._chunks: dict[str, tuple[Chunk, list[float]]] = {}

    async def upsert(self, chunks, embeddings):
        for c, e in zip(chunks, embeddings):
            self._chunks[c.chunk_id] = (c, e)

    async def query(self, embedding, top_k=5):
        results = []
        for chunk, _ in list(self._chunks.values())[:top_k]:
            results.append(RetrievalResult(chunk=chunk, score=0.9))
        return results

    async def count(self):
        return len(self._chunks)

    async def delete(self, chunk_ids):
        for cid in chunk_ids:
            self._chunks.pop(cid, None)


class FakeLLM(LLMProvider):
    async def complete(self, system_prompt, user_prompt, max_tokens=1024):
        return LLMResponse(
            text="RAG stands for Retrieval Augmented Generation [00:00].",
            input_tokens=100,
            output_tokens=20,
        )


def _build_test_app():
    from fastapi import FastAPI
    from backend.app.api import health, ingest, search

    embedder = FakeEmbedder()
    store = FakeStore()
    llm = FakeLLM()

    app = FastAPI()
    app.state.settings = type("S", (), {"llm_model": "fake-model"})()
    app.state.embedder = embedder
    app.state.store = store
    app.state.llm = llm
    app.state.retriever = SemanticRetriever(embedder, store)

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(search.router)
    return app


@pytest.mark.asyncio
async def test_ingest_and_search():
    app = _build_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Ingest
        resp = await client.post(
            "/ingest",
            json={
                "episode_id": "ep_test",
                "text": "[00:00] Hello world\n[00:15] RAG is cool\n",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["episode_id"] == "ep_test"
        assert data["chunks_created"] == 2

        # Health
        resp = await client.get("/health")
        assert resp.status_code == 200

        # Ready
        resp = await client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["chunks_indexed"] == 2

        # Search
        resp = await client.post("/search", json={"query": "what is RAG?"})
        assert resp.status_code == 200
        result = resp.json()
        assert "answer" in result
        assert "citations" in result
        assert result["model"] == "fake-model"
