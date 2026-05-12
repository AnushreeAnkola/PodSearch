import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models.domain import Chunk, LLMResponse, RetrievalResult
from backend.app.providers.embeddings.base import EmbeddingProvider
from backend.app.providers.llm.base import LLMProvider
from backend.app.providers.vector_store.base import VectorStore
from backend.app.services.retrieval.bm25 import BM25Retriever
from backend.app.services.retrieval.hybrid import HybridRetriever
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

    async def query(self, embedding, top_k=5, filters=None):
        results = []
        for chunk, _ in self._chunks.values():
            if filters and not _match(chunk, filters):
                continue
            results.append(RetrievalResult(chunk=chunk, score=0.9))
            if len(results) >= top_k:
                break
        return results

    async def count(self):
        return len(self._chunks)

    async def delete(self, chunk_ids):
        for cid in chunk_ids:
            self._chunks.pop(cid, None)

    async def fetch_all(self):
        return [c for c, _ in self._chunks.values()]


def _match(chunk: Chunk, filters: dict) -> bool:
    for key, value in filters.items():
        if "__" in key:
            continue
        if getattr(chunk, key, None) != value:
            return False
    return True


class FakeLLM(LLMProvider):
    async def complete(self, system_prompt, user_prompt, max_tokens=1024):
        return LLMResponse(
            text="RAG stands for Retrieval Augmented Generation [00:00].",
            input_tokens=100,
            output_tokens=20,
        )


def _build_test_app(retriever_mode: str = "hybrid"):
    from fastapi import FastAPI
    from backend.app.api import health, ingest, search

    embedder = FakeEmbedder()
    store = FakeStore()
    llm = FakeLLM()
    bm25 = BM25Retriever()
    semantic = SemanticRetriever(embedder, store)
    hybrid = HybridRetriever(semantic, bm25, k_rrf=60)

    app = FastAPI()
    app.state.settings = type(
        "S", (), {"llm_model": "fake-model", "retriever_mode": retriever_mode}
    )()
    app.state.embedder = embedder
    app.state.store = store
    app.state.llm = llm
    app.state.bm25 = bm25
    app.state.retrievers = {"semantic": semantic, "bm25": bm25, "hybrid": hybrid}

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(search.router)
    return app


@pytest.mark.asyncio
async def test_ingest_and_search_default_mode():
    app = _build_test_app(retriever_mode="hybrid")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
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

        resp = await client.get("/health")
        assert resp.status_code == 200

        resp = await client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["chunks_indexed"] == 2

        resp = await client.post("/search", json={"query": "what is RAG?"})
        assert resp.status_code == 200
        result = resp.json()
        assert "answer" in result
        assert "citations" in result
        assert result["model"] == "fake-model"


@pytest.mark.asyncio
async def test_search_with_explicit_modes():
    app = _build_test_app(retriever_mode="semantic")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/ingest",
            json={
                "episode_id": "ep_test",
                "text": (
                    "[00:00] Hello world\n"
                    "[00:15] retrieval augmented generation\n"
                    "[00:30] hybrid search combines BM25 and vectors\n"
                    "[00:45] vector embeddings capture meaning\n"
                ),
            },
        )

        for mode in ("semantic", "bm25", "hybrid"):
            resp = await client.post(
                "/search",
                json={"query": "retrieval augmented generation", "retriever_mode": mode},
            )
            assert resp.status_code == 200, f"{mode} failed: {resp.text}"
            assert resp.json()["model"] == "fake-model"


@pytest.mark.asyncio
async def test_search_with_episode_filter():
    app = _build_test_app(retriever_mode="semantic")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/ingest",
            json={"episode_id": "ep_a", "text": "[00:00] alpha content\n[00:15] more alpha\n"},
        )
        await client.post(
            "/ingest",
            json={"episode_id": "ep_b", "text": "[00:00] beta content\n[00:15] more beta\n"},
        )

        resp = await client.post(
            "/search",
            json={
                "query": "content",
                "retriever_mode": "semantic",
                "filters": {"episode_id": "ep_a"},
            },
        )
        assert resp.status_code == 200
        for citation in resp.json()["citations"]:
            assert citation["episode_id"] == "ep_a"


@pytest.mark.asyncio
async def test_invalid_retriever_mode_returns_422():
    app = _build_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/search", json={"query": "anything", "retriever_mode": "made_up"}
        )
        assert resp.status_code == 422
