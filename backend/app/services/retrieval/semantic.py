from backend.app.models.domain import RetrievalResult
from backend.app.providers.embeddings.base import EmbeddingProvider
from backend.app.providers.vector_store.base import VectorStore
from backend.app.services.retrieval.base import Retriever


class SemanticRetriever(Retriever):
    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_embedding = await self._embedder.embed_query(query)
        return await self._store.query(query_embedding, top_k=top_k)
