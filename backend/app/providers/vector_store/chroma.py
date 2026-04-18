import chromadb

from backend.app.models.domain import Chunk, RetrievalResult
from backend.app.providers.vector_store.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "episode_id": c.episode_id,
                    "start_seconds": c.start_seconds,
                    "end_seconds": c.end_seconds,
                }
                for c in chunks
            ],
        )

    async def query(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievalResult]:
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results: list[RetrievalResult] = []
        if not results["ids"] or not results["ids"][0]:
            return retrieval_results

        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            chunk = Chunk(
                chunk_id=chunk_id,
                text=results["documents"][0][i],
                episode_id=meta["episode_id"],
                start_seconds=meta["start_seconds"],
                end_seconds=meta["end_seconds"],
            )
            distance = results["distances"][0][i]
            score = 1.0 - distance
            retrieval_results.append(RetrievalResult(chunk=chunk, score=score))

        return retrieval_results

    async def count(self) -> int:
        return self._collection.count()

    async def delete(self, chunk_ids: list[str]) -> None:
        self._collection.delete(ids=chunk_ids)
