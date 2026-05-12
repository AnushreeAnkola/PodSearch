import asyncio

from backend.app.models.domain import RetrievalResult
from backend.app.services.retrieval.base import Retriever


class HybridRetriever(Retriever):
    def __init__(
        self,
        semantic: Retriever,
        bm25: Retriever,
        k_rrf: int = 60,
        fetch_k: int | None = None,
    ) -> None:
        self._semantic = semantic
        self._bm25 = bm25
        self._k_rrf = k_rrf
        self._fetch_k = fetch_k

    def _per_child_fetch_k(self, top_k: int) -> int:
        if self._fetch_k is not None:
            return max(self._fetch_k, top_k)
        return max(50, top_k * 10)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        fetch_k = self._per_child_fetch_k(top_k)
        semantic_task = self._semantic.retrieve(query, top_k=fetch_k, filters=filters)
        bm25_task = self._bm25.retrieve(query, top_k=fetch_k, filters=filters)
        semantic_results, bm25_results = await asyncio.gather(semantic_task, bm25_task)
        fused = self._rrf_fuse([semantic_results, bm25_results])
        return fused[:top_k]

    def _rrf_fuse(
        self, ranked_lists: list[list[RetrievalResult]]
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, RetrievalResult] = {}
        for results in ranked_lists:
            for rank, result in enumerate(results):
                chunk_id = result.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                    self._k_rrf + rank + 1
                )
                if chunk_id not in chunk_by_id:
                    chunk_by_id[chunk_id] = result
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            RetrievalResult(chunk=chunk_by_id[cid].chunk, score=score)
            for cid, score in ordered
        ]
