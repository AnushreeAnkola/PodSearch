import asyncio
import re
from operator import itemgetter

from rank_bm25 import BM25Okapi

from backend.app.models.domain import Chunk, RetrievalResult
from backend.app.services.retrieval.base import Retriever


_TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\[\d{1,2}:\d{2}\]\s*")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    stripped = _TIMESTAMP_PREFIX_RE.sub("", text)
    return _TOKEN_RE.findall(stripped.lower())


_RANGE_OPS = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "ne": lambda a, b: a != b,
}


def _match_filters(chunk: Chunk, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        if "__" in key:
            field, op = key.rsplit("__", 1)
            cmp = _RANGE_OPS.get(op)
            if cmp is None:
                raise ValueError(f"Unsupported filter operator: {op}")
            actual = getattr(chunk, field, None)
            if actual is None or not cmp(actual, value):
                return False
        else:
            if getattr(chunk, key, None) != value:
                return False
    return True


class BM25Retriever(Retriever):
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        if chunks:
            self.rebuild(chunks)

    def rebuild(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokenized = [_tokenize(c.text) for c in self._chunks]
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.rebuild(self._chunks + list(chunks))

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = await asyncio.to_thread(self._bm25.get_scores, tokens)
        scored = [
            (chunk, float(score))
            for chunk, score in zip(self._chunks, scores)
            if _match_filters(chunk, filters)
        ]
        scored.sort(key=itemgetter(1), reverse=True)
        return [
            RetrievalResult(chunk=chunk, score=score)
            for chunk, score in scored[:top_k]
        ]
