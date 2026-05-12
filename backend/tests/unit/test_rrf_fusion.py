import pytest

from backend.app.models.domain import Chunk, RetrievalResult
from backend.app.services.retrieval.base import Retriever
from backend.app.services.retrieval.hybrid import HybridRetriever


def _result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=f"text-{chunk_id}",
            episode_id="ep1",
            start_seconds=0,
            end_seconds=10,
        ),
        score=score,
    )


class FakeRetriever(Retriever):
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.last_filters: dict | None = None
        self.last_top_k: int | None = None

    async def retrieve(self, query, top_k=5, filters=None):
        self.last_filters = filters
        self.last_top_k = top_k
        return self._results[:top_k]


def _ids(results: list[RetrievalResult]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


@pytest.mark.asyncio
async def test_identical_lists_preserve_order():
    a = [_result("c0"), _result("c1"), _result("c2")]
    b = [_result("c0"), _result("c1"), _result("c2")]
    hybrid = HybridRetriever(FakeRetriever(a), FakeRetriever(b), k_rrf=60)
    fused = await hybrid.retrieve("q", top_k=3)
    assert _ids(fused) == ["c0", "c1", "c2"]


@pytest.mark.asyncio
async def test_disjoint_lists_both_appear():
    a = [_result("a0"), _result("a1")]
    b = [_result("b0"), _result("b1")]
    hybrid = HybridRetriever(FakeRetriever(a), FakeRetriever(b), k_rrf=60)
    fused = await hybrid.retrieve("q", top_k=4)
    assert set(_ids(fused)) == {"a0", "a1", "b0", "b1"}


@pytest.mark.asyncio
async def test_chunk_in_both_outranks_chunk_in_one():
    a = [_result("shared"), _result("a_only"), _result("a_extra")]
    b = [_result("b_only"), _result("shared"), _result("b_extra")]
    hybrid = HybridRetriever(FakeRetriever(a), FakeRetriever(b), k_rrf=60)
    fused = await hybrid.retrieve("q", top_k=10)
    assert fused[0].chunk.chunk_id == "shared"


@pytest.mark.asyncio
async def test_respects_top_k():
    a = [_result(f"c{i}") for i in range(10)]
    b = [_result(f"d{i}") for i in range(10)]
    hybrid = HybridRetriever(FakeRetriever(a), FakeRetriever(b), k_rrf=60)
    fused = await hybrid.retrieve("q", top_k=3)
    assert len(fused) == 3


@pytest.mark.asyncio
async def test_filters_pushed_to_children():
    a = FakeRetriever([_result("c0")])
    b = FakeRetriever([_result("c0")])
    hybrid = HybridRetriever(a, b, k_rrf=60)
    await hybrid.retrieve("q", top_k=3, filters={"episode_id": "ep_x"})
    assert a.last_filters == {"episode_id": "ep_x"}
    assert b.last_filters == {"episode_id": "ep_x"}


@pytest.mark.asyncio
async def test_fetch_k_is_at_least_50():
    a = FakeRetriever([_result(f"c{i}") for i in range(100)])
    b = FakeRetriever([_result(f"d{i}") for i in range(100)])
    hybrid = HybridRetriever(a, b, k_rrf=60)
    await hybrid.retrieve("q", top_k=5)
    assert a.last_top_k >= 50
    assert b.last_top_k >= 50
