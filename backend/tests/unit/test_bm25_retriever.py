import pytest

from backend.app.models.domain import Chunk
from backend.app.services.retrieval.bm25 import BM25Retriever, _tokenize


def _chunk(chunk_id: str, text: str, episode_id: str = "ep1", start: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        episode_id=episode_id,
        start_seconds=start,
        end_seconds=start + 10,
    )


def test_tokenize_strips_timestamp_prefix():
    assert _tokenize("[00:15] Hello world!") == ["hello", "world"]
    assert _tokenize("  [12:34]   Mixed CASE") == ["mixed", "case"]


def test_tokenize_keeps_alphanumerics():
    assert _tokenize("RAG2 hybrid-search") == ["rag2", "hybrid", "search"]


@pytest.mark.asyncio
async def test_empty_corpus_returns_no_results():
    retriever = BM25Retriever()
    assert await retriever.retrieve("anything") == []

    retriever_with_empty_list = BM25Retriever([])
    assert await retriever_with_empty_list.retrieve("anything") == []


@pytest.mark.asyncio
async def test_relevance_ordering_on_handcrafted_corpus():
    chunks = [
        _chunk("c0", "[00:00] The weather is sunny today"),
        _chunk("c1", "[00:10] We discussed retrieval augmented generation in detail"),
        _chunk("c2", "[00:20] Lunch break and coffee"),
        _chunk("c3", "[00:30] Hybrid retrieval combines BM25 and vector search"),
        _chunk("c4", "[00:40] Vector embeddings capture semantic similarity"),
    ]
    retriever = BM25Retriever(chunks)
    results = await retriever.retrieve("hybrid retrieval BM25", top_k=3)
    assert len(results) == 3
    assert results[0].chunk.chunk_id == "c3"
    returned_ids = {r.chunk.chunk_id for r in results}
    assert "c2" not in returned_ids


@pytest.mark.asyncio
async def test_add_updates_index():
    retriever = BM25Retriever(
        [
            _chunk("c0", "[00:00] hello world"),
            _chunk("c1", "[00:10] coffee break"),
            _chunk("c2", "[00:20] sunny weather"),
        ]
    )
    retriever.add([_chunk("c_new", "[00:30] retrieval augmented generation")])
    results = await retriever.retrieve("retrieval", top_k=5)
    assert results
    assert results[0].chunk.chunk_id == "c_new"


@pytest.mark.asyncio
async def test_episode_id_filter_excludes_other_episodes():
    chunks = [
        _chunk("a0", "[00:00] retrieval is great", episode_id="epA"),
        _chunk("b0", "[00:00] retrieval is great", episode_id="epB"),
    ]
    retriever = BM25Retriever(chunks)
    results = await retriever.retrieve(
        "retrieval", top_k=5, filters={"episode_id": "epA"}
    )
    assert len(results) == 1
    assert results[0].chunk.episode_id == "epA"


@pytest.mark.asyncio
async def test_start_seconds_range_filter():
    chunks = [
        _chunk("c0", "[00:00] retrieval rules", start=0),
        _chunk("c1", "[01:00] retrieval rules", start=60),
        _chunk("c2", "[02:00] retrieval rules", start=120),
    ]
    retriever = BM25Retriever(chunks)
    results = await retriever.retrieve(
        "retrieval", top_k=5, filters={"start_seconds__gte": 60}
    )
    returned_starts = sorted(r.chunk.start_seconds for r in results)
    assert returned_starts == [60, 120]


@pytest.mark.asyncio
async def test_query_with_no_tokens_returns_empty():
    retriever = BM25Retriever([_chunk("c0", "[00:00] content")])
    assert await retriever.retrieve("???") == []
