from backend.app.core.logging import get_logger
from backend.app.core.telemetry import track_latency
from backend.app.models.domain import Chunk
from backend.app.providers.embeddings.base import EmbeddingProvider
from backend.app.providers.vector_store.base import VectorStore
from backend.app.services.chunker import chunk_naive
from backend.app.services.retrieval.bm25 import BM25Retriever
from backend.app.services.transcript_parser import parse_transcript

log = get_logger(__name__)


async def ingest_text(
    episode_id: str,
    raw_text: str,
    embedder: EmbeddingProvider,
    store: VectorStore,
    bm25: BM25Retriever | None = None,
) -> list[Chunk]:
    with track_latency("parse_transcript"):
        lines = parse_transcript(raw_text, episode_id)
    log.info("parsed_transcript", episode_id=episode_id, line_count=len(lines))

    with track_latency("chunk"):
        chunks = chunk_naive(lines, episode_id)
    log.info("chunked_transcript", episode_id=episode_id, chunk_count=len(chunks))

    with track_latency("embed"):
        embeddings = await embedder.embed([c.text for c in chunks])

    with track_latency("upsert"):
        await store.upsert(chunks, embeddings)
    log.info("ingested", episode_id=episode_id, chunks_stored=len(chunks))

    if bm25 is not None:
        bm25.add(chunks)
        log.info("bm25_updated", episode_id=episode_id)

    return chunks
