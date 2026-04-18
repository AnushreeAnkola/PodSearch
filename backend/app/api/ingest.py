from fastapi import APIRouter, Depends

from backend.app.api.deps import get_embedder, get_store
from backend.app.models.api import IngestRequest, IngestResponse
from backend.app.providers.embeddings.base import EmbeddingProvider
from backend.app.providers.vector_store.base import VectorStore
from backend.app.services.ingestion import ingest_text

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    embedder: EmbeddingProvider = Depends(get_embedder),
    store: VectorStore = Depends(get_store),
):
    chunks = await ingest_text(
        episode_id=body.episode_id,
        raw_text=body.text,
        embedder=embedder,
        store=store,
    )
    return IngestResponse(episode_id=body.episode_id, chunks_created=len(chunks))
