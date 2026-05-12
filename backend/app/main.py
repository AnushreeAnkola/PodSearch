from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api import health, ingest, search
from backend.app.core.config import get_settings
from backend.app.core.logging import RequestIDMiddleware, configure_logging, get_logger
from backend.app.providers.factory import (
    build_embedding_provider,
    build_llm_provider,
    build_vector_store,
)
from backend.app.services.retrieval.bm25 import BM25Retriever
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.retrieval.semantic import SemanticRetriever

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    log.info("startup", environment=settings.environment)

    app.state.settings = settings
    app.state.embedder = build_embedding_provider(settings)
    app.state.store = build_vector_store(settings)
    app.state.llm = build_llm_provider(settings)

    existing_chunks = await app.state.store.fetch_all()
    semantic = SemanticRetriever(app.state.embedder, app.state.store)
    bm25 = BM25Retriever(existing_chunks)
    hybrid = HybridRetriever(semantic, bm25, k_rrf=settings.rrf_k)

    app.state.bm25 = bm25
    app.state.retrievers = {
        "semantic": semantic,
        "bm25": bm25,
        "hybrid": hybrid,
    }

    log.info("providers_ready", indexed_chunks=len(existing_chunks))
    yield
    log.info("shutdown")


app = FastAPI(title="PodSearch", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(ingest.router)
