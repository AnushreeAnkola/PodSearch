from fastapi import APIRouter, Depends

from backend.app.api.deps import get_store
from backend.app.providers.vector_store.base import VectorStore

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(store: VectorStore = Depends(get_store)):
    count = await store.count()
    return {"status": "ready", "chunks_indexed": count}
