from fastapi import Request

from backend.app.core.config import Settings
from backend.app.providers.embeddings.base import EmbeddingProvider
from backend.app.providers.llm.base import LLMProvider
from backend.app.providers.vector_store.base import VectorStore
from backend.app.services.retrieval.base import Retriever
from backend.app.services.retrieval.bm25 import BM25Retriever


def get_embedder(request: Request) -> EmbeddingProvider:
    return request.app.state.embedder


def get_store(request: Request) -> VectorStore:
    return request.app.state.store


def get_llm(request: Request) -> LLMProvider:
    return request.app.state.llm


def get_retrievers(request: Request) -> dict[str, Retriever]:
    return request.app.state.retrievers


def get_bm25(request: Request) -> BM25Retriever:
    return request.app.state.bm25


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
