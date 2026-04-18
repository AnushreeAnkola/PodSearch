from fastapi import Depends, Request

from backend.app.core.config import Settings, get_settings
from backend.app.providers.embeddings.base import EmbeddingProvider
from backend.app.providers.llm.base import LLMProvider
from backend.app.providers.vector_store.base import VectorStore
from backend.app.services.retrieval.base import Retriever


def get_embedder(request: Request) -> EmbeddingProvider:
    return request.app.state.embedder


def get_store(request: Request) -> VectorStore:
    return request.app.state.store


def get_llm(request: Request) -> LLMProvider:
    return request.app.state.llm


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
