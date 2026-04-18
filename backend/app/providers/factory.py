from backend.app.core.config import Settings
from backend.app.providers.embeddings.base import EmbeddingProvider
from backend.app.providers.llm.base import LLMProvider
from backend.app.providers.vector_store.base import VectorStore


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        from backend.app.providers.embeddings.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store_type == "chroma":
        from backend.app.providers.vector_store.chroma import ChromaVectorStore

        return ChromaVectorStore(
            persist_dir=settings.chroma_db_path,
            collection_name=settings.chroma_collection,
        )
    raise ValueError(f"Unknown vector store type: {settings.vector_store_type}")


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic":
        from backend.app.providers.llm.anthropic import AnthropicLLMProvider

        return AnthropicLLMProvider(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
