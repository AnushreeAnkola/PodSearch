from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # API
    environment: str = "development"
    port: int = 8000
    log_level: str = "INFO"

    # Embedding provider
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: str = ""

    # LLM provider
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str = ""

    # Vector store
    vector_store_type: str = "chroma"
    chroma_db_path: str = "./data/chroma_db"
    chroma_collection: str = "podcast_chunks"

    # Retrieval
    top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
