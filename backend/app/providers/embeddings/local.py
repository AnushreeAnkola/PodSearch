import asyncio

from sentence_transformers import SentenceTransformer

from backend.app.providers.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._model = SentenceTransformer(model)
        self._dim = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(
            self._model.encode, texts, convert_to_numpy=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

    @property
    def dim(self) -> int:
        return self._dim
