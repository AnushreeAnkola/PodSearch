from openai import AsyncOpenAI

from backend.app.providers.embeddings.base import EmbeddingProvider

_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

BATCH_SIZE = 100


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dim = _MODEL_DIMS.get(model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = await self._client.embeddings.create(
                input=batch, model=self._model
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

    @property
    def dim(self) -> int:
        return self._dim
