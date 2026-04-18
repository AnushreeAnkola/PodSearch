from abc import ABC, abstractmethod

from backend.app.models.domain import LLMResponse


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 1024
    ) -> LLMResponse:
        ...
