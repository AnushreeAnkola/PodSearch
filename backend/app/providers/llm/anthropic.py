import anthropic

from backend.app.models.domain import LLMResponse
from backend.app.providers.llm.base import LLMProvider


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 1024
    ) -> LLMResponse:
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return LLMResponse(
            text=message.content[0].text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
