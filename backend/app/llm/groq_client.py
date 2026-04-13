from __future__ import annotations

from typing import AsyncIterator

from groq import AsyncGroq

from app.core.config import get_settings


class GroqLLMClient:
    """Thin async wrapper over Groq chat completions."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        self.client = AsyncGroq(api_key=self.settings.groq_api_key)

    async def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a complete answer in a single response."""
        response = await self.client.chat.completions.create(
            model=self.settings.groq_model_name,
            temperature=self.settings.groq_temperature,
            max_completion_tokens=self.settings.groq_max_completion_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty response.")
        return content

    async def stream_answer(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream partial answer tokens from Groq."""
        stream = await self.client.chat.completions.create(
            model=self.settings.groq_model_name,
            temperature=self.settings.groq_temperature,
            max_completion_tokens=self.settings.groq_max_completion_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
