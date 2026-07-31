"""Client for any OpenAI-compatible chat-completions endpoint.

Primary target is a vLLM server. Multi-LoRA routing works by passing the
character's adapter name as the ``model`` field of the request — vLLM resolves
registered adapters by name, so N characters share one base model deployment
(see docs/roadmap.md for the serving setup).
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from . import BaseLLM


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.8,
        max_tokens: int = 220,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def stream_reply(
        self, messages: list[dict[str, str]], *, lora: str | None = None
    ) -> AsyncIterator[str]:
        payload = {
            "model": lora or self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        async with self._client.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def aclose(self) -> None:
        await self._client.aclose()
