"""Dialogue-generation backends.

``stream_reply`` yields text deltas. ``lora`` is the character's adapter name;
backends that support multi-LoRA serving (vLLM) route on it, others ignore it.
"""

from __future__ import annotations

from typing import AsyncIterator


class BaseLLM:
    def stream_reply(
        self, messages: list[dict[str, str]], *, lora: str | None = None
    ) -> AsyncIterator[str]:
        raise NotImplementedError
