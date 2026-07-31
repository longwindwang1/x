"""Canned LLM for development and tests.

Echoes enough of the user's words to make the loop feel alive in the browser
client, and paces tokens with ``token_delay_s`` so streaming behavior
(chunking, barge-in cancellation) is exercised realistically.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from . import BaseLLM

_TEMPLATES = [
    'Ah, "{user}" — is that what you ask? A fine question, traveler. '
    "Few around here would dare say it aloud. Stay a while and I shall tell you more.",
    "Hm. {user} You are not the first to wonder about that. "
    "Keep your voice down and listen closely, friend.",
]


class MockLLM(BaseLLM):
    def __init__(self, token_delay_s: float = 0.0, replies: list[str] | None = None) -> None:
        self.token_delay_s = token_delay_s
        self.replies = replies
        self._n = 0

    async def stream_reply(
        self, messages: list[dict[str, str]], *, lora: str | None = None
    ) -> AsyncIterator[str]:
        user_text = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_text = msg["content"]
                break
        if self.replies:
            reply = self.replies[self._n % len(self.replies)]
        else:
            reply = _TEMPLATES[self._n % len(_TEMPLATES)].format(user=user_text.strip())
        self._n += 1

        for word in reply.split(" "):
            if self.token_delay_s:
                await asyncio.sleep(self.token_delay_s)
            yield word + " "
