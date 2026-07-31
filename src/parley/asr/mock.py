"""Deterministic ASR stand-in for development and tests."""

from __future__ import annotations

import asyncio

from . import BaseASR

_DEFAULT_LINES = [
    "Hello there, who are you?",
    "What do you sell in this place?",
    "Have you heard any rumors lately?",
]


class MockASR(BaseASR):
    def __init__(self, lines: list[str] | None = None, delay_s: float = 0.0) -> None:
        self.lines = lines or _DEFAULT_LINES
        self.delay_s = delay_s
        self._n = 0

    async def transcribe(self, pcm16: bytes) -> str:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        text = self.lines[self._n % len(self.lines)]
        self._n += 1
        return text
