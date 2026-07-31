"""Speech-to-text backends. Input is a complete PCM16 mono utterance."""

from __future__ import annotations


class BaseASR:
    sample_rate: int = 16000

    async def transcribe(self, pcm16: bytes) -> str:
        raise NotImplementedError
