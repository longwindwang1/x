"""Tone-generator TTS stand-in.

Produces a soft tone whose duration tracks text length, pitched per voice id
so different characters are audibly distinct in the dev client. Chunked output
mimics a streaming synthesizer.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from ..characters import VoiceSpec
from . import BaseTTS

_CHUNK_MS = 40
_MS_PER_CHAR = 35
_MAX_MS = 3000


class MockTTS(BaseTTS):
    def __init__(self, sample_rate: int = 24000, chunk_delay_s: float = 0.0) -> None:
        self.sample_rate = sample_rate
        self.chunk_delay_s = chunk_delay_s

    async def synth(self, text: str, voice: VoiceSpec | None = None) -> AsyncIterator[bytes]:
        duration_ms = min(max(len(text) * _MS_PER_CHAR, 200), _MAX_MS)
        voice_id = (voice.tts_voice or "default") if voice else "default"
        freq = 180.0 + (hash(voice_id) % 12) * 20.0  # stable pitch per voice

        total = int(self.sample_rate * duration_ms / 1000.0)
        t = np.arange(total, dtype=np.float32) / self.sample_rate
        envelope = np.minimum(1.0, np.minimum(t / 0.02, (total / self.sample_rate - t) / 0.05))
        wave = (0.15 * envelope * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

        chunk_samples = int(self.sample_rate * _CHUNK_MS / 1000.0)
        for i in range(0, total, chunk_samples):
            if self.chunk_delay_s:
                await asyncio.sleep(self.chunk_delay_s)
            yield wave[i : i + chunk_samples].tobytes()
