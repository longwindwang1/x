"""Kokoro-82M TTS backend (optional; ``pip install "parley[tts]"``).

Small enough to co-locate with the LLM on one GPU and fast enough for
sub-second first audio; the low-latency default while CosyVoice (voice
cloning + instruct emotion control) lands as the character-voice backend.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np
from kokoro import KPipeline

from ..characters import VoiceSpec
from . import BaseTTS


class KokoroTTS(BaseTTS):
    sample_rate = 24000

    def __init__(self, lang_code: str = "a", default_voice: str = "af_heart") -> None:
        self._pipeline = KPipeline(lang_code=lang_code)
        self.default_voice = default_voice

    async def synth(self, text: str, voice: VoiceSpec | None = None) -> AsyncIterator[bytes]:
        voice_id = (voice.tts_voice if voice and voice.tts_voice else self.default_voice)
        segments = await asyncio.to_thread(self._synth_sync, text, voice_id)
        for pcm in segments:
            yield pcm

    def _synth_sync(self, text: str, voice_id: str) -> list[bytes]:
        out: list[bytes] = []
        for _gs, _ps, audio in self._pipeline(text, voice=voice_id):
            samples = np.asarray(audio, dtype=np.float32)
            out.append((samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
        return out
