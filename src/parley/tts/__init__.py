"""Text-to-speech backends.

``synth`` yields PCM16 mono chunks at ``sample_rate``. One call per sentence
chunk (see chunker.py); streaming chunk-by-chunk keeps time-to-first-audio low
even for backends that synthesize a sentence at once.
"""

from __future__ import annotations

from typing import AsyncIterator

from ..characters import VoiceSpec


class BaseTTS:
    sample_rate: int = 24000

    def synth(self, text: str, voice: VoiceSpec | None = None) -> AsyncIterator[bytes]:
        raise NotImplementedError
