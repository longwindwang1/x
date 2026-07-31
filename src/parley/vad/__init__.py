"""Voice activity detection backends.

A VAD consumes raw PCM16 frames and emits events; ``SPEECH_END`` carries the
complete utterance (including pre-roll) ready for ASR. The interface is
synchronous because frame handling sits on the hot audio path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VADEventType(Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VADEvent:
    type: VADEventType
    utterance: bytes | None = None  # populated on SPEECH_END


class BaseVAD:
    sample_rate: int = 16000

    def process(self, frame: bytes) -> list[VADEvent]:
        raise NotImplementedError

    def flush(self) -> list[VADEvent]:
        """Force-close an open utterance (client sent input.end / disconnect)."""
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
