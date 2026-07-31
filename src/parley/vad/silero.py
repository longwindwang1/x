"""Silero VAD backend (optional; ``pip install "parley[vad]"``).

Neural VAD, far more robust than energy thresholding in noisy rooms and over
game-audio bleed. Wraps silero-vad's streaming VADIterator behind the same
event interface as EnergyVAD.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch  # noqa: F401  (silero-vad requires torch at runtime)
from silero_vad import VADIterator, load_silero_vad

from . import BaseVAD, VADEvent, VADEventType

_WINDOW = 512  # samples per silero inference window @16 kHz


class SileroVAD(BaseVAD):
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        end_ms: int = 600,
        pre_roll_ms: int = 200,
        max_utterance_s: int = 30,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("SileroVAD supports 16 kHz input only")
        self.sample_rate = sample_rate
        self.pre_roll_ms = pre_roll_ms
        self.max_utterance_bytes = max_utterance_s * sample_rate * 2
        self._model = load_silero_vad()
        self._iter = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=end_ms,
        )
        self.reset()

    def reset(self) -> None:
        if hasattr(self, "_iter"):
            self._iter.reset_states()
        self._in_speech = False
        self._pending = np.empty(0, dtype=np.float32)
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_bytes = 0
        self._utterance = bytearray()

    def process(self, frame: bytes) -> list[VADEvent]:
        events: list[VADEvent] = []
        if not frame:
            return events
        if self._in_speech:
            self._utterance.extend(frame)
        else:
            self._push_pre_roll(frame)

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= _WINDOW:
            window, self._pending = self._pending[:_WINDOW], self._pending[_WINDOW:]
            result = self._iter(window)
            if result and "start" in result and not self._in_speech:
                self._in_speech = True
                self._utterance = bytearray(b"".join(self._pre_roll))
                events.append(VADEvent(VADEventType.SPEECH_START))
            elif result and "end" in result and self._in_speech:
                events.append(self._end_event())

        if self._in_speech and len(self._utterance) >= self.max_utterance_bytes:
            events.append(self._end_event())
        return events

    def flush(self) -> list[VADEvent]:
        if self._in_speech and self._utterance:
            return [self._end_event()]
        self.reset()
        return []

    def _end_event(self) -> VADEvent:
        utterance = bytes(self._utterance)
        self.reset()
        return VADEvent(VADEventType.SPEECH_END, utterance=utterance)

    def _push_pre_roll(self, frame: bytes) -> None:
        self._pre_roll.append(frame)
        self._pre_roll_bytes += len(frame)
        max_bytes = int(self.pre_roll_ms / 1000.0 * self.sample_rate) * 2
        while self._pre_roll_bytes > max_bytes and len(self._pre_roll) > 1:
            dropped = self._pre_roll.popleft()
            self._pre_roll_bytes -= len(dropped)
