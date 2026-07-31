"""RMS-energy VAD.

Dependency-free default that is good enough for quiet dev environments and
deterministic tests. Production deployments should prefer the Silero backend
(robust to noise); both emit identical events so they are drop-in swaps.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import BaseVAD, VADEvent, VADEventType


class EnergyVAD(BaseVAD):
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.015,
        start_ms: int = 60,
        end_ms: int = 600,
        pre_roll_ms: int = 200,
        max_utterance_s: int = 30,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.pre_roll_ms = pre_roll_ms
        self.max_utterance_bytes = max_utterance_s * sample_rate * 2
        self.reset()

    def reset(self) -> None:
        self._in_speech = False
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_bytes = 0
        self._utterance = bytearray()

    def process(self, frame: bytes) -> list[VADEvent]:
        if not frame:
            return []
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        frame_ms = len(samples) / self.sample_rate * 1000.0
        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        voiced = rms >= self.threshold
        events: list[VADEvent] = []

        if not self._in_speech:
            self._push_pre_roll(frame)
            if voiced:
                self._voiced_ms += frame_ms
                if self._voiced_ms >= self.start_ms:
                    self._in_speech = True
                    self._silence_ms = 0.0
                    self._utterance = bytearray(b"".join(self._pre_roll))
                    events.append(VADEvent(VADEventType.SPEECH_START))
            else:
                self._voiced_ms = 0.0
            return events

        self._utterance.extend(frame)
        if voiced:
            self._silence_ms = 0.0
        else:
            self._silence_ms += frame_ms

        if self._silence_ms >= self.end_ms or len(self._utterance) >= self.max_utterance_bytes:
            events.append(VADEvent(VADEventType.SPEECH_END, utterance=bytes(self._utterance)))
            self.reset()
        return events

    def flush(self) -> list[VADEvent]:
        if self._in_speech and self._utterance:
            utterance = bytes(self._utterance)
            self.reset()
            return [VADEvent(VADEventType.SPEECH_END, utterance=utterance)]
        self.reset()
        return []

    def _push_pre_roll(self, frame: bytes) -> None:
        self._pre_roll.append(frame)
        self._pre_roll_bytes += len(frame)
        max_bytes = int(self.pre_roll_ms / 1000.0 * self.sample_rate) * 2
        while self._pre_roll_bytes > max_bytes and len(self._pre_roll) > 1:
            dropped = self._pre_roll.popleft()
            self._pre_roll_bytes -= len(dropped)
