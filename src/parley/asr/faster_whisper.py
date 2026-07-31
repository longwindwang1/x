"""faster-whisper ASR backend (optional; ``pip install "parley[asr]"``).

Utterance-level transcription: the VAD already segments speech, so we run one
batched decode per utterance instead of true streaming decode — simpler, and
for utterances of a few seconds the added latency is dominated by model speed.
The model loads once and inference runs in a worker thread to keep the event
loop responsive.
"""

from __future__ import annotations

import asyncio

import numpy as np
from faster_whisper import WhisperModel

from . import BaseASR


class FasterWhisperASR(BaseASR):
    def __init__(
        self,
        model: str = "small.en",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "en",
        beam_size: int = 1,
    ) -> None:
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size

    async def transcribe(self, pcm16: bytes) -> str:
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # VAD segmentation already happened upstream
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
