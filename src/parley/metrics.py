"""Per-turn latency instrumentation.

Every conversational turn records a monotonic-clock timeline. The derived
breakdown is (a) sent to the client in `turn.metrics` and (b) appended to
``logs/turns.jsonl`` for offline aggregation (see evals/latency_report.py).

Timeline anchors (all ``time.monotonic()`` seconds):
    t_turn_start       voice: end of user utterance (VAD);  text: message received
    t_asr_done         transcript ready (voice turns only)
    t_llm_first_token  first delta from the LLM
    t_llm_done         LLM stream finished
    t_tts_first_chunk  first PCM chunk produced by TTS
    t_first_audio_sent first binary audio frame handed to the socket
    t_audio_done       last audio frame sent
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


def now() -> float:
    return time.monotonic()


@dataclass
class TurnTimeline:
    turn_id: int
    source: str  # "voice" | "text" | "greeting"
    character_id: str = ""
    user_chars: int = 0
    reply_chars: int = 0
    t_turn_start: float = field(default_factory=now)
    t_asr_done: float | None = None
    t_llm_first_token: float | None = None
    t_llm_done: float | None = None
    t_tts_first_chunk: float | None = None
    t_first_audio_sent: float | None = None
    t_audio_done: float | None = None
    cancelled: bool = False

    def mark(self, name: str) -> None:
        attr = f"t_{name}"
        if getattr(self, attr, None) is None:
            setattr(self, attr, now())

    def breakdown(self) -> dict[str, float]:
        """Millisecond deltas between anchors; only what was actually reached."""
        out: dict[str, float] = {}

        def ms(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return round((b - a) * 1000.0, 1)

        start = self.t_turn_start
        pairs = {
            "asr_ms": ms(start, self.t_asr_done),
            "llm_ttft_ms": ms(self.t_asr_done or start, self.t_llm_first_token),
            "llm_total_ms": ms(self.t_llm_first_token, self.t_llm_done),
            "tts_first_chunk_ms": ms(self.t_llm_first_token, self.t_tts_first_chunk),
            "first_audio_ms": ms(start, self.t_first_audio_sent),  # the headline number
            "total_ms": ms(start, self.t_audio_done),
        }
        for key, val in pairs.items():
            if val is not None:
                out[key] = val
        return out

    def to_record(self) -> dict:
        rec = {
            "ts": time.time(),
            "turn_id": self.turn_id,
            "source": self.source,
            "character_id": self.character_id,
            "user_chars": self.user_chars,
            "reply_chars": self.reply_chars,
            "cancelled": self.cancelled,
        }
        rec.update(self.breakdown())
        return rec


class MetricsLog:
    """Append-only JSONL sink for turn records."""

    def __init__(self, log_dir: str | Path | None) -> None:
        self._path: Path | None = None
        if log_dir:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._path = directory / "turns.jsonl"

    def write(self, timeline: TurnTimeline) -> None:
        if self._path is None:
            return
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(timeline.to_record()) + "\n")
