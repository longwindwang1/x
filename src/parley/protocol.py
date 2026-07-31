"""WebSocket protocol message builders.

Single source of truth for the wire protocol described in docs/protocol.md.
Control messages are JSON text frames; audio travels as raw binary frames
(PCM16 mono little-endian). Keeping builders here means the server, tests,
and any future Python client agree on field names.
"""

from __future__ import annotations

from typing import Any

PROTOCOL_VERSION = "0.1"

# Client -> server message types.
C_SESSION_START = "session.start"
C_INPUT_TEXT = "input.text"
C_INPUT_END = "input.end"
C_BARGE_IN = "barge_in"
C_SESSION_END = "session.end"

# Server -> client message types.
S_SESSION_READY = "session.ready"
S_VAD = "vad"
S_TURN_START = "turn.start"
S_ASR_FINAL = "asr.final"
S_REPLY_DELTA = "reply.delta"
S_REPLY_SENTENCE = "reply.sentence"
S_AUDIO_START = "audio.start"
S_AUDIO_END = "audio.end"
S_TURN_METRICS = "turn.metrics"
S_TURN_CANCELLED = "turn.cancelled"
S_TURN_END = "turn.end"
S_ERROR = "error"


def session_ready(
    session_id: str, character: dict[str, Any], greeting: str | None, tts_sample_rate: int
) -> dict[str, Any]:
    return {
        "type": S_SESSION_READY,
        "protocol": PROTOCOL_VERSION,
        "session_id": session_id,
        "character": character,
        "greeting": greeting,
        "tts_sample_rate": tts_sample_rate,
    }


def vad_event(event: str) -> dict[str, Any]:
    return {"type": S_VAD, "event": event}


def turn_start(turn_id: int, source: str) -> dict[str, Any]:
    return {"type": S_TURN_START, "turn_id": turn_id, "source": source}


def asr_final(turn_id: int, text: str) -> dict[str, Any]:
    return {"type": S_ASR_FINAL, "turn_id": turn_id, "text": text}


def reply_delta(turn_id: int, text: str) -> dict[str, Any]:
    return {"type": S_REPLY_DELTA, "turn_id": turn_id, "text": text}


def reply_sentence(turn_id: int, index: int, text: str) -> dict[str, Any]:
    return {"type": S_REPLY_SENTENCE, "turn_id": turn_id, "index": index, "text": text}


def audio_start(turn_id: int, sample_rate: int) -> dict[str, Any]:
    return {"type": S_AUDIO_START, "turn_id": turn_id, "sample_rate": sample_rate, "format": "pcm16"}


def audio_end(turn_id: int) -> dict[str, Any]:
    return {"type": S_AUDIO_END, "turn_id": turn_id}


def turn_metrics(turn_id: int, breakdown: dict[str, float]) -> dict[str, Any]:
    return {"type": S_TURN_METRICS, "turn_id": turn_id, "breakdown": breakdown}


def turn_cancelled(turn_id: int, reason: str) -> dict[str, Any]:
    return {"type": S_TURN_CANCELLED, "turn_id": turn_id, "reason": reason}


def turn_end(turn_id: int) -> dict[str, Any]:
    return {"type": S_TURN_END, "turn_id": turn_id}


def error(message: str) -> dict[str, Any]:
    return {"type": S_ERROR, "message": message}
