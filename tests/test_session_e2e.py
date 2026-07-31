"""End-to-end session tests over the real WebSocket endpoint with mock backends.

These exercise the full pipeline — protocol handshake, VAD segmentation, ASR,
LLM streaming, sentence chunking, TTS streaming, metrics, barge-in — without
any model dependencies, so they run everywhere (CI included).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from parley.config import BackendCfg, ServerConfig
from parley.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000


def make_client(speak_greeting: bool = False, token_delay_s: float = 0.0) -> TestClient:
    cfg = ServerConfig(
        characters_dir=str(REPO_ROOT / "characters"),
        log_dir=None,
        web_dir=None,
        speak_greeting=speak_greeting,
        vad=BackendCfg(backend="energy", options={"end_ms": 300}),
        asr=BackendCfg(backend="mock"),
        llm=BackendCfg(backend="mock", options={"token_delay_s": token_delay_s}),
        tts=BackendCfg(backend="mock"),
    )
    return TestClient(create_app(cfg))


def recv(ws):
    """Normalize a frame to ('json', dict) or ('bytes', bytes)."""
    msg = ws.receive()
    if msg.get("text") is not None:
        return ("json", json.loads(msg["text"]))
    return ("bytes", msg["bytes"])


def collect_turn(ws, max_frames: int = 1000):
    """Receive frames until turn.end or turn.cancelled; return (events, audio_bytes)."""
    events, audio = [], b""
    for _ in range(max_frames):
        kind, payload = recv(ws)
        if kind == "bytes":
            audio += payload
            continue
        events.append(payload)
        if payload["type"] in ("turn.end", "turn.cancelled"):
            return events, audio
    pytest.fail("turn never completed")


def start_session(ws, character_id: str = "garrick") -> dict:
    ws.send_text(json.dumps({
        "type": "session.start", "character_id": character_id,
        "sample_rate": SAMPLE_RATE, "format": "pcm16",
    }))
    kind, ready = recv(ws)
    assert kind == "json" and ready["type"] == "session.ready", ready
    return ready


def speech_frames(speech_ms: int = 500, trailing_silence_ms: int = 700, frame_ms: int = 40):
    """Synthetic utterance: leading silence, loud noise, trailing silence."""
    rng = np.random.default_rng(seed=7)

    def frames_of(signal: np.ndarray):
        step = SAMPLE_RATE * frame_ms // 1000
        return [signal[i : i + step].tobytes() for i in range(0, len(signal), step)]

    silence = np.zeros(SAMPLE_RATE * 200 // 1000, dtype=np.int16)
    speech = (rng.uniform(-0.3, 0.3, SAMPLE_RATE * speech_ms // 1000) * 32767).astype(np.int16)
    tail = np.zeros(SAMPLE_RATE * trailing_silence_ms // 1000, dtype=np.int16)
    return frames_of(silence) + frames_of(speech) + frames_of(tail)


# -----------------------------------------------------------------------------


def test_text_turn_full_pipeline():
    with make_client().websocket_connect("/ws") as ws:
        ready = start_session(ws)
        assert ready["character"]["id"] == "garrick"
        assert ready["tts_sample_rate"] == 24000

        ws.send_text(json.dumps({"type": "input.text", "text": "Hello there, who are you?"}))
        events, audio = collect_turn(ws)

        types = [e["type"] for e in events]
        assert types[0] == "turn.start" and events[0]["source"] == "text"
        assert "reply.delta" in types
        assert "reply.sentence" in types
        assert "audio.start" in types
        assert "audio.end" in types
        assert types[-1] == "turn.end"
        assert len(audio) > 0 and len(audio) % 2 == 0  # valid PCM16

        metrics = next(e for e in events if e["type"] == "turn.metrics")
        assert "first_audio_ms" in metrics["breakdown"]
        assert "llm_ttft_ms" in metrics["breakdown"]

        # Sentence chunks concatenate to the full reply text.
        deltas = "".join(e["text"] for e in events if e["type"] == "reply.delta")
        sentences = [e["text"] for e in events if e["type"] == "reply.sentence"]
        for sentence in sentences:
            assert sentence in " ".join(deltas.split())


def test_voice_turn_via_vad():
    with make_client().websocket_connect("/ws") as ws:
        start_session(ws, "nyx")
        for frame in speech_frames():
            ws.send_bytes(frame)

        events, audio = collect_turn(ws)
        types = [e["type"] for e in events]
        assert {"event": "speech_start", "type": "vad"} in events
        assert {"event": "speech_end", "type": "vad"} in events
        assert events[[i for i, t in enumerate(types) if t == "turn.start"][0]]["source"] == "voice"
        assert "asr.final" in types  # mock ASR transcript
        assert "audio.start" in types and len(audio) > 0

        metrics = next(e for e in events if e["type"] == "turn.metrics")
        assert "asr_ms" in metrics["breakdown"]


def test_greeting_spoken_on_session_start():
    with make_client(speak_greeting=True).websocket_connect("/ws") as ws:
        start_session(ws, "liriel")
        events, audio = collect_turn(ws)
        assert events[0]["type"] == "turn.start" and events[0]["source"] == "greeting"
        sentences = [e["text"] for e in events if e["type"] == "reply.sentence"]
        assert any("visitor among the stacks" in s for s in sentences)
        assert len(audio) > 0


def test_barge_in_cancels_active_turn():
    with make_client(token_delay_s=0.05).websocket_connect("/ws") as ws:
        start_session(ws)
        ws.send_text(json.dumps({"type": "input.text", "text": "Tell me a long story."}))

        # Let the turn get going, then interrupt it.
        kind, payload = recv(ws)
        assert payload["type"] == "turn.start"
        while True:
            kind, payload = recv(ws)
            if kind == "json" and payload["type"] == "reply.delta":
                break
        ws.send_text(json.dumps({"type": "barge_in"}))

        cancelled = None
        for _ in range(1000):
            kind, payload = recv(ws)
            if kind == "json" and payload["type"] == "turn.cancelled":
                cancelled = payload
                break
        assert cancelled is not None and cancelled["reason"] == "barge_in"

        # Session stays usable after the interruption.
        ws.send_text(json.dumps({"type": "input.text", "text": "Sorry. Go on."}))
        events, _ = collect_turn(ws)
        assert events[-1]["type"] == "turn.end"


def test_unknown_character_errors_cleanly():
    with make_client().websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "session.start", "character_id": "nobody"}))
        kind, payload = recv(ws)
        assert payload["type"] == "error"
        assert "nobody" in payload["message"]


def test_characters_api():
    client = make_client()
    res = client.get("/api/characters")
    assert res.status_code == 200
    ids = {c["id"] for c in res.json()}
    assert {"garrick", "nyx", "liriel"} <= ids
