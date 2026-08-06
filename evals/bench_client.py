"""Latency benchmark client for a running Parley server.

Drives real conversation turns over the WebSocket protocol and reports
percentiles for both views of latency:

- client voice-to-voice: last speech frame sent -> first audio frame received
  (includes VAD endpointing wait and network; what a player feels)
- server breakdown: the `turn.metrics` the runtime measured internally
  (`first_audio_ms` = end-of-utterance -> first audio frame; the pipeline
  number quoted in the README)

Usage (on the GPU pod, clean numbers):
    python evals/bench_client.py --turns 20
    python evals/bench_client.py --turns 20 --text-only

From a laptop against a remote pod (adds network + proxy):
    python evals/bench_client.py --url wss://<POD_ID>-8000.proxy.runpod.net/ws --turns 20

The spoken utterance is taken from --wav, synthesized with kokoro when
installed, or falls back to a noise burst (only meaningful against the mock
ASR backend; real whisper will transcribe silence).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave

import numpy as np

try:
    import websockets
except ImportError:  # pragma: no cover
    sys.exit('bench_client needs the "websockets" package: pip install websockets')

MIC_RATE = 16000
UTTERANCE_TEXT = "Have you heard any rumors about the road to the capital lately?"
TEXT_PROMPTS = [
    "Hello there. Who are you, and what is this place?",
    "What do you sell here? Anything special?",
    "Have you heard any rumors lately?",
    "What do you think of the people who run this town?",
]
SERVER_FIELDS = ["asr_ms", "llm_ttft_ms", "tts_first_chunk_ms", "first_audio_ms", "total_ms"]


# -- utterance sources ---------------------------------------------------------


def resample_to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == MIC_RATE:
        return samples
    out_len = int(len(samples) * MIC_RATE / rate)
    positions = np.linspace(0, len(samples) - 1, out_len)
    return np.interp(positions, np.arange(len(samples)), samples)


def load_wav(path: str) -> bytes:
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if wav.getnchannels() > 1:
            samples = samples.reshape(-1, wav.getnchannels()).mean(axis=1)
    return resample_to_16k(samples, rate).astype(np.int16).tobytes()


def synth_utterance() -> bytes | None:
    try:
        from kokoro import KPipeline
    except ImportError:
        return None
    print(f'synthesizing bench utterance with kokoro: "{UTTERANCE_TEXT}"')
    pipeline = KPipeline(lang_code="a")
    parts = [np.asarray(audio, dtype=np.float32) for _, _, audio in pipeline(UTTERANCE_TEXT, voice="af_heart")]
    samples = np.concatenate(parts) * 32767
    return resample_to_16k(samples, 24000).astype(np.int16).tobytes()


def noise_utterance(duration_s: float = 1.5) -> bytes:
    print("WARNING: no --wav and no kokoro; using a noise burst "
          "(only valid against the mock ASR backend)")
    rng = np.random.default_rng(seed=7)
    return (rng.uniform(-0.3, 0.3, int(MIC_RATE * duration_s)) * 32767).astype(np.int16).tobytes()


# -- protocol helpers ----------------------------------------------------------


async def recv_event(ws):
    msg = await ws.recv()
    if isinstance(msg, (bytes, bytearray)):
        return ("bytes", msg)
    return ("json", json.loads(msg))


async def drain_turn(ws) -> None:
    """Consume everything up to turn.end / turn.cancelled (e.g. the greeting)."""
    while True:
        kind, payload = await recv_event(ws)
        if kind == "json" and payload["type"] in ("turn.end", "turn.cancelled"):
            return


async def start_session(ws, character: str) -> dict:
    await ws.send(json.dumps({
        "type": "session.start", "character_id": character,
        "sample_rate": MIC_RATE, "format": "pcm16",
    }))
    kind, ready = await recv_event(ws)
    if kind != "json" or ready.get("type") != "session.ready":
        raise RuntimeError(f"expected session.ready, got: {ready}")
    if ready.get("greeting"):
        await drain_turn(ws)
    return ready


# -- turn drivers --------------------------------------------------------------


async def send_utterance(ws, pcm: bytes, frame_ms: int, pace: float) -> float:
    """Stream speech + trailing silence; return perf-time of last speech frame."""
    frame_bytes = MIC_RATE * frame_ms // 1000 * 2
    lead = b"\x00" * (MIC_RATE * 200 // 1000 * 2)              # 200 ms leading silence
    tail = b"\x00" * (MIC_RATE * 900 // 1000 * 2)              # > vad end_ms
    t_last_speech = 0.0
    for buf, is_speech in ((lead, False), (pcm, True), (tail, False)):
        for i in range(0, len(buf), frame_bytes):
            await ws.send(buf[i : i + frame_bytes])
            if is_speech:
                t_last_speech = time.perf_counter()
            await asyncio.sleep(frame_ms / 1000.0 / pace)
    return t_last_speech


async def collect_turn(ws) -> dict:
    """Receive until turn.end; record time of first audio frame and metrics."""
    result = {"t_first_audio": None, "breakdown": {}, "cancelled": False}
    while True:
        kind, payload = await recv_event(ws)
        if kind == "bytes":
            if result["t_first_audio"] is None:
                result["t_first_audio"] = time.perf_counter()
            continue
        if payload["type"] == "turn.metrics":
            result["breakdown"] = payload["breakdown"]
        elif payload["type"] == "asr.final":
            result["asr_text"] = payload["text"]
        elif payload["type"] == "error":
            print(f"  server error: {payload['message']}")
        elif payload["type"] == "turn.cancelled":
            result["cancelled"] = True
            return result
        elif payload["type"] == "turn.end":
            return result


async def voice_turn(ws, pcm: bytes, frame_ms: int, pace: float) -> dict:
    send_task = asyncio.create_task(send_utterance(ws, pcm, frame_ms, pace))
    collect_task = asyncio.create_task(collect_turn(ws))
    t_last_speech = await send_task
    result = await collect_task
    if result["t_first_audio"] is not None:
        result["client_ms"] = (result["t_first_audio"] - t_last_speech) * 1000.0
    return result


async def text_turn(ws, prompt: str) -> dict:
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "input.text", "text": prompt}))
    result = await collect_turn(ws)
    if result["t_first_audio"] is not None:
        result["client_ms"] = (result["t_first_audio"] - t0) * 1000.0
    return result


# -- reporting -----------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (k - lower)


def report(rows: list[dict], mode: str) -> None:
    print(f"\n== bench results: {mode} turns (n={len(rows)}) ==")
    print(f"{'metric':<26}{'p50':>10}{'p95':>10}{'mean':>10}{'n':>6}")

    def line(name: str, values: list[float]) -> None:
        if values:
            print(f"{name:<26}{percentile(values, 50):>10.0f}{percentile(values, 95):>10.0f}"
                  f"{statistics.fmean(values):>10.0f}{len(values):>6}")

    line("client_voice_to_voice_ms" if mode == "voice" else "client_ttfa_ms",
         [r["client_ms"] for r in rows if "client_ms" in r])
    for field in SERVER_FIELDS:
        line(f"server:{field}", [r["breakdown"][field] for r in rows if field in r["breakdown"]])


# -- main ----------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    if args.text_only:
        pcm = None
    elif args.wav:
        pcm = load_wav(args.wav)
    else:
        pcm = synth_utterance() or noise_utterance()

    async with websockets.connect(args.url, max_size=None) as ws:
        ready = await start_session(ws, args.character)
        print(f"session {ready['session_id']} with {ready['character']['name']} at {args.url}")

        rows: list[dict] = []
        total = args.warmup + args.turns
        for i in range(total):
            if pcm is None:
                result = await text_turn(ws, TEXT_PROMPTS[i % len(TEXT_PROMPTS)])
            else:
                result = await voice_turn(ws, pcm, args.frame_ms, args.pace)
            if result["cancelled"]:
                print(f"turn {i + 1}/{total}: cancelled?! skipping")
                continue
            phase = "warmup" if i < args.warmup else "bench"
            client_ms = result.get("client_ms")
            server_ms = result["breakdown"].get("first_audio_ms")
            asr_note = f' asr={result["asr_text"]!r}' if "asr_text" in result else ""
            print(f"turn {i + 1}/{total} [{phase}]: "
                  f"client={client_ms and f'{client_ms:.0f}ms'} "
                  f"server_first_audio={server_ms and f'{server_ms}ms'}{asr_note}")
            if i >= args.warmup:
                rows.append(result)
            await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "session.end"}))

    report(rows, "text" if pcm is None else "voice")
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps({k: v for k, v in row.items() if k != "t_first_audio"}) + "\n")
        print(f"rows appended to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--character", default="garrick")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--text-only", action="store_true", help="skip audio input; bench input.text turns")
    parser.add_argument("--wav", help="utterance WAV (any rate; resampled to 16 kHz mono)")
    parser.add_argument("--frame-ms", type=int, default=40)
    parser.add_argument("--pace", type=float, default=1.0,
                        help="audio send speed multiple (1.0 = real time; >1 compresses the VAD wait)")
    parser.add_argument("--out", help="append raw per-turn rows to this JSONL file")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
