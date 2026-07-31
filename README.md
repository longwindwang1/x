# Parley

**Open-source real-time voice framework for game NPCs.**
Streaming ASR → LLM → TTS on a single 24 GB GPU, character cards, and an
automated per-character fine-tuning pipeline with built-in persona evals.

> Working name — final name TBD before first public release.

## Why

Giving a game character a voice today means either paying a closed platform
(Convai, Inworld, NVIDIA ACE) or gluing APIs together yourself. The open-source
projects that exist are either bound to one game (Mantella for Skyrim), built
for companions/VTubers rather than game integration (AIRI, Open-LLM-VTuber),
or are generic voice-agent frameworks with no notion of a character (Pipecat,
LiveKit Agents).

Parley is the missing piece: an engine-agnostic framework where a character is
a **config file**, high-fidelity personas are **one command away**, and persona
quality is **measured, not vibes**.

## Core ideas

1. **Characters are cards.** A `character.yaml` defines identity, speech
   style, knowledge boundaries, taboos, and voice. Drop it in `characters/`
   and it is live — prompt mode needs no training at all.

2. **Character Forge (automated fine-tuning).** `forge create <card>`
   synthesizes in-character training dialogues from the card, trains a LoRA
   adapter, runs the persona evals, and registers the adapter. At runtime,
   vLLM multi-LoRA serving lets N characters share one base model — adding a
   character costs megabytes, not a GPU.

3. **Persona quality is measured.** Every character ships with an eval
   report: in-character consistency (human-calibrated LLM judge) and
   world-knowledge leakage (does your medieval blacksmith know what a phone
   is?). Prompt mode vs. LoRA is an apples-to-apples comparison.

4. **Latency is a feature.** Fully streaming pipeline — VAD, utterance-level
   ASR, token streaming, sentence-chunked TTS, barge-in — instrumented
   end-to-end. Every turn reports its voice-to-voice latency breakdown.

5. **Engine-agnostic by contract.** The runtime speaks a documented
   [WebSocket protocol](docs/protocol.md). The Unity SDK and the browser dev
   client are just two clients of it.

## Architecture

```
game client (Unity SDK / browser / your engine)
  │  WebSocket: PCM16 up, PCM16 + JSON events down     docs/protocol.md
  ▼
parley runtime (FastAPI, one 24 GB GPU)
  ├─ VAD          Silero VAD (or energy fallback)
  ├─ ASR          faster-whisper, utterance-level
  ├─ LLM          vLLM · Llama-3.1-8B AWQ + per-character LoRA (multi-LoRA routing)
  ├─ chunker      LLM token stream → sentence chunks (low time-to-first-audio)
  └─ TTS          Kokoro (fast path) · CosyVoice 2 (voice cloning, roadmap)
  ▲
characters/*.yaml  ←  forge create (data synthesis → LoRA → eval report)
```

## Quickstart (no GPU needed)

The full pipeline runs with mock backends out of the box — real model
backends are config swaps, not code changes.

```bash
git clone <repo> && cd parley
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate on POSIX
pip install -e ".[dev]"
parley serve
```

Open http://127.0.0.1:8000 — pick a character, connect, and talk (or use the
text box). You'll see streaming subtitles, hear (mock) audio, and get a
per-turn latency breakdown in the sidebar.

Run the test suite:

```bash
pytest
```

### Real backends

Edit `server.yaml` (start from `server.example.yaml`):

| Stage | Backend | Install |
|---|---|---|
| VAD | `silero` | `pip install "parley[vad]"` |
| ASR | `faster_whisper` | `pip install "parley[asr]"` |
| LLM | `openai_compat` → any vLLM/OpenAI-compatible endpoint | — |
| TTS | `kokoro` | `pip install "parley[tts]"` |

For multi-character LoRA serving, run vLLM with `--enable-lora` and register
adapters produced by the Forge; characters with `model.mode: lora` route to
their adapter by name automatically.

## Project layout

```
src/parley/     runtime: session, protocol, VAD/ASR/LLM/TTS backends
characters/     character cards (3 demo characters included)
web/            browser dev client (protocol reference implementation)
docs/           protocol spec, roadmap
forge/          character fine-tuning pipeline        (in progress)
evals/          persona consistency / leakage / latency evals (in progress)
sdk-unity/      Unity client SDK                      (in progress)
```

## Status

Early v0.1 — Week 1 of active development.

- [x] Streaming runtime with pluggable backends, mock-mode e2e tested
- [x] WebSocket protocol v0.1 + browser reference client
- [x] Character card system (prompt mode)
- [x] Per-turn latency instrumentation
- [ ] Character Forge: data synthesis → LoRA → eval report
- [ ] Persona evals: consistency judge + world-leakage probes
- [ ] GPU deployment profile (faster-whisper + vLLM multi-LoRA + Kokoro/CosyVoice)
- [ ] Unity SDK + sample scene
- [ ] Latency report: p50/p95 voice-to-voice on reference hardware

## Related work

| | Streaming voice | Persona fine-tuning | Engine integration | Persona evals |
|---|---|---|---|---|
| **Convai / Inworld** (commercial) | ✅ | ❌ prompt/config only | ✅ Unity/Unreal SDKs | ❌ |
| **NVIDIA ACE** (proprietary, RTX-gated) | ✅ on-device | ⚠️ NVIDIA-tuned SLMs, not per-character | ✅ UE5 plugins | ❌ |
| **Mantella / Herika** (OSS, Skyrim) | ⚠️ turn-based | ❌ prompt-only | ⚠️ single game | ❌ |
| **AIRI / Open-LLM-VTuber** (OSS companions) | ✅ | ❌ prompt-only | ❌ not a game SDK | ❌ |
| **Pipecat / LiveKit Agents** (OSS voice infra) | ✅ | — no character layer | ⚠️ generic examples | ❌ |
| **Parley** | ✅ | ✅ card → LoRA, multi-LoRA serving | ✅ open protocol + Unity SDK | ✅ built-in |

Research that informs the Forge design: OpenCharacter (synthetic persona
data → SFT), Neeko (dynamic LoRA for multi-character role-play), TimeChara
(character hallucination / world-leakage evaluation).

## License

MIT
