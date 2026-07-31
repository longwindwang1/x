# Roadmap

## v0.1 (current 4-week sprint)

- **Week 1 — done:** streaming runtime (VAD → ASR → LLM → sentence-chunked
  TTS), pluggable backends with mock mode, WebSocket protocol v0.1, character
  cards + 3 demo characters, browser dev client, per-turn latency
  instrumentation, e2e tests.
- **Week 2:** Character Forge (data synthesis → LoRA → eval report), vLLM
  multi-LoRA serving profile, persona-consistency + world-leakage evals with
  human-calibrated judge.
- **Week 3:** CosyVoice 2 backend (per-character voice cloning + instruct
  emotion control), GPU latency optimization pass (AWQ, chunk tuning,
  barge-in latency), Unity SDK + sample scene.
- **Week 4:** Docker one-command deploy, quickstart docs, full eval run +
  README technical report, demo video.

## Beyond v0.1

- Unreal / Godot adapters over the same protocol
- Long-term NPC memory across sessions (summary + retrieval)
- NPC-initiated speech (ambient barks driven by game events via the protocol)
- Multilingual characters (zh: Qwen + CosyVoice zh voices)
- End-to-end speech model backend as an alternative profile, once
  controllability/cost make sense

## Non-goals

- Lip sync / facial animation (belongs to the engine side; sentence events
  give integrators the timing hooks they need)
- A game. The Unity scene stays a minimal demo.
