# Parley Unity SDK (Week 3 — planned)

A thin C# client for the [Parley WebSocket protocol](../docs/protocol.md),
packaged as a UPM package. The browser client (`web/`) is the reference
implementation to port.

Planned surface:

- `ParleyClient` — connection + protocol (WebSocket, JSON events, PCM frames)
- `ParleyNpc` MonoBehaviour — attach to a character; exposes
  `UnityEvent<string>` for subtitles/sentences and plays streamed audio via
  `AudioSource` (barge-in aware: flushes on `turn.cancelled`)
- `ParleyMicrophone` — mic capture → 16 kHz PCM16 frames, push-to-talk or
  open-mic modes
- Sample scene: three NPCs (Garrick, Nyx, Liriel), walk up and talk

Scope guard: no lip sync, no animation rigging — this SDK moves audio and
events; everything visual stays in the sample scene.
