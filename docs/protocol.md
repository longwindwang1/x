# Parley WebSocket Protocol — v0.1

Engine-agnostic wire protocol between a game client and the Parley runtime.
The bundled Unity SDK and browser client are reference implementations; any
engine that can open a WebSocket and play PCM can integrate.

## Transport

- Single WebSocket connection per NPC conversation session: `ws://<host>/ws`
- **Text frames** carry JSON control messages (UTF-8).
- **Binary frames** carry raw audio: PCM16, mono, little-endian.
  - Client → server: microphone audio at the `sample_rate` declared in
    `session.start` (16000 recommended). Send continuously in 20–100 ms
    frames — including while the NPC is speaking, so the server can detect
    barge-in.
  - Server → client: NPC speech at `tts_sample_rate` from `session.ready`.
    Frames between `audio.start` and `audio.end` belong to that turn.

## Lifecycle

```
client                                 server
  │ session.start ──────────────────────▶
  ◀────────────────────── session.ready │
  ◀──── turn.start/…audio (greeting) …  │   (if the character has a greeting)
  │ ~~ mic binary frames ~~ ───────────▶│
  ◀──────────── vad {speech_start}      │
  ◀──────────── vad {speech_end}        │
  ◀──────────── turn.start {n, voice}   │
  ◀──────────── asr.final               │
  ◀──────────── reply.delta ×N          │   (subtitles)
  ◀──────────── reply.sentence ×M       │
  ◀──────────── audio.start             │
  ◀──────────── ~~ binary audio ~~      │
  ◀──────────── audio.end               │
  ◀──────────── turn.metrics            │
  ◀──────────── turn.end                │
```

## Client → server messages

### `session.start`
Must be the first message. Selects the character and declares audio format.

```json
{"type": "session.start", "character_id": "garrick", "sample_rate": 16000, "format": "pcm16"}
```

### `input.text`
Text input that bypasses ASR (chat box, accessibility, dev tooling, evals).
Supersedes any turn in progress.

```json
{"type": "input.text", "text": "What do you sell here?"}
```

### `input.end`
Force-close the current utterance instead of waiting for VAD silence
(push-to-talk release).

```json
{"type": "input.end"}
```

### `barge_in`
Explicitly interrupt the NPC (client-side detection or a UI button). The
server also detects barge-in from mic audio on its own; this message is the
low-latency path.

```json
{"type": "barge_in"}
```

### `session.end`
Graceful close.

```json
{"type": "session.end"}
```

## Server → client messages

### `session.ready`

```json
{
  "type": "session.ready",
  "protocol": "0.1",
  "session_id": "a1b2c3d4e5f6",
  "character": {"id": "garrick", "name": "Garrick Thorne", "role": "the village blacksmith", "world": "..."},
  "greeting": "Hmph. Forge's hot and I've work to do, traveler. Speak yer business.",
  "tts_sample_rate": 24000
}
```

### `vad`
Server-side voice activity, for UI feedback (mic glow, "listening…" state).

```json
{"type": "vad", "event": "speech_start"}
{"type": "vad", "event": "speech_end"}
```

### `turn.start`
A new NPC turn began. `source` is `voice`, `text`, or `greeting`.
`turn_id` increases monotonically within the session; all subsequent
messages of this turn carry it.

```json
{"type": "turn.start", "turn_id": 3, "source": "voice"}
```

### `asr.final`
The transcript of the player's utterance (voice turns only).

```json
{"type": "asr.final", "turn_id": 3, "text": "What do you sell here?"}
```

### `reply.delta`
Streaming text of the NPC reply, for subtitles. Concatenate deltas for the
full line.

```json
{"type": "reply.delta", "turn_id": 3, "text": "Steel, "}
```

### `reply.sentence`
A completed sentence chunk, in the exact order it will be spoken. Useful for
caption timing and lip-sync segmentation.

```json
{"type": "reply.sentence", "turn_id": 3, "index": 0, "text": "Steel, traveler."}
```

### `audio.start` / binary frames / `audio.end`
Brackets the NPC speech audio for a turn.

```json
{"type": "audio.start", "turn_id": 3, "sample_rate": 24000, "format": "pcm16"}
{"type": "audio.end", "turn_id": 3}
```

### `turn.metrics`
Latency breakdown for the turn, in milliseconds. `first_audio_ms` is the
headline voice-to-voice number (end of player speech → first audio frame).

```json
{
  "type": "turn.metrics",
  "turn_id": 3,
  "breakdown": {
    "asr_ms": 210.4,
    "llm_ttft_ms": 190.2,
    "llm_total_ms": 850.0,
    "tts_first_chunk_ms": 120.9,
    "first_audio_ms": 545.1,
    "total_ms": 4102.3
  }
}
```

### `turn.cancelled`
The turn was interrupted; treat the audio stream as over and flush any
buffered playback immediately. `reason` is `barge_in`, `superseded`
(replaced by newer input), or `session_end`.

```json
{"type": "turn.cancelled", "turn_id": 3, "reason": "barge_in"}
```

### `turn.end`
Normal completion marker for a turn.

```json
{"type": "turn.end", "turn_id": 3}
```

### `error`
Non-fatal problem; the session stays open.

```json
{"type": "error", "message": "Unknown character_id: 'bob'"}
```

## Client implementation notes

- **Playback buffering:** schedule received PCM back-to-back on an audio
  clock; a ~50 ms jitter buffer is enough on LAN. On `turn.cancelled`, stop
  and drop everything buffered.
- **Echo:** if the player's speakers can reach the microphone, enable
  platform echo cancellation (browser `echoCancellation: true`, Unity mic
  processing) or the NPC will barge-in on itself.
- **Reconnect:** sessions are ephemeral; on drop, reconnect and
  `session.start` again. Dialogue memory does not survive reconnects in v0.1.

## Versioning

`session.ready.protocol` carries the protocol version. v0.x may break
compatibility between minor versions; from 1.0 changes will be additive.
