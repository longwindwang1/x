# Parley Evals

Measurement is the product: every character and every deployment gets numbers,
not vibes. Three eval families live here.

## 1. Latency (`latency_report.py` — working)

The runtime appends a per-turn timeline to `logs/turns.jsonl`
(see `src/parley/metrics.py`). Aggregate it:

```bash
python evals/latency_report.py logs/turns.jsonl
```

Reports p50/p95/mean per pipeline stage (`asr_ms`, `llm_ttft_ms`,
`tts_first_chunk_ms`, `first_audio_ms`, `total_ms`), split by turn source.
`first_audio_ms` — end of player speech to first NPC audio frame — is the
headline voice-to-voice number.

## 2. Persona consistency (planned — Week 2)

- ~40 probe prompts per character (small talk, lore questions, provocations,
  out-of-character bait), answered by the model under test.
- LLM judge scores each reply against the character card on a 1–5
  in-character scale; judge is calibrated against ~100 human-labeled samples
  and we report Cohen's κ alongside every result.
- Output: consistency rate per character, prompt mode vs. LoRA.

## 3. World-knowledge leakage (planned — Week 2)

- Adversarial probe set generated from the card's `knowledge.must_not_know`
  (anachronism questions: "what's your phone number?", "have you been to
  America?").
- A reply leaks if it acknowledges, defines, or uses the concept; judged
  automatically, spot-checked by hand.
- Output: leakage rate per character; the design target is that LoRA
  characters refuse-in-character rather than break the fourth wall.

Both judge-based evals plug into the Forge (`forge create` finishes by
running them and emitting the character's eval report).
