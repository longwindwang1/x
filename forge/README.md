# Character Forge (Week 2 — in progress)

`forge create characters/<card>.yaml` turns a character card into a
high-fidelity persona:

```
card ──▶ 1. synthesize dialogues ──▶ 2. filter ──▶ 3. train LoRA ──▶ 4. eval ──▶ register
```

1. **Synthesize** — a teacher model (Claude API) generates 2–3k multi-turn
   dialogues from the card: daily chatter, lore Q&A, quest-style exchanges,
   and adversarial player turns that try to pull the character out of world.
2. **Filter** — rule checks (length, format, banned-term leakage from
   `knowledge.must_not_know`) plus a judge pass on style adherence.
3. **Train** — LoRA (rank 16–32) on the runtime's base model
   (Llama-3.1-8B-Instruct) via unsloth/TRL; hours on one rented GPU.
4. **Evaluate** — persona-consistency and world-leakage evals
   (see `evals/README.md`), reported as prompt-mode vs. LoRA deltas.
5. **Register** — the adapter lands in the model registry; setting
   `model.mode: lora` + `lora_adapter` in the card routes runtime requests
   to it through vLLM multi-LoRA serving (no extra GPU per character).

Design references: OpenCharacter (synthetic persona SFT), Neeko (multi-role
LoRA), TimeChara (point-in-time character hallucination evals).
