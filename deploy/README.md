# Deploying Parley on a GPU (RunPod runbook)

Reference target: one RTX 4090 (24 GB), ~$0.40–0.70/hr on RunPod.
Result: full real-backend stack (Silero VAD + faster-whisper + vLLM/Qwen2.5-7B-AWQ
+ Kokoro) on a single GPU, ready for latency baselining.

## 1. Launch the pod (~2 min, web console)

1. runpod.io → **Deploy** → GPU: **RTX 4090**.
2. Template: **RunPod PyTorch 2.x** (CUDA 12.x).
3. Edit template → **Expose HTTP port 8000** (keep SSH enabled).
4. Container/volume disk: ≥ 40 GB volume (model caches live in /workspace).
5. Deploy, then open the **web terminal** (or SSH).

## 2. One command on the pod

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/longwindwang1/x/main/deploy/runpod_setup.sh)
```

First run downloads ~8 GB of models (~10 min). It starts vLLM on :8001 in the
background and Parley on :8000 in the foreground. When it prints the uvicorn
banner, open:

```
https://<POD_ID>-8000.proxy.runpod.net
```

— the browser client works over the RunPod proxy (wss). Talk with the mic or
the text box; the sidebar shows per-turn latency.

## 3. Baseline numbers (the actual deliverable)

**On-pod (clean server-side numbers, no internet in the loop):**

```bash
cd /workspace/parley
python evals/bench_client.py --turns 20            # voice turns (synthesized utterance)
python evals/bench_client.py --turns 20 --text-only
python evals/latency_report.py logs/turns.jsonl    # aggregate everything recorded
```

**From your laptop (adds real network + RunPod proxy):**

```bash
python evals/bench_client.py --url wss://<POD_ID>-8000.proxy.runpod.net/ws --turns 20
```

Record both: server-side `first_audio_ms` is the pipeline number for the
README; the laptop run shows what a remote player would feel.

## 4. Iterating / cost hygiene

- Parley code changes: `git pull` on the pod, Ctrl-C and rerun the setup
  script (vLLM stays up, restart is seconds).
- vLLM logs: `/workspace/vllm.log`. GPU pressure: `nvidia-smi`.
- **Stop the pod when done** — model caches persist on the volume, restarts
  are fast and only volume storage bills while stopped.

## Troubleshooting

- **vLLM OOM**: lower `--gpu-memory-utilization` to 0.65 in the setup script,
  or drop `max_tokens` in `deploy/server.gpu.yaml`.
- **VAD never triggers on mic**: RunPod proxy is fine for wss, but browser
  mic requires the page be https — the proxy URL is. If open-mic misfires,
  test with `--text-only` bench first, then flip `vad.backend` to `energy`
  to isolate Silero vs audio-path issues.
- **Gated models**: default model is ungated Qwen AWQ. For Llama:
  `export PARLEY_VLLM_MODEL=<llama-awq-repo>` and `huggingface-cli login`
  before running the script.
