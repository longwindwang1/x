# Deploying Parley on a GPU pod (RunPod / AutoDL)

Reference target: one RTX 4090 (24 GB) — ~$0.40–0.70/hr on RunPod,
~¥1.5–2/hr on AutoDL. Result: full real-backend stack (Silero VAD +
faster-whisper + vLLM/Qwen2.5-7B-AWQ + Kokoro) on a single GPU, ready for
latency baselining. One script serves both platforms:
[`pod_setup.sh`](pod_setup.sh) auto-detects AutoDL and switches to the
HuggingFace mirror + 学术加速 + `/root/autodl-tmp` paths.

## Option A: RunPod

1. runpod.io → **Deploy** → GPU **RTX 4090** → template **RunPod PyTorch 2.x**
   (CUDA 12.x) → edit template and **expose HTTP port 8000** → volume ≥ 40 GB.
2. Open the pod's web terminal:

   ```bash
   bash <(curl -fsSL https://raw.githubusercontent.com/longwindwang1/x/main/deploy/pod_setup.sh)
   ```

3. When uvicorn is up, open `https://<POD_ID>-8000.proxy.runpod.net` — the
   browser client works over the proxy (mic included; the page is https).

## Option B: AutoDL

1. autodl.com → 租一台 **RTX 4090**，镜像选 **PyTorch 2.x / CUDA 12.x**。
2. 打开 JupyterLab 终端（或 SSH）：

   ```bash
   source /etc/network_turbo   # 学术加速：让 GitHub / pip 提速
   bash <(curl -fsSL https://raw.githubusercontent.com/longwindwang1/x/main/deploy/pod_setup.sh)
   ```

3. 浏览器访问推荐 **SSH 隧道**（AutoDL 实例页有现成的 SSH 命令，加 `-L` 即可）：

   ```bash
   ssh -L 8000:127.0.0.1:8000 -p <SSH端口> root@<实例地址>
   ```

   然后本机打开 http://127.0.0.1:8000 （localhost 页面浏览器允许开麦）。
   另一条路是 AutoDL 的「自定义服务」（公网暴露 6006 端口，需实名认证）：
   启动时用 `PARLEY_PORT=6006 bash <(curl ...)`，再从实例页打开自定义服务链接。

## Baseline numbers (the actual deliverable)

**On-pod (clean server-side numbers, no internet in the loop):**

```bash
cd /workspace/parley                       # AutoDL: cd /root/autodl-tmp/parley
source ../parley-venv/bin/activate         # the venv created by pod_setup.sh
python evals/bench_client.py --turns 20            # voice turns (synthesized utterance)
python evals/bench_client.py --turns 20 --text-only
python evals/latency_report.py logs/turns.jsonl    # aggregate everything recorded
```

**From your laptop (adds real network):**

```bash
python evals/bench_client.py --url ws://127.0.0.1:8000/ws --turns 20   # via SSH tunnel
# RunPod proxy: --url wss://<POD_ID>-8000.proxy.runpod.net/ws
```

Record both: server-side `first_audio_ms` is the pipeline number for the
README; the laptop run shows what a remote player would feel.

## Iterating / cost hygiene

- Code changes: `git pull` on the pod, Ctrl-C and rerun the setup script
  (vLLM stays up in the background, restart takes seconds).
- vLLM logs: `/workspace/vllm.log` (AutoDL: `/root/autodl-tmp/vllm.log`).
  GPU pressure: `nvidia-smi`.
- **Stop the pod when done** — model caches persist on the volume/data disk,
  so restarts are fast and only storage bills while stopped.

## Troubleshooting

- **pip `ResolutionImpossible` / `resolution-too-deep` (misaki/kokoro)**:
  two known traps — the image's preloaded packages conflict, and pip ≥ 25.2
  gives up on kokoro→misaki's deep dependency graph. The setup script avoids
  both: clean venv at `<workroot>/parley-venv` + installs via `uv pip`.
  Installing manually? Activate the venv and use `uv pip install`, not pip.
  If it still fails, use a Python 3.10–3.12 image.
- **vLLM OOM**: lower `--gpu-memory-utilization` to 0.65 in `pod_setup.sh`,
  or drop `max_tokens` in `deploy/server.gpu.yaml`.
- **Mic doesn't work in the browser**: the page must be https or localhost —
  RunPod proxy and the SSH tunnel both satisfy this; a bare `http://<ip>`
  will be blocked by the browser. Test with `--text-only` bench to isolate.
- **VAD never triggers**: flip `vad.backend` to `energy` in
  `deploy/server.gpu.yaml` to isolate Silero vs audio-path issues.
- **Gated models (Llama)**: default model is ungated Qwen AWQ. For Llama:
  `export PARLEY_VLLM_MODEL=<llama-awq-repo>` and `huggingface-cli login`
  first (AutoDL: also keep `HF_ENDPOINT=https://hf-mirror.com`).
