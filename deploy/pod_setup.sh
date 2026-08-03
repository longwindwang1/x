#!/usr/bin/env bash
# One-shot setup for a fresh GPU pod — works on RunPod and AutoDL
# (PyTorch/CUDA 12.x image, 24 GB GPU, e.g. RTX 4090).
#
#   RunPod:  bash <(curl -fsSL https://raw.githubusercontent.com/longwindwang1/x/main/deploy/pod_setup.sh)
#   AutoDL:  source /etc/network_turbo   # 学术加速, then the same command
#
# Takes ~10 min on first run (model downloads); later starts reuse caches.
# Env overrides: PARLEY_REPO, PARLEY_VLLM_MODEL, PARLEY_PORT (default 8000;
# use 6006 on AutoDL to expose via 自定义服务).
set -euo pipefail

REPO_URL="${PARLEY_REPO:-https://github.com/longwindwang1/x.git}"
VLLM_MODEL="${PARLEY_VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${PARLEY_PORT:-8000}"

# Platform detection: AutoDL boxes ship /etc/network_turbo (GitHub proxy) and
# need the HuggingFace mirror; both are no-ops elsewhere.
if [ -f /etc/network_turbo ]; then
  echo "AutoDL detected: enabling 学术加速 + hf-mirror"
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  WORKROOT=/root/autodl-tmp        # the big data disk on AutoDL
else
  WORKROOT=/workspace
fi
WORKDIR="$WORKROOT/parley"
export HF_HOME="${HF_HOME:-$WORKROOT/hf-cache}"

echo "== [1/5] clone/update repo =="
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" pull --ff-only
else
  git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

echo "== [2/5] system deps (espeak-ng for kokoro g2p fallback) =="
apt-get update -qq && apt-get install -y -qq espeak-ng > /dev/null

echo "== [3/5] python deps (clean venv — pod base images preinstall pinned"
echo "         packages that break dependency resolution, e.g. kokoro/misaki) =="
VENV="$WORKROOT/parley-venv"
if [ ! -f "$VENV/bin/activate" ]; then
  python -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# uv's resolver handles the kokoro/misaki dependency graph that trips pip's
# resolution depth limit (pip >= 25.2 "resolution-too-deep").
pip install -q --upgrade uv
# vllm first: it pins an exact torch build; everything else accepts it.
uv pip install -q vllm
uv pip install -q -e ".[asr,tts,vad,dev]"

echo "== [4/5] start vLLM (:8001) =="
if ! curl -sf http://127.0.0.1:8001/v1/models > /dev/null 2>&1; then
  nohup python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.72 \
    --port 8001 \
    > "$WORKROOT/vllm.log" 2>&1 &
  echo "waiting for vLLM to load $VLLM_MODEL (see $WORKROOT/vllm.log)..."
  for i in $(seq 1 120); do
    sleep 5
    if curl -sf http://127.0.0.1:8001/v1/models > /dev/null 2>&1; then break; fi
    if [ "$i" -eq 120 ]; then echo "vLLM failed to start; tail $WORKROOT/vllm.log" >&2; exit 1; fi
  done
fi
echo "vLLM is up."

echo "== [5/5] start parley (:$PORT) =="
echo "Bench on-pod (in another terminal):"
echo "  source $VENV/bin/activate && cd $WORKDIR && python evals/bench_client.py --turns 20"
exec parley serve --config deploy/server.gpu.yaml --port "$PORT"
