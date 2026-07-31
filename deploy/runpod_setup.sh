#!/usr/bin/env bash
# One-shot setup for a fresh RunPod GPU pod (PyTorch template, CUDA 12.x).
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/longwindwang1/x/main/deploy/runpod_setup.sh)
#
# Assumes: 24 GB GPU (RTX 4090), port 8000 exposed as HTTP, /workspace volume.
# Takes ~10 min on first run (model downloads); subsequent starts reuse caches.
set -euo pipefail

REPO_URL="${PARLEY_REPO:-https://github.com/longwindwang1/x.git}"
VLLM_MODEL="${PARLEY_VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
WORKDIR=/workspace/parley

echo "== [1/5] clone/update repo =="
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" pull --ff-only
else
  git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

echo "== [2/5] system deps (espeak-ng for kokoro g2p fallback) =="
apt-get update -qq && apt-get install -y -qq espeak-ng > /dev/null

echo "== [3/5] python deps =="
pip install -q -e ".[asr,tts,vad,dev]"
pip install -q vllm

echo "== [4/5] start vLLM (:8001) =="
if ! curl -sf http://127.0.0.1:8001/v1/models > /dev/null 2>&1; then
  nohup python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.72 \
    --port 8001 \
    > /workspace/vllm.log 2>&1 &
  echo "waiting for vLLM to load $VLLM_MODEL (see /workspace/vllm.log)..."
  for i in $(seq 1 120); do
    sleep 5
    if curl -sf http://127.0.0.1:8001/v1/models > /dev/null 2>&1; then break; fi
    if [ "$i" -eq 120 ]; then echo "vLLM failed to start; tail /workspace/vllm.log" >&2; exit 1; fi
  done
fi
echo "vLLM is up."

echo "== [5/5] start parley (:8000) =="
echo "After startup: open the pod's port-8000 URL for the browser client,"
echo "or run the bench on-pod:  python evals/bench_client.py --turns 20"
exec parley serve --config deploy/server.gpu.yaml
