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
IS_CN=0
if [ -f /etc/network_turbo ]; then
  echo "AutoDL detected: enabling 学术加速 + domestic mirrors"
  IS_CN=1
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  # uv does not read pip's mirror config — point it at the TUNA mirror
  # explicitly or it will pull ~6 GB of torch from overseas PyPI.
  export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  export UV_INDEX_URL="$UV_DEFAULT_INDEX"   # legacy var for older uv
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

if [ "$IS_CN" = "1" ]; then
  # 学术加速 only proxies whitelisted overseas hosts (GitHub/HF) and is only
  # needed for the git step above. Everything after this point talks to
  # domestic mirrors (TUNA pip index, ModelScope weights, hf-mirror), and
  # tunneling those through the proxy causes flaky downloads
  # ("tunnel error: unexpected end of file").
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  echo "学术加速 proxy disabled for the rest of setup (domestic mirrors are direct)"
fi

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
echo "installing vllm + torch (~6 GB on first run — progress below)"
uv pip install vllm
uv pip install -e ".[asr,tts,vad,dev]"

echo "== [4/5] start vLLM (:8001) =="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo unknown)
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 || echo 0)
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 || echo 0)
echo "GPU: $GPU_NAME  total=${GPU_TOTAL} MiB  in use=${GPU_USED} MiB"

if [ "${GPU_USED:-0}" -gt 2000 ] && ! curl -sf http://127.0.0.1:8001/v1/models > /dev/null 2>&1; then
  echo "WARNING: GPU already has ${GPU_USED} MiB in use but vLLM is not serving —"
  echo "         likely a stale process. Run:  pkill -f vllm ; sleep 5  and retry."
fi

# Size the deployment to the GPU actually rented. The 7B AWQ weights are
# ~5.5 GB; the rest of the budget is KV cache, CUDA graphs, and the ASR/TTS
# models sharing the card.
MAX_LEN=4096
GPU_UTIL=0.72
EXTRA_ARGS=""
if [ "${GPU_TOTAL:-0}" -lt 14000 ]; then
  : "${PARLEY_VLLM_MODEL:=Qwen/Qwen2.5-3B-Instruct-AWQ}"
  VLLM_MODEL="$PARLEY_VLLM_MODEL"
  MAX_LEN=2048
  GPU_UTIL=0.80
  EXTRA_ARGS="--enforce-eager"
  echo "NOTE: ${GPU_TOTAL} MiB GPU — using $VLLM_MODEL with a reduced footprint."
elif [ "${GPU_TOTAL:-0}" -lt 22000 ]; then
  MAX_LEN=3072
  GPU_UTIL=0.80
  EXTRA_ARGS="--enforce-eager"
  echo "NOTE: ${GPU_TOTAL} MiB GPU — tightening KV cache and disabling CUDA graphs."
fi

# Fetch weights up front rather than letting vLLM download them silently:
# progress is visible, failures are resumable, and in CN we can use
# ModelScope (domestic, no proxy) instead of timing out against HuggingFace.
MODEL_DIR="$WORKROOT/models/$(basename "$VLLM_MODEL")"
if [ -f "$MODEL_DIR/config.json" ]; then
  echo "weights already present at $MODEL_DIR"
elif [ "$IS_CN" = "1" ]; then
  echo "downloading $VLLM_MODEL from ModelScope (resumable; ~5.5 GB)"
  uv pip install -q modelscope
  modelscope download --model "$VLLM_MODEL" --local_dir "$MODEL_DIR"
else
  echo "downloading $VLLM_MODEL from HuggingFace (resumable; ~5.5 GB)"
  uv pip install -q "huggingface_hub[cli]"
  hf download "$VLLM_MODEL" --local-dir "$MODEL_DIR"
fi

if ! curl -sf http://127.0.0.1:8001/v1/models > /dev/null 2>&1; then
  if ! pgrep -f "vllm serve" > /dev/null 2>&1; then
    echo "starting: vllm serve $MODEL_DIR --max-model-len $MAX_LEN" \
         "--gpu-memory-utilization $GPU_UTIL $EXTRA_ARGS"
    # --served-model-name keeps the API model id equal to the HF name, so
    # deploy/server.gpu.yaml needs no change when weights come from disk.
    # shellcheck disable=SC2086
    nohup vllm serve "$MODEL_DIR" \
      --served-model-name "$VLLM_MODEL" \
      --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --port 8001 \
      $EXTRA_ARGS \
      > "$WORKROOT/vllm.log" 2>&1 &
  else
    echo "vLLM process already running (still loading); waiting on it"
  fi
  echo "waiting for vLLM to load $VLLM_MODEL from disk"
  echo "(watch progress with: tail -f $WORKROOT/vllm.log)"
  for i in $(seq 1 240); do
    sleep 5
    if curl -sf http://127.0.0.1:8001/v1/models > /dev/null 2>&1; then break; fi
    if [ "$i" -eq 240 ]; then
      echo "vLLM did not come up within 20 min." >&2
      # The engine subprocess raises the real exception; the API server only
      # reports that the engine died. Print the END of the engine traceback.
      echo "--- engine core failure (root cause) ---" >&2
      sed -n '/EngineCore failed to start/,$p' "$WORKROOT/vllm.log" \
        | grep -v "^(APIServer" | tail -n 20 >&2 || true
      echo "--- log tail ---" >&2
      tail -n 15 "$WORKROOT/vllm.log" >&2
      exit 1
    fi
  done
fi
echo "vLLM is up."

echo "== [5/5] start parley (:$PORT) =="
echo "Bench on-pod (in another terminal):"
echo "  source $VENV/bin/activate && cd $WORKDIR && python evals/bench_client.py --turns 20"
exec parley serve --config deploy/server.gpu.yaml --port "$PORT"
