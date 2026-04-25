#!/usr/bin/env bash
set -euo pipefail

# Catch any failure and idle instead of exiting — exiting triggers HF restart loop.
trap 'echo "[ERROR] Script failed at line $LINENO. Idling instead of restarting."; while true; do sleep 300; done' ERR

# Diagnostic mode: dump /data/results/* to stdout and exit, no GPU work.
# Set DUMP_RESULTS_AND_EXIT=1 as a Space variable, switch hardware to CPU
# basic, then Factory rebuild. /data is persistent storage so prior-run
# artifacts survive. Logs are fetched via fetch_space_logs.
if [ "${DUMP_RESULTS_AND_EXIT:-0}" = "1" ]; then
    echo "=== DUMP MODE: /data/results/ ==="
    ls -la /data/ 2>&1 || true
    ls -la /data/results/ 2>&1 || true
    for f in /data/results/*; do
        if [ -f "$f" ]; then
            echo ""
            echo "=========================================="
            echo "FILE: $f ($(wc -c < "$f") bytes)"
            echo "=========================================="
            cat "$f"
            echo ""
            echo "=========================================="
            echo "END FILE: $f"
            echo "=========================================="
        fi
    done
    echo "=== DUMP COMPLETE ==="
    exit 0
fi

# HF Spaces runs as non-root; ensure HOME points to writable space
export HOME="${HOME:-/data}"
if [ ! -w "$HOME" ]; then
    export HOME=/tmp/home
fi
mkdir -p "$HOME"
echo "Using HOME=$HOME"

echo "=== MutantHunter Layer 6/7 GPU Run ==="
date
nvidia-smi || echo "WARN: nvidia-smi failed"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

mkdir -p /data/results

# Auth — best-effort. A non-fatal warning here must not kill the script.
set +e

if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN" 2>/dev/null \
    || python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" \
    || echo "WARN: HF login failed, continuing (Qwen2.5-Coder-1.5B is public)"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    # Use Python API instead of `wandb login` CLI to avoid /.netrc permission error
    python -c "
import os
os.environ['WANDB_API_KEY'] = '$WANDB_API_KEY'
os.environ.setdefault('WANDB_DIR', '/data/wandb')
os.environ.setdefault('WANDB_CACHE_DIR', '/data/wandb-cache')
os.environ.setdefault('WANDB_CONFIG_DIR', '/data/wandb-config')
os.makedirs(os.environ['WANDB_DIR'], exist_ok=True)
os.makedirs(os.environ['WANDB_CACHE_DIR'], exist_ok=True)
os.makedirs(os.environ['WANDB_CONFIG_DIR'], exist_ok=True)
import wandb
wandb.login(key='$WANDB_API_KEY', verify=True)
print('W&B login OK')
" || echo "WARN: W&B login failed, training will skip W&B logging"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DIR=/data/wandb
export WANDB_CACHE_DIR=/data/wandb-cache
export WANDB_CONFIG_DIR=/data/wandb-config

set -e

# Pre-cache model
echo "=== Caching Qwen2.5-Coder-1.5B-Instruct ==="
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', torch_dtype=torch.bfloat16)
print('Model cached.')
"

# Start env server
echo "=== Starting env server ==="
uvicorn mutant_hunter.server.app:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
sleep 5

echo "Smoke check:"
curl -sS -X POST http://127.0.0.1:8000/reset -H "Content-Type: application/json" -d '{"seed":0}' | head -c 300
echo

# Layer 6
echo "=== Layer 6: Zero-shot distribution (15 episodes) ==="
python evaluation/zero_shot_distribution.py \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --episodes 15 \
    --max-new-tokens 1024 \
    --device cuda \
    2>&1 | tee /data/results/layer6.log

echo "=== Layer 6 finished. Exit code: $? ==="

# Layer 7
echo "=== Layer 7: 30-step GRPO smoke ==="
python evaluation/grpo_smoke_run.py \
    --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --steps 30 \
    --rollouts-per-step 4 \
    --max-new-tokens 1024 \
    --wandb-project mutant-hunter-validation \
    2>&1 | tee /data/results/layer7.log

echo "=== Layer 7 finished. Exit code: $? ==="

kill $SERVER_PID 2>/dev/null || true

# Persist evaluation/_results to /data
if [ -d evaluation/_results ]; then
    cp -rv evaluation/_results/* /data/results/ || true
fi

echo ""
echo "==========================================="
echo "ALL LAYERS COMPLETE — script entering idle"
echo "==========================================="
echo "Logs and JSON in /data/results/"
ls -la /data/results/ || true
echo ""
echo "MANUALLY SWITCH HARDWARE TO CPU BASIC NOW TO STOP BILLING"
echo ""
# Sleep forever to prevent restart loop. Switch hardware to CPU to actually stop.
while true; do sleep 300; echo "[$(date)] Idle, waiting for hardware downgrade..."; done
