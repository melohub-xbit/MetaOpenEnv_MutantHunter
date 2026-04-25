#!/usr/bin/env bash
set -euo pipefail

echo "=== MutantHunter Layer 6/7 GPU Run ==="
date
nvidia-smi || echo "WARN: nvidia-smi failed"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

mkdir -p /data/results

# Auth
if [ -n "${HF_TOKEN:-}" ]; then
    # New CLI (huggingface_hub >= 1.0)
    hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
    || python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=True)" \
    || echo "WARN: HF login failed, model download may still work for public models"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY"
fi

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

echo "=== ALL DONE ==="
echo "Logs and JSON in /data/results/"
ls -la /data/results/

# Brief sleep so the Space stays alive long enough to grab logs.
# /data is persistent storage, so files survive even if Space restarts.
echo "Sleeping 90 seconds for log retrieval..."
sleep 90
echo "Exiting. Switch hardware to CPU basic NOW to stop billing."
