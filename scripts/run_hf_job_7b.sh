#!/usr/bin/env bash
# Runs INSIDE the HF Job container. Self-contained: clones the repo, installs
# deps, walks Phases 0..6. On any failure the ERR trap pushes whatever is in
# /tmp/results/ to the dataset repo under partial/.
#
# Required env vars (set via `hf jobs run --secret` / `--env`):
#   HF_TOKEN          — HF write token
#   WANDB_API_KEY     — W&B API key
#
# Optional env vars:
#   SKIP_BASELINES=1  — skip Phase 1 + 2; pull baseline JSONs from HF Hub instead

set -euo pipefail

MODEL_REPO="jester1177/mutant-hunter-qwen-coder-7b-lora"
DATASET_REPO="jester1177/mutant-hunter-results"
RESULTS_DIR="/tmp/results"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"

# ------------------------------------------------------------------------------
# ERR trap — best-effort upload of partial artifacts when any step fails
# ------------------------------------------------------------------------------
push_partial() {
    rc=$?
    echo ""
    echo "[trap] caught exit code ${rc}; uploading ${RESULTS_DIR}/ as partial/"
    if [ -d "${RESULTS_DIR}" ]; then
        python - <<'PY' || echo "[trap] partial upload itself failed; giving up"
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/tmp/results",
    repo_id="jester1177/mutant-hunter-results",
    repo_type="dataset",
    path_in_repo="partial",
)
print("[trap] partial artifacts pushed under partial/")
PY
    else
        echo "[trap] no ${RESULTS_DIR}/ directory; nothing to push"
    fi
    exit "${rc}"
}
trap push_partial ERR

# ==============================================================================
# Phase 0: setup
# ==============================================================================
echo "=== Phase 0: setup ==="
cd /tmp
[ -d MetaOpenEnv_MutantHunter ] || git clone https://github.com/melohub-xbit/MetaOpenEnv_MutantHunter.git
cd MetaOpenEnv_MutantHunter
mkdir -p "${RESULTS_DIR}" "${RESULTS_DIR}/training" "${RESULTS_DIR}/plots"

# Install huggingface_hub up front so the ERR trap can always upload partials,
# even if a later install step fails.
pip install --no-cache-dir huggingface_hub

# Wait for the GPU driver. nvidia-smi talks to NVML and typically succeeds even
# when the CUDA driver API (cuInit) is still returning 802. ~120s budget.
echo "Waiting for GPU driver ..."
for i in $(seq 1 30); do
    if nvidia-smi -L 2>/dev/null | grep -q GPU; then
        echo "GPU driver ready: $(nvidia-smi -L | head -n1)"
        break
    fi
    echo "  attempt ${i}/30: nvidia-smi not ready yet, sleeping 4s ..."
    sleep 4
    if [ "$i" = "30" ]; then
        echo "ERROR: nvidia-smi never reported a GPU after ~120s" >&2
        exit 1
    fi
done

# The base image (pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel) ships
# torch==2.5.1 / torchvision==0.20.1 / torchaudio==2.5.1. TRL >= 1.x needs
# FSDPModule which was only added in torch 2.6, so we MUST upgrade. Plain
# --upgrade can't move torchaudio (it pins torch==2.5.1), and pip will
# otherwise say "already satisfied" for torch. --force-reinstall sidesteps
# both: all three wheels are overwritten as a matched set.
pip install --no-cache-dir --upgrade --force-reinstall \
    torch torchvision torchaudio --index-url "${PYTORCH_INDEX}"

# Static checks (torch version, FSDPModule import, torchvision C++ ops):
python -c "
import torch, torchvision
from torchvision import io as _io
from torch.distributed.fsdp import FSDPModule
maj, mn = (int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
assert (maj, mn) >= (2, 6), f'torch is {torch.__version__}, need >=2.6 for FSDPModule'
print(f'torch={torch.__version__}, torchvision={torchvision.__version__} — static imports OK')
"

# CUDA-driver-ready check: nvidia-smi/NVML can be ready before the CUDA driver
# API (cuInit) is, so torch.cuda.is_available() trips error 802 on the first
# probes. torch caches the failed result inside one process, so we retry from
# a *fresh* python invocation each iteration. ~120s budget.
echo "Waiting for CUDA driver API (cuInit) ..."
for i in $(seq 1 30); do
    if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)" 2>/dev/null; then
        echo "CUDA driver API ready: $(python -c 'import torch; print(torch.cuda.device_count(), "device(s)")')"
        break
    fi
    echo "  attempt ${i}/30: torch.cuda.is_available() False, sleeping 4s ..."
    sleep 4
    if [ "$i" = "30" ]; then
        echo "ERROR: torch.cuda.is_available() never True after ~120s" >&2
        exit 1
    fi
done

# Project + extras (training extras pull trl/transformers/peft/accelerate/datasets)
pip install --no-cache-dir -e ".[training]"
pip install --no-cache-dir bitsandbytes wandb

python -c "
import trl, bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
print(f'TRL={trl.__version__}, bitsandbytes={bitsandbytes.__version__} — full training-pipeline imports OK')
"

# Hub + W&B logins
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
test -n "${WANDB_API_KEY:-}" || { echo "ERROR: WANDB_API_KEY is empty in container" >&2; exit 1; }
HOME=/tmp wandb login --relogin "${WANDB_API_KEY}"

python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("jester1177/mutant-hunter-qwen-coder-7b-lora", repo_type="model", exist_ok=True)
api.create_repo("jester1177/mutant-hunter-results", repo_type="dataset", exist_ok=True)
print("repos ready")
PY

# ==============================================================================
# Phase 1 + 2: baselines (skippable via SKIP_BASELINES=1)
# ==============================================================================
if [ -z "${SKIP_BASELINES:-}" ]; then
    echo "=== Phase 1: heuristic baseline (~5 min) ==="
    python training/baseline_eval.py \
        --episodes 15 \
        --policy mutation_aware \
        --seed-start 0 \
        --out "${RESULTS_DIR}/baseline_heuristic.json"

    python - <<'PY'
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="/tmp/results/baseline_heuristic.json",
    path_in_repo="baseline_heuristic.json",
    repo_id="jester1177/mutant-hunter-results",
    repo_type="dataset",
)
print("baseline_heuristic.json pushed")
PY

    echo "=== Phase 2: zero-shot LLM baseline (~30 min) ==="
    python evaluation/zero_shot_distribution.py \
        --episodes 15 \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --max-new-tokens 1024 \
        --seed-start 0 \
        --device auto

    cp evaluation/_results/zero_shot_distribution.json "${RESULTS_DIR}/baseline_zeroshot.json"

    python - <<'PY'
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="/tmp/results/baseline_zeroshot.json",
    path_in_repo="baseline_zeroshot.json",
    repo_id="jester1177/mutant-hunter-results",
    repo_type="dataset",
)
print("baseline_zeroshot.json pushed")
PY
else
    echo "=== Skipping Phase 1 + 2 (SKIP_BASELINES set); pulling baselines from HF Hub ==="
    python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="jester1177/mutant-hunter-results", filename="baseline_heuristic.json",
                repo_type="dataset", local_dir="/tmp/results")
hf_hub_download(repo_id="jester1177/mutant-hunter-results", filename="baseline_zeroshot.json",
                repo_type="dataset", local_dir="/tmp/results")
print("Baselines downloaded from HF Hub")
PY
fi

# ==============================================================================
# Phase 3: GRPO training (~4h)
# ==============================================================================
echo "=== Phase 3: GRPO training ==="
python training/train_grpo.py \
    --steps 80 \
    --rollouts-per-step 3 \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --max-new-tokens 1024 \
    --learning-rate 5e-6 \
    --seed 42 \
    --output-dir "${RESULTS_DIR}/training" \
    --wandb-project mutant-hunter-final

if [ ! -d "${RESULTS_DIR}/training/final" ]; then
    echo "ERROR: ${RESULTS_DIR}/training/final/ not found after training" >&2
    exit 1
fi

python - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/tmp/results/training/final",
    repo_id="jester1177/mutant-hunter-qwen-coder-7b-lora",
    repo_type="model",
)
print("LoRA adapter pushed to model repo")
PY

if [ -f "${RESULTS_DIR}/training/training_log.jsonl" ]; then
    python - <<'PY'
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="/tmp/results/training/training_log.jsonl",
    path_in_repo="training_log.jsonl",
    repo_id="jester1177/mutant-hunter-qwen-coder-7b-lora",
    repo_type="model",
)
print("training_log.jsonl pushed")
PY
else
    echo "[warn] training_log.jsonl not found; skipping upload"
fi

# ==============================================================================
# Phase 4: trained eval (~30 min)
# ==============================================================================
echo "=== Phase 4: trained eval ==="
# Stash Phase 2 output so the trained-eval write does not clobber it.
cp "${RESULTS_DIR}/baseline_zeroshot.json" "${RESULTS_DIR}/baseline_zeroshot.keep.json"

python evaluation/zero_shot_distribution.py \
    --episodes 15 \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --lora-path "${RESULTS_DIR}/training/final" \
    --max-new-tokens 1024 \
    --seed-start 0 \
    --device auto

cp evaluation/_results/zero_shot_distribution.json "${RESULTS_DIR}/trained_eval.json"

python - <<'PY'
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="/tmp/results/trained_eval.json",
    path_in_repo="trained_eval.json",
    repo_id="jester1177/mutant-hunter-results",
    repo_type="dataset",
)
print("trained_eval.json pushed")
PY

# ==============================================================================
# Phase 5: plots
# ==============================================================================
echo "=== Phase 5: plots ==="
python evaluation/make_plots.py \
    --baseline-heuristic-json "${RESULTS_DIR}/baseline_heuristic.json" \
    --baseline-zeroshot-json "${RESULTS_DIR}/baseline_zeroshot.json" \
    --trained-eval-json "${RESULTS_DIR}/trained_eval.json" \
    --training-log-json "${RESULTS_DIR}/training/training_log.jsonl" \
    --out-dir "${RESULTS_DIR}/plots/"

python - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/tmp/results/plots",
    repo_id="jester1177/mutant-hunter-qwen-coder-7b-lora",
    repo_type="model",
    path_in_repo="plots",
)
print("plots/ pushed to model repo")
PY

# ==============================================================================
# Phase 6: summary
# ==============================================================================
echo "=== Phase 6: summary ==="
echo "Model:   https://huggingface.co/${MODEL_REPO}"
echo "Dataset: https://huggingface.co/datasets/${DATASET_REPO}"

python - <<'PY' || echo "[warn] could not read W&B run URL"
import os
os.environ.setdefault("HOME", "/tmp")
try:
    import wandb
    api = wandb.Api()
    runs = api.runs("mutant-hunter-final", order="-created_at", per_page=1)
    for r in runs:
        print(f"W&B run: {r.url}")
        break
except Exception as e:
    print(f"[warn] wandb URL lookup failed: {e}")
PY

echo ""
echo "=== Done ==="
