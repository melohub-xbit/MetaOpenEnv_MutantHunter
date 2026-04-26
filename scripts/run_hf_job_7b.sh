#!/usr/bin/env bash
# Runs INSIDE the HF Job container. Self-contained: clones the repo, installs
# deps, and walks Phases 0..6. On any failure, the trap pushes whatever
# artifacts already landed in /tmp/results/ to the dataset repo under partial/.
#
# Required env vars (set via `hf jobs run --secret`):
#   HF_TOKEN
#   WANDB_API_KEY

set -euo pipefail

MODEL_REPO="jester1177/mutant-hunter-qwen-coder-7b-lora"
DATASET_REPO="jester1177/mutant-hunter-results"
RESULTS_DIR="/tmp/results"

push_partial() {
    rc=$?
    echo ""
    echo "[trap] caught exit code ${rc}; uploading whatever is in ${RESULTS_DIR}/ as partial/"
    if [ -d "${RESULTS_DIR}" ]; then
        python - <<'PY' || echo "[trap] partial upload itself failed; giving up"
import os
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/tmp/results",
    repo_id="jester1177/mutant-hunter-results",
    repo_type="dataset",
    path_in_repo="partial",
)
print("[trap] partial artifacts pushed to dataset repo under partial/")
PY
    else
        echo "[trap] no ${RESULTS_DIR}/ directory exists; nothing to push"
    fi
    exit "${rc}"
}
trap push_partial ERR

echo "=== Phase 0: setup (clean install) ==="
cd /tmp
if [ ! -d MetaOpenEnv_MutantHunter ]; then
    git clone https://github.com/melohub-xbit/MetaOpenEnv_MutantHunter.git
fi
cd MetaOpenEnv_MutantHunter
mkdir -p "${RESULTS_DIR}" "${RESULTS_DIR}/training" "${RESULTS_DIR}/plots"

# Install torch + torchvision + torchaudio together from official cu124 wheels
# (must move as a set: torchaudio in the base image pins torch==2.5.1, and
# pip will not upgrade it on its own → "torchaudio requires torch==X" conflict)
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify torch + torchvision import without ABI errors
python -c "
import torch
import torchvision
from torchvision import io as _io
print(f'torch={torch.__version__}, torchvision={torchvision.__version__}, CUDA={torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'No CUDA'
"

# Install project + extras
pip install --no-cache-dir -e ".[training]"

# Verify TRL imports cleanly
python -c "
import trl
from trl import GRPOTrainer, GRPOConfig
print(f'TRL {trl.__version__} GRPOTrainer importable')
"

pip install --no-cache-dir bitsandbytes wandb

# Verify full import chain that previously crashed
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import bitsandbytes
from trl import GRPOTrainer, GRPOConfig
print('All training-pipeline imports succeed')
"

python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python -c "import os; os.environ['HOME']='/tmp'; import wandb; wandb.login(key='${WANDB_API_KEY}')"

python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("jester1177/mutant-hunter-qwen-coder-7b-lora", repo_type="model", exist_ok=True)
api.create_repo("jester1177/mutant-hunter-results", repo_type="dataset", exist_ok=True)
print("repos ready")
PY

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
    echo "=== Skipping Phase 1 and Phase 2 (SKIP_BASELINES set) ==="
    echo "Downloading baseline JSONs from HF Hub..."
    python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="jester1177/mutant-hunter-results", filename="baseline_heuristic.json",
                repo_type="dataset", local_dir="/tmp/results")
hf_hub_download(repo_id="jester1177/mutant-hunter-results", filename="baseline_zeroshot.json",
                repo_type="dataset", local_dir="/tmp/results")
print("Baselines downloaded from HF Hub")
PY
fi

echo "=== Phase 3: 200-step GRPO training (~4h) ==="
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

echo "=== Phase 4: trained eval (~30 min) ==="
# Stash Phase 2 output so the next run does not clobber it.
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
