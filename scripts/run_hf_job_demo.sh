#!/usr/bin/env bash
# In-context demonstration learning eval — runs INSIDE the HF Job container.
# Self-contained: clones the repo, installs deps, mines demonstrations,
# and runs the zero-shot eval with the demos spliced into each prompt.
#
# Required env vars (set via `hf jobs run --secrets` / `--env`):
#   HF_TOKEN              — HF write token
#   WANDB_API_KEY         — W&B API key (kept for parity; this script does
#                           not currently log to W&B)
#
# Optional env vars:
#   USE_DEMONSTRATIONS=1  — enable demo injection (default ON; set to 0 to
#                           run a plain zero-shot eval as a control)
#   DEMO_EPISODES=15      — number of episodes to run
#   DEMO_MODEL=...        — override the eval model (default Qwen2.5-Coder-7B)

set -euo pipefail

DATASET_REPO="jester1177/mutant-hunter-results"
RESULTS_DIR="/tmp/results"
DEMO_EPISODES="${DEMO_EPISODES:-15}"
DEMO_MODEL="${DEMO_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
USE_DEMONSTRATIONS="${USE_DEMONSTRATIONS:-1}"

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
    path_in_repo="partial_demo",
)
print("[trap] partial artifacts pushed under partial_demo/")
PY
    else
        echo "[trap] no ${RESULTS_DIR}/ directory; nothing to push"
    fi
    exit "${rc}"
}
trap push_partial ERR

echo "=== Phase 0: setup ==="
cd /tmp
[ -d MetaOpenEnv_MutantHunter ] || git clone https://github.com/melohub-xbit/MetaOpenEnv_MutantHunter.git
cd MetaOpenEnv_MutantHunter
mkdir -p "${RESULTS_DIR}"

pip install --no-cache-dir huggingface_hub

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

python -c "
import torch
maj, mn = (int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
assert (maj, mn) >= (2, 6), f'torch is {torch.__version__}, need >=2.6'
assert torch.cuda.is_available() and torch.cuda.device_count() > 0, 'torch reports no CUDA'
print(f'torch={torch.__version__}, devices={torch.cuda.device_count()} — OK')
"

# Eval extras only — we don't train here.
pip install --no-cache-dir -e ".[training]"
pip install --no-cache-dir bitsandbytes wandb

python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python - <<'PY'
from huggingface_hub import HfApi
HfApi().create_repo("jester1177/mutant-hunter-results", repo_type="dataset", exist_ok=True)
print("dataset repo ready")
PY

echo "=== Phase 1: mine demonstrations ==="
# Pulls baseline_zeroshot.json from the dataset repo, validates GOOD examples
# with pytest, and writes training/data/demonstrations.json.
python training/mine_demonstrations.py
cp training/data/demonstrations.json "${RESULTS_DIR}/demonstrations.json"

python - <<'PY'
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="/tmp/results/demonstrations.json",
    path_in_repo="demonstrations.json",
    repo_id="jester1177/mutant-hunter-results",
    repo_type="dataset",
)
print("demonstrations.json pushed")
PY

echo "=== Phase 2: zero-shot eval (USE_DEMONSTRATIONS=${USE_DEMONSTRATIONS}) ==="

DEMO_FLAG=""
OUT_NAME="zeroshot_demo_eval.json"
if [ "${USE_DEMONSTRATIONS}" = "1" ]; then
    DEMO_FLAG="--use-demonstrations training/data/demonstrations.json"
    echo "[demo] in-context demonstrations ENABLED"
else
    OUT_NAME="zeroshot_control_eval.json"
    echo "[demo] in-context demonstrations DISABLED — running plain zero-shot control"
fi

python evaluation/zero_shot_distribution.py \
    --episodes "${DEMO_EPISODES}" \
    --model "${DEMO_MODEL}" \
    --max-new-tokens 1024 \
    --seed-start 0 \
    --device auto \
    ${DEMO_FLAG}

cp evaluation/_results/zero_shot_distribution.json "${RESULTS_DIR}/${OUT_NAME}"

python - <<PY
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="/tmp/results/${OUT_NAME}",
    path_in_repo="${OUT_NAME}",
    repo_id="jester1177/mutant-hunter-results",
    repo_type="dataset",
)
print("${OUT_NAME} pushed")
PY

echo "=== Phase 3: summary ==="
python - <<PY
import json
data = json.load(open("/tmp/results/${OUT_NAME}"))
s = data["summary"]
print(f"n={s['n_episodes']} mean={s['mean_reward']:.4f} "
      f"std={s['std_reward']:.4f} "
      f"p_gt_0.3={s['fraction_reward_gt_0.3']:.3f} "
      f"p_gate_zero={s['fraction_regression_gate_zero']:.3f}")
PY

echo "Dataset: https://huggingface.co/datasets/${DATASET_REPO}"
echo ""
echo "=== Done ==="
