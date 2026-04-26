#!/usr/bin/env bash
# Inference-time self-correction eval — runs INSIDE the HF Job container.
# Self-contained: clones the repo, installs deps, and runs the zero-shot eval
# with --max-retries 2 (so up to 3 generation attempts per episode, with
# error-feedback in the prompt on retry). NO training. NO baselines re-run.
#
# Required env vars (via `hf jobs run --secrets` / `--env`):
#   HF_TOKEN           — HF write token
#   WANDB_API_KEY      — kept for parity (not used by this script)
#
# Optional env vars:
#   MAX_RETRIES=2      — retry budget passed to zero_shot_distribution.py
#   RETRY_EPISODES=15  — number of episodes
#   RETRY_MODEL=...    — override eval model (default Qwen2.5-Coder-7B)

set -euo pipefail

DATASET_REPO="jester1177/mutant-hunter-results"
RESULTS_DIR="/tmp/results"
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_EPISODES="${RETRY_EPISODES:-15}"
RETRY_MODEL="${RETRY_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

push_partial() {
    rc=$?
    echo ""
    echo "[trap] caught exit code ${rc}; uploading ${RESULTS_DIR}/ as partial_retry/"
    if [ -d "${RESULTS_DIR}" ]; then
        python - <<'PY' || echo "[trap] partial upload itself failed; giving up"
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/tmp/results",
    repo_id="jester1177/mutant-hunter-results",
    repo_type="dataset",
    path_in_repo="partial_retry",
)
print("[trap] partial artifacts pushed under partial_retry/")
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

pip install --no-cache-dir -e ".[training]"
pip install --no-cache-dir bitsandbytes

python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
python - <<'PY'
from huggingface_hub import HfApi
HfApi().create_repo("jester1177/mutant-hunter-results", repo_type="dataset", exist_ok=True)
print("dataset repo ready")
PY

echo "=== Phase 1: zero-shot eval with retries (max=${MAX_RETRIES}, episodes=${RETRY_EPISODES}) ==="
python evaluation/zero_shot_distribution.py \
    --episodes "${RETRY_EPISODES}" \
    --model "${RETRY_MODEL}" \
    --max-new-tokens 1024 \
    --seed-start 0 \
    --device auto \
    --max-retries "${MAX_RETRIES}" \
    --retry-output "${RESULTS_DIR}/retry_stats.json"

cp evaluation/_results/zero_shot_distribution.json "${RESULTS_DIR}/baseline_zeroshot_with_retries.json"

python - <<'PY'
from huggingface_hub import upload_file
for fname in ("baseline_zeroshot_with_retries.json", "retry_stats.json"):
    upload_file(
        path_or_fileobj=f"/tmp/results/{fname}",
        path_in_repo=fname,
        repo_id="jester1177/mutant-hunter-results",
        repo_type="dataset",
    )
    print(f"{fname} pushed")
PY

echo "=== Summary ==="
python - <<'PY'
import json
data = json.load(open("/tmp/results/baseline_zeroshot_with_retries.json"))
s = data["summary"]
print(f"n={s['n_episodes']} mean={s['mean_reward']:.4f} "
      f"std={s['std_reward']:.4f} "
      f"p_gt_0.3={s['fraction_reward_gt_0.3']:.3f} "
      f"p_format_zero={s['fraction_format_zero']:.3f} "
      f"p_gate_zero={s['fraction_regression_gate_zero']:.3f}")
print(f"max_retries={s['max_retries']} "
      f"used_retries={s['n_episodes_with_retries']} "
      f"recovered={s['n_episodes_success_after_retry']} "
      f"failed_all={s['n_episodes_failed_all_retries']}")
PY

echo "Dataset: https://huggingface.co/datasets/${DATASET_REPO}"
echo ""
echo "=== Done ==="
