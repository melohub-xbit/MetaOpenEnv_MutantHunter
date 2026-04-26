#!/usr/bin/env bash
# Submit the MutantHunter GRPO training run as an HF Job.
#
# Required local env vars:
#   HF_TOKEN          auto-injected by `hf jobs run` when --secrets HF_TOKEN
#                     is bare. Make sure you ran `hf auth login` first.
#   WANDB_API_KEY     forwarded as a plaintext --env (the --secrets KEY=VALUE
#                     form does not propagate reliably in the current CLI).
#                     The value will appear in the job's env/UI; rotate after
#                     if concerned. Or pre-store as a named HF secret and
#                     switch back to bare `--secrets WANDB_API_KEY`.

set -euo pipefail

: "${WANDB_API_KEY:?WANDB_API_KEY must be set in the local shell before submitting}"

hf jobs run \
    --flavor a100-large \
    --timeout 6h \
    --secrets HF_TOKEN \
    --env "WANDB_API_KEY=${WANDB_API_KEY}" \
    pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    bash -c 'apt-get update && apt-get install -y --no-install-recommends git curl && curl -fsSL https://raw.githubusercontent.com/melohub-xbit/MetaOpenEnv_MutantHunter/main/scripts/run_hf_job.sh | bash'
