#!/usr/bin/env bash
hf jobs run \
    --hardware a100-large \
    --image "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel" \
    --secret HF_TOKEN \
    --secret WANDB_API_KEY \
    --timeout 6h \
    -- \
    bash -c 'apt-get update && apt-get install -y --no-install-recommends git curl && curl -fsSL https://raw.githubusercontent.com/jester1177/MetaOpenEnv_MutantHunter/main/scripts/run_hf_job.sh | bash'
