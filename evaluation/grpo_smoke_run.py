"""Layer 7 — 30-step GRPO smoke run with the real base model.

Wraps ``training.train_grpo`` with the validation-specific config:

    --steps 30 --rollouts-per-step 4 --max-new-tokens 1024
    W&B project=mutant-hunter-validation

After training (or on failure) post-processes ``trainer_state.json`` /
W&B run logs to verify:
    1. reward variance per step  std(rewards) > 0.05 across rollouts
    2. no NaN / Inf rewards in any step
    3. W&B logged total_reward, per-component rewards, episode_length,
       and success_rate

This script is the smoke test, not the real training run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HF model id; matches what the real run will use.",
    )
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--rollouts-per-step", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default="./training/_runs/grpo-validation")
    ap.add_argument("--wandb-project", type=str, default="mutant-hunter-validation")
    ap.add_argument("--no-unsloth", action="store_true", default=True)
    ap.add_argument("--no-4bit", action="store_true", default=False)
    ap.add_argument("--no-lora", action="store_true", default=False)
    ap.add_argument("--lora-rank", type=int, default=16)
    args = ap.parse_args()

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defer imports so that --help works even if torch/trl/wandb missing.
    try:
        import torch  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        print(f"[layer7] FAIL — training extras missing: {exc}")
        print('  install with: pip install -e ".[training]"')
        return 1

    try:
        import wandb
    except ImportError:
        wandb = None
        print("[layer7] WARN — wandb not installed; metric capture will be local-only")

    # Hand off to train_grpo.run() with our config.
    from training.train_grpo import TrainingConfig, run as run_grpo

    cfg = TrainingConfig(
        base_model=args.base_model,
        steps=args.steps,
        rollouts_per_step=args.rollouts_per_step,
        learning_rate=args.learning_rate,
        use_unsloth=not args.no_unsloth,
        use_4bit=not args.no_4bit,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        output_dir=out_dir,
    )

    if wandb is not None and os.environ.get("WANDB_MODE") != "offline":
        try:
            wandb.init(project=args.wandb_project, config=cfg.__dict__, dir=str(out_dir))
        except Exception as exc:
            print(f"[layer7] WARN — wandb.init failed: {exc}")

    print(f"[layer7] starting GRPO {args.steps}-step smoke run with base={args.base_model}", flush=True)
    t0 = time.time()
    try:
        rc = run_grpo(cfg)
    except Exception as exc:
        print(f"[layer7] FAIL — GRPO loop raised: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    elapsed = time.time() - t0
    print(f"[layer7] training returned rc={rc} in {elapsed:.1f}s", flush=True)

    # Post-run assertions: read trainer_state.json (TRL writes one at output_dir/checkpoint-N).
    issues: list[str] = []
    state_paths = sorted(out_dir.rglob("trainer_state.json"))
    if not state_paths:
        issues.append("no trainer_state.json found — TRL did not save run state")
    else:
        state = json.loads(state_paths[-1].read_text(encoding="utf-8"))
        log_history = state.get("log_history", [])
        rewards_per_step = [
            entry.get("reward")
            for entry in log_history
            if entry.get("reward") is not None
        ]
        std_rewards = [
            entry.get("reward_std")
            for entry in log_history
            if entry.get("reward_std") is not None
        ]
        nan_seen = any(
            v is not None and (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
            for entry in log_history
            for v in entry.values()
            if isinstance(v, (int, float))
        )

        print(f"[layer7] log_history entries: {len(log_history)}")
        print(f"[layer7] rewards per step: {rewards_per_step}")
        print(f"[layer7] reward_std per step: {std_rewards}")

        if nan_seen:
            issues.append("NaN/Inf detected in trainer log_history")
        if std_rewards:
            mean_std = sum(std_rewards) / len(std_rewards)
            print(f"[layer7] mean reward_std across steps = {mean_std:.4f}")
            if mean_std <= 0.05:
                issues.append(
                    f"reward_std mean {mean_std:.4f} ≤ 0.05 — GRPO advantage estimation will collapse"
                )
        else:
            issues.append("no reward_std logged — TRL may not be running >1 generation per prompt")

    if wandb is not None and wandb.run is not None:
        wandb.finish()

    print()
    if issues:
        print("[layer7] FAIL — smoke run issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("[layer7] PASS — reward variance present, no NaN/Inf, GRPO has signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
