"""Generate README plots from training + eval JSON outputs.

Produces 4 PNGs in --out-dir:

    reward_curve.png            — total reward vs training step
    per_reward_breakdown.png    — each rubric component vs training step
    mutation_kill_rate.png      — mutation_kill component vs training step
    baseline_vs_trained.png     — bar chart: heuristic / untrained / trained
                                  on each of {final_reward, mutation_kill,
                                  format, no_regression}

Inputs
------

* ``--baseline-heuristic-json``  output of ``training/baseline_eval.py`` re-run
  with ``--out`` (or any JSON shaped like ``EvalReport.to_json()``).
* ``--baseline-zeroshot-json``   output of ``evaluation/zero_shot_distribution.py``
  for the *untrained* model.
* ``--trained-eval-json``        output of ``evaluation/zero_shot_distribution.py``
  re-run with ``--lora-path`` pointing at the trained adapter.
* ``--wandb-run-path``           a W&B run path ``entity/project/run_id`` OR
  a path to a local W&B run directory containing ``wandb-history.jsonl``
  (or any ``.jsonl`` with one row per logged step).

Run
---

    python evaluation/make_plots.py \
        --baseline-heuristic-json evaluation/_results/heuristic.json \
        --baseline-zeroshot-json  evaluation/_results/zero_shot_distribution.json \
        --trained-eval-json       evaluation/_results/trained_distribution.json \
        --wandb-run-path          entity/mutant-hunter/abc123 \
        --out-dir plots/
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


COMPONENT_KEYS = ("mutation_kill", "coverage_delta", "format", "parsimony")
BAR_METRICS = ("final_reward", "mutation_kill", "format", "no_regression")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_rows(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the two JSON shapes we accept into a list of episode dicts."""
    if "episodes" in blob and isinstance(blob["episodes"], list):
        return blob["episodes"]
    return []


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _bar_metrics_from_eval(blob: dict[str, Any]) -> dict[str, float]:
    """Pull the four bar-chart metrics from an EvalReport / zero_shot blob."""
    rows = _episode_rows(blob)
    if not rows:
        return {k: 0.0 for k in BAR_METRICS}
    final_reward = _mean([float(r.get("final_reward", r.get("reward", 0.0)) or 0.0) for r in rows])
    mutation_kill = _mean([float((r.get("components") or {}).get("mutation_kill", 0.0) or 0.0) for r in rows])
    fmt = _mean([float((r.get("components") or {}).get("format", 0.0) or 0.0) for r in rows])
    gate = _mean([float(r.get("no_regression_gate", 0.0) or 0.0) for r in rows])
    return {
        "final_reward": final_reward,
        "mutation_kill": mutation_kill,
        "format": fmt,
        "no_regression": gate,
    }


def _load_wandb_history(spec: str) -> list[dict[str, Any]]:
    """Return list of step-rows. Accepts a wandb run path or a local jsonl path."""
    p = Path(spec)
    if p.exists():
        if p.is_dir():
            candidates = list(p.glob("wandb-history.jsonl")) + list(p.glob("*.jsonl"))
            if not candidates:
                raise FileNotFoundError(f"no .jsonl history under {p}")
            p = candidates[0]
        rows: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"--wandb-run-path={spec!r} is not a local file and wandb is not installed: {exc}"
        ) from exc
    api = wandb.Api()
    run = api.run(spec)
    return [dict(row) for row in run.scan_history()]


def _series(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for i, row in enumerate(history):
        if key in row and row[key] is not None:
            try:
                ys.append(float(row[key]))
            except (TypeError, ValueError):
                continue
            xs.append(int(row.get("_step", i)))
    return xs, ys


def _plot_reward_curve(history: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    xs, ys = _series(history, "reward")
    if not ys:
        xs, ys = _series(history, "train/reward")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, color="C0")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Total reward over training")
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5, -0.02,
        "Caption: per-step mean of the rubric-composed scalar reward (range [0, 1]).",
        ha="center", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_reward_breakdown(history: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = 0
    for key in COMPONENT_KEYS:
        xs, ys = _series(history, key)
        if not ys:
            xs, ys = _series(history, f"components/{key}")
        if not ys:
            xs, ys = _series(history, f"reward/{key}")
        if ys:
            ax.plot(xs, ys, label=key)
            plotted += 1
    ax.set_xlabel("Training step")
    ax.set_ylabel("Component value (range [0, 1])")
    ax.set_title("Reward components over training")
    if plotted:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5, -0.02,
        "Caption: each rubric component logged per step. Missing series mean the "
        "training run did not log that component name.",
        ha="center", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mutation_kill_rate(history: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    xs, ys = _series(history, "mutation_kill")
    if not ys:
        xs, ys = _series(history, "components/mutation_kill")
    if not ys:
        xs, ys = _series(history, "reward/mutation_kill")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, color="C3")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mutation kill component (range [0, 1])")
    ax.set_title("Mutation kill rate over training")
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5, -0.02,
        "Caption: fraction of baseline-surviving mutants killed by the agent's "
        "submission, averaged per step.",
        ha="center", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_baseline_vs_trained(
    heuristic: dict[str, float],
    untrained: dict[str, float],
    trained: dict[str, float],
    out: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(BAR_METRICS)
    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, [heuristic[k] for k in labels], width, label="heuristic baseline", color="C7")
    ax.bar(x,         [untrained[k] for k in labels], width, label="untrained Qwen",     color="C0")
    ax.bar(x + width, [trained[k]   for k in labels], width, label="trained Qwen (LoRA)", color="C2")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean value across eval episodes (range [0, 1])")
    ax.set_title("Baselines vs trained policy")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.text(
        0.5, -0.02,
        "Caption: per-metric mean over the same eval seeds. "
        "no_regression is the gate value (0 or 1); others are continuous components.",
        ha="center", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-heuristic-json", type=Path, required=True)
    ap.add_argument("--baseline-zeroshot-json", type=Path, required=True)
    ap.add_argument("--trained-eval-json", type=Path, required=True)
    ap.add_argument("--wandb-run-path", type=str, default=None,
                    help="wandb 'entity/project/run_id' OR a local path to a wandb history .jsonl / run dir")
    ap.add_argument("--training-log-json", type=str, default=None,
                    help="Local JSONL of per-step training metrics. Alias for --wandb-run-path "
                         "when given a local file.")
    ap.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = ap.parse_args()

    history_spec = args.wandb_run_path or args.training_log_json
    if not history_spec:
        ap.error("must provide either --wandb-run-path or --training-log-json")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    heuristic = _bar_metrics_from_eval(_read_json(args.baseline_heuristic_json))
    untrained = _bar_metrics_from_eval(_read_json(args.baseline_zeroshot_json))
    trained   = _bar_metrics_from_eval(_read_json(args.trained_eval_json))

    history = _load_wandb_history(history_spec)

    _plot_reward_curve(history,           args.out_dir / "reward_curve.png")
    _plot_per_reward_breakdown(history,   args.out_dir / "per_reward_breakdown.png")
    _plot_mutation_kill_rate(history,     args.out_dir / "mutation_kill_rate.png")
    _plot_baseline_vs_trained(heuristic, untrained, trained,
                              args.out_dir / "baseline_vs_trained.png")

    print(f"Wrote 4 plots to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
