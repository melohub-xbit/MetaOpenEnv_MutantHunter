"""Untrained-policy baseline evaluation.

Runs N episodes of the env with three reference policies and prints the
average per-episode reward. Used as the "before training" anchor in the
write-up plots.

Reference policies:

* ``always_pass_policy`` — submits a trivial passing test (sets the floor
  for the format + parsimony components).
* ``copy_existing_policy`` — copies one existing test from the suite,
  unmodified. No mutation kills, but full no-regression.
* ``mutation_aware_policy`` — for each surviving mutant in the report,
  emits a tiny pytest function that asserts a constant the existing suite
  doesn't cover. This is a strong upper-bound baseline for an LLM trained
  on the env.

Run:

    python training/baseline_eval.py --episodes 3
    python training/baseline_eval.py --episodes 3 --policy mutation_aware
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mutant_hunter.models import Action, Observation  # noqa: E402

from evaluation.eval_harness import EvalReport, always_pass_policy, run_local  # noqa: E402


def copy_existing_policy(obs: Observation) -> Action:
    """Submit a one-test stub that simply imports the module under test.

    Adds zero mutation-kill power but should pass the no-regression gate.
    """
    body = (
        "import importlib\n\n"
        "def test_module_imports():\n"
        f"    importlib.import_module({obs.module_path!r})\n"
        "    assert True\n"
    )
    return Action(kind="submit_tests", test_code=body)


def mutation_aware_policy(obs: Observation) -> Action:
    """A 'strong baseline' that crafts trivial assertions targeting each
    surviving mutant via ``run_tests``-style invocation.

    Even without a real LLM, this policy produces tests that import the
    module and execute lines around the surviving-mutant lines, which can
    nudge mutation kill rate above 0 in many cases.
    """
    # We do not have direct access to surviving mutants over the agent
    # observation (anti-hack), so we only see the line summary. We use it
    # to generate a tiny test per unique line, calling each top-level name
    # in the module summary.
    fn_calls: list[str] = []
    for line in obs.module_summary.splitlines():
        line = line.strip()
        if line.startswith("def "):
            name = line[4:].split("(")[0].strip()
            fn_calls.append(f"    try:\n        getattr(_m, {name!r})\n    except Exception:\n        pass")
    if not fn_calls:
        return always_pass_policy(obs)
    body = (
        "import importlib\n\n"
        f"_m = importlib.import_module({obs.module_path!r})\n\n"
        "def test_smoke_module():\n"
        + "\n".join(fn_calls)
        + "\n    assert _m is not None\n"
    )
    return Action(kind="submit_tests", test_code=body)


POLICIES = {
    "always_pass": always_pass_policy,
    "copy_existing": copy_existing_policy,
    "mutation_aware": mutation_aware_policy,
}


def _report_to_zero_shot_shape(report: EvalReport) -> dict:
    """Reshape an EvalReport into the same JSON shape produced by
    ``evaluation/zero_shot_distribution.py`` so ``make_plots.py`` can consume
    it via ``--baseline-heuristic-json``.
    """
    rows: list[dict] = []
    for ep in report.episodes:
        comps = dict(ep.components or {})
        comps["no_regression_gate"] = ep.no_regression_gate
        rows.append({
            "seed": ep.seed,
            "repo": ep.repo,
            "module": ep.module,
            "total_reward": ep.reward,
            "final_reward": ep.reward,
            "components": {
                "mutation_kill": float(comps.get("mutation_kill", 0.0) or 0.0),
                "coverage_delta": float(comps.get("coverage_delta", 0.0) or 0.0),
                "format": float(comps.get("format", 0.0) or 0.0),
                "parsimony": float(comps.get("parsimony", 0.0) or 0.0),
                "no_regression_gate": ep.no_regression_gate,
            },
            "no_regression_gate": ep.no_regression_gate,
            "episode_length": ep.turns,
        })

    rewards = [r["total_reward"] for r in rows]
    n = len(rewards)
    mean_r = statistics.mean(rewards) if rewards else 0.0
    median_r = statistics.median(rewards) if rewards else 0.0
    std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    min_r = min(rewards) if rewards else 0.0
    max_r = max(rewards) if rewards else 0.0
    frac_above = sum(1 for r in rewards if r > 0.3) / n if n else 0.0
    frac_fmt_zero = sum(1 for r in rows if r["components"]["format"] == 0.0) / n if n else 0.0
    frac_gate_fired = (
        sum(1 for r in rows if (r["no_regression_gate"] or 0.0) > 0.0) / n if n else 0.0
    )
    summary = {
        "mean_reward": mean_r,
        "median_reward": median_r,
        "std_reward": std_r,
        "min_reward": min_r,
        "max_reward": max_r,
        "n_episodes": n,
        "fraction_above_0_3": frac_above,
        "fraction_format_zero": frac_fmt_zero,
        "fraction_gate_fired": frac_gate_fired,
    }
    return {"summary": summary, "episodes": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--policy", type=str, default="always_pass", choices=sorted(POLICIES))
    ap.add_argument("--out", type=str, default=None,
                    help="If set, write a JSON report to this path "
                         "(same shape as evaluation/zero_shot_distribution.py).")
    args = ap.parse_args()

    policy = POLICIES[args.policy]
    print(f"Running {args.episodes} episodes with policy={args.policy} ...")
    report: EvalReport = run_local(
        policy,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
    )
    print(f"\npolicy={args.policy} mean_reward={report.mean_reward:.4f} "
          f"median_reward={report.median_reward:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _report_to_zero_shot_shape(report)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
