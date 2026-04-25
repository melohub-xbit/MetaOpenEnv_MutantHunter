"""Layer 5 — adversarial deep-dive with full component breakdown.

Re-runs every case from ``reward_hacking_tests.py`` against a real
``MutantHunterEnvironment`` and saves the full component dict (not just
the final reward) to ``evaluation/_results/adversarial_components.json``.

A leak is flagged when ``mutation_kill > 0.1`` for any adversarial case —
that would mean the rubric is rewarding a hostile submission for killing
mutants it should never have been allowed to run against.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from evaluation.reward_hacking_tests import CASES  # noqa: E402
from mutant_hunter.models import Action  # noqa: E402
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    rows: list[dict] = []
    env = MutantHunterEnvironment()
    leaks: list[str] = []
    over_ceiling: list[str] = []
    try:
        for name, test_code, expected_max in CASES:
            seed = hash(name) & 0xFFFF
            print(f"  case={name} seed={seed} ...", flush=True)
            t0 = time.time()
            entry: dict = {
                "name": name,
                "expected_max": expected_max,
                "seed": seed,
            }

            try:
                env.reset(seed=seed)
            except Exception as exc:
                entry["error"] = f"reset: {type(exc).__name__}: {exc}"
                rows.append(entry)
                continue

            try:
                action = Action(kind="submit_tests", test_code=test_code)
            except ValidationError as exc:
                entry["rejected_at_action"] = True
                entry["action_error"] = str(exc)[:200]
                entry["final_reward"] = 0.0
                entry["components"] = {
                    "mutation_kill": 0.0,
                    "coverage_delta": 0.0,
                    "format": 0.0,
                    "parsimony": 0.0,
                }
                entry["no_regression_gate"] = 0.0
                entry["elapsed_s"] = round(time.time() - t0, 2)
                rows.append(entry)
                print(f"    rejected_at_action  reward=0.0", flush=True)
                continue

            try:
                obs = env.step(action)
            except Exception as exc:
                entry["error"] = f"step: {type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()[:1000]
                rows.append(entry)
                print(f"    ERROR: {entry['error']}", flush=True)
                continue

            md = obs.metadata or {}
            entry["final_reward"] = float(obs.reward or 0.0)
            entry["components"] = dict(md.get("components") or {})
            entry["no_regression_gate"] = md.get("no_regression_gate")
            entry["killed_by_new_only"] = md.get("killed_by_new_only")
            entry["new_coverage"] = md.get("new_coverage")
            entry["new_tests_pass_clean"] = md.get("new_tests_pass_clean")
            entry["status"] = md.get("status")
            entry["elapsed_s"] = round(time.time() - t0, 2)
            rows.append(entry)

            kill = entry["components"].get("mutation_kill", 0.0)
            if kill > 0.1:
                leaks.append(f"{name}: mutation_kill={kill:.4f}")
            if entry["final_reward"] > expected_max + 1e-6:
                over_ceiling.append(
                    f"{name}: reward={entry['final_reward']:.4f} > ceiling={expected_max}"
                )
            print(
                f"    reward={entry['final_reward']:.4f}"
                f" kill={kill:.4f}"
                f" gate={entry['no_regression_gate']}"
                f" t={entry['elapsed_s']}s",
                flush=True,
            )
    finally:
        env.close()

    out_path = RESULTS_DIR / "adversarial_components.json"
    out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\n[layer5] wrote {out_path}", flush=True)

    print()
    if leaks:
        print("[layer5] FAIL — mutation_kill leaked > 0.1 on:")
        for entry in leaks:
            print(f"  - {entry}")
    if over_ceiling:
        print("[layer5] FAIL — final_reward exceeded expected ceiling on:")
        for entry in over_ceiling:
            print(f"  - {entry}")
    if not leaks and not over_ceiling:
        print("[layer5] PASS — no kill leaks, every case at or below its reward ceiling.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
