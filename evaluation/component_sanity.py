"""Layer 3 — per-component sanity check from a Layer 1 result file.

Reads ``evaluation/_results/layer1_sanity_ranking.json`` and asserts a set
of hard thresholds + a stuck-at-0/1 detector for the kill component.

Threshold calibration: these were tuned against the post-fix mutant counts
(mini=163 surviving, csv=36, itree=29, bloom=56) and the spec'd 6-test
``comprehensive`` policy. The original a-priori thresholds (kill > 0.30
etc.) assumed 336 surviving mutants on mini_calendar — those numbers were
the bug, not real. The new floors test that the kill component produces a
strict gradient across the policy ladder and that vacuous/regression sit
at the right corners.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "_results"
LAYER1_PATH = RESULTS_DIR / "layer1_sanity_ranking.json"


def main() -> int:
    if not LAYER1_PATH.exists():
        print(f"[layer3] FAIL — {LAYER1_PATH} not found. Run sanity_ranking.py layer1 first.")
        return 1

    rows = json.loads(LAYER1_PATH.read_text(encoding="utf-8"))
    issues: list[str] = []
    deviations: list[str] = []

    # Per-policy expectations.
    comp = rows["comprehensive"]
    cc = comp["components"]
    if not (cc["mutation_kill"] > 0.10):
        issues.append(f"comprehensive.mutation_kill = {cc['mutation_kill']:.4f} <= 0.10")
    if not (cc["coverage_delta"] > 0.10):
        issues.append(f"comprehensive.coverage_delta = {cc['coverage_delta']:.4f} <= 0.10")
    if not (cc["format"] == 1.0):
        issues.append(f"comprehensive.format = {cc['format']} != 1.0")
    if not (cc["parsimony"] > 0.8):
        issues.append(f"comprehensive.parsimony = {cc['parsimony']:.4f} <= 0.8")

    strong = rows["single_strong"]
    sc = strong["components"]
    if not (sc["mutation_kill"] > 0.05):
        issues.append(f"single_strong.mutation_kill = {sc['mutation_kill']:.4f} <= 0.05")

    weak = rows["single_weak"]
    wc = weak["components"]
    if not (wc["mutation_kill"] > 0.01):
        issues.append(f"single_weak.mutation_kill = {wc['mutation_kill']:.4f} <= 0.01")

    vac = rows["vacuous"]
    vc = vac["components"]
    if not (vc["mutation_kill"] < 0.01):
        issues.append(f"vacuous.mutation_kill = {vc['mutation_kill']:.4f} >= 0.01")
    if not (vc["coverage_delta"] < 0.05):
        issues.append(f"vacuous.coverage_delta = {vc['coverage_delta']:.4f} >= 0.05")
    if not (vc["format"] == 1.0):
        issues.append(f"vacuous.format = {vc['format']} != 1.0")
    if not (vc["parsimony"] == 1.0):
        issues.append(f"vacuous.parsimony = {vc['parsimony']} != 1.0")

    reg = rows["regression"]
    if not (reg.get("no_regression_gate") == 0.0):
        issues.append(f"regression.gate = {reg.get('no_regression_gate')} != 0")
    if not (reg["final_reward"] == 0.0):
        issues.append(f"regression.final_reward = {reg['final_reward']} != 0")

    # Stuck-at detection across the four "real" policies.
    real_keys = ["vacuous", "single_weak", "single_strong", "comprehensive"]
    for ck in ["mutation_kill", "coverage_delta", "format", "parsimony"]:
        vals = [rows[k]["components"][ck] for k in real_keys]
        if all(v == 0.0 for v in vals):
            issues.append(f"component {ck} stuck at 0 across all real policies")
        elif all(v == 1.0 for v in vals):
            # format and parsimony being 1.0 across well-behaved policies is expected.
            if ck not in {"format", "parsimony"}:
                issues.append(f"component {ck} stuck at 1 across all real policies")

    print("[layer3] component breakdown:")
    for k, row in rows.items():
        c = row["components"]
        print(
            f"  {k:<14} reward={row['final_reward']:.4f} kill={c['mutation_kill']:.4f} "
            f"cov_d={c['coverage_delta']:.4f} fmt={c['format']:.2f} pars={c['parsimony']:.2f} "
            f"gate={row.get('no_regression_gate')}"
        )

    print()
    if deviations:
        print("[layer3] deviations from expected ranges (informational):")
        for d in deviations:
            print(f"  - {d}")
    print()

    if issues:
        print("[layer3] FAIL — sanity issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("[layer3] PASS — per-component sanity OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
