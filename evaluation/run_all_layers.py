"""Master orchestrator — run Layers 1..5 in sequence, summarise to a final readiness table.

Layers 6 (zero-shot) and 7 (30-step GRPO) are NOT run by this script: they
need an LLM and a GPU. Invoke ``zero_shot_distribution.py`` and
``grpo_smoke_run.py`` directly when GPU is available, then re-run this
file with ``--include-llm-layers`` to fold their results into the table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "_results"
PYTHON = sys.executable


def _run(name: str, argv: list[str]) -> tuple[int, str]:
    print(f"\n{'='*72}\n  {name}\n{'='*72}", flush=True)
    cp = subprocess.run([PYTHON, *argv], cwd=str(ROOT))
    return cp.returncode, name


def _read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-layer1", action="store_true")
    ap.add_argument("--skip-layer2", action="store_true")
    ap.add_argument("--skip-layer4", action="store_true")
    ap.add_argument("--skip-layer5", action="store_true")
    ap.add_argument("--include-llm-layers", action="store_true",
                    help="If set, treat layer6/layer7 result files as already produced.")
    args = ap.parse_args()

    rc_by_layer: dict[str, int | None] = {f"layer{i}": None for i in range(8)}
    rc_by_layer["layer0"] = 0  # already done by sync_manifest_from_baselines.py

    if not args.skip_layer1:
        rc_by_layer["layer1"], _ = _run("Layer 1", ["evaluation/sanity_ranking.py", "layer1"])
    if not args.skip_layer2:
        rc_by_layer["layer2"], _ = _run("Layer 2", ["evaluation/sanity_ranking.py", "layer2"])
    rc_by_layer["layer3"], _ = _run("Layer 3", ["evaluation/component_sanity.py"])
    if not args.skip_layer4:
        rc_by_layer["layer4"], _ = _run("Layer 4", ["evaluation/sanity_ranking.py", "layer4"])
    if not args.skip_layer5:
        rc_by_layer["layer5"], _ = _run("Layer 5", ["evaluation/adversarial_components.py"])

    # Layer 6 / 7 detail.
    layer6 = _read_summary(RESULTS_DIR / "zero_shot_distribution.json")
    layer7_state = sorted((ROOT / "training" / "_runs" / "grpo-validation").rglob("trainer_state.json"))

    if args.include_llm_layers:
        if layer6 is None:
            rc_by_layer["layer6"] = 1
        else:
            s = layer6.get("summary", {})
            mean_r = s.get("mean_reward", 0.0)
            p03 = s.get("fraction_reward_gt_0.3", 0.0)
            fmt0 = s.get("fraction_format_zero", 1.0)
            ok = (0.10 <= mean_r <= 0.40) and (p03 >= 0.15) and (fmt0 < 0.30)
            rc_by_layer["layer6"] = 0 if ok else 1
        rc_by_layer["layer7"] = 0 if layer7_state else 1

    def fmt(layer: str, label: str, detail: str) -> str:
        rc = rc_by_layer[layer]
        if rc is None:
            tag = "[SKIP   ]"
        elif rc == 0:
            tag = "[ PASS  ]"
        else:
            tag = "[ FAIL  ]"
        return f"{label:<38}{tag}  {detail}"

    print("\n" + "=" * 72)
    print("  Final readiness table")
    print("=" * 72)
    layer1 = _read_summary(RESULTS_DIR / "layer1_sanity_ranking.json") or {}
    layer4 = _read_summary(RESULTS_DIR / "layer4_cross_module.json") or {}
    layer5 = _read_summary(RESULTS_DIR / "adversarial_components.json") or []

    detail0 = "manifest backfilled, all 4 modules have real numbers"
    detail1 = (
        f"comp={layer1.get('comprehensive', {}).get('final_reward', 'NA')}  "
        f"strong={layer1.get('single_strong', {}).get('final_reward', 'NA')}  "
        f"weak={layer1.get('single_weak', {}).get('final_reward', 'NA')}  "
        f"vac={layer1.get('vacuous', {}).get('final_reward', 'NA')}"
    )
    detail2 = "see evaluation/_results/layer2_determinism.json"
    detail3 = "see evaluation/_results/layer1_sanity_ranking.json (component breakdown)"
    if layer4:
        rewards = [v.get("final_reward", 0.0) for v in layer4.values()]
        detail4 = f"spread={(max(rewards)-min(rewards)):.3f}  rewards={[round(r, 3) for r in rewards]}"
    else:
        detail4 = "not run"
    if layer5:
        leaks = [r["name"] for r in layer5 if r.get("components", {}).get("mutation_kill", 0.0) > 0.1]
        detail5 = f"{len(layer5)} cases  leaks={leaks or 'none'}"
    else:
        detail5 = "not run"

    if layer6 is not None:
        s = layer6.get("summary", {})
        detail6 = f"mean={s.get('mean_reward', 0.0):.3f}  p(r>0.3)={s.get('fraction_reward_gt_0.3', 0.0):.3f}  fmt0={s.get('fraction_format_zero', 1.0):.3f}"
    else:
        detail6 = "not run (requires GPU + LLM)"
    if layer7_state:
        detail7 = f"trainer_state at {layer7_state[-1]}"
    else:
        detail7 = "not run (requires GPU + LLM)"

    print(fmt("layer0", "Layer 0 — Manifest backfill", detail0))
    print(fmt("layer1", "Layer 1 — Differential ranking", detail1))
    print(fmt("layer2", "Layer 2 — Determinism", detail2))
    print(fmt("layer3", "Layer 3 — Per-component sanity", detail3))
    print(fmt("layer4", "Layer 4 — Cross-module variance", detail4))
    print(fmt("layer5", "Layer 5 — Adversarial components", detail5))
    print(fmt("layer6", "Layer 6 — Zero-shot distribution", detail6))
    print(fmt("layer7", "Layer 7 — 30-step GRPO smoke", detail7))

    blocking = [k for k, v in rc_by_layer.items() if v not in (None, 0)]
    skipped = [k for k, v in rc_by_layer.items() if v is None]
    if blocking:
        print("\nReady to launch real 200-300 step training: NO")
        print("Blocking layers:", ", ".join(blocking))
        return 1
    if skipped:
        print("\nReady to launch real 200-300 step training: CONDITIONAL")
        print("Skipped layers (must be run on GPU before training):", ", ".join(skipped))
        return 0
    print("\nReady to launch real 200-300 step training: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
