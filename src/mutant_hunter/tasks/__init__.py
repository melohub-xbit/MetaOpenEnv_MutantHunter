"""Task generation + curriculum.

Modules:

    generator.py   — deterministic episode sampling: (repo, module, mutation_set, seed)
    curriculum.py  — three difficulty tiers + advancement policy based on rolling reward
    seeds/         — JSON files holding hand-picked deterministic eval episodes

The eval seed set under `seeds/eval_set_v1.json` is the load-bearing artifact
for any "before vs after" plot — it is **never** used for training and must
remain stable across runs.
"""

from __future__ import annotations

from pathlib import Path

TASKS_ROOT = Path(__file__).resolve().parent
SEEDS_ROOT = TASKS_ROOT / "seeds"
EVAL_SET_V1 = SEEDS_ROOT / "eval_set_v1.json"

__all__ = ["TASKS_ROOT", "SEEDS_ROOT", "EVAL_SET_V1"]
