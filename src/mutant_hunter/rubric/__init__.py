"""Reward functions composing the final scalar fed back to GRPO.

Five components, combined in `compose.py`:

    mutation_kill   (0.60)  — fraction of baseline-surviving mutants killed
    coverage_delta  (0.20)  — improvement in line coverage over baseline
    format          (0.15)  — pytest validity + forbidden-pattern check
    parsimony       (0.05)  — anti-bloat: penalise oversize tests

A multiplicative `no_regression` gate (∈ {0.0, 1.0}) sits in front of the
weighted sum: if any submitted test fails on the unmodified code, the final
reward collapses to zero. This is the single most important shield against
reward hacking; do not weaken it.
"""

from __future__ import annotations

from .compose import RewardBreakdown, compose_reward

__all__ = ["RewardBreakdown", "compose_reward"]
