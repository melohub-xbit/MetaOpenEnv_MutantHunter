"""Reward component: mutation kill rate on baseline-surviving mutants.

The agent only gets credit for killing mutants that the EXISTING suite
failed to kill. This is the primary signal — if the agent's tests don't
add any mutation-kill power, this component is 0.
"""

from __future__ import annotations


def reward_mutation_kill(*, baseline_surviving: int, killed_by_new_only: int) -> float:
    if baseline_surviving <= 0:
        # No surviving mutants means the existing suite is already perfect.
        # Return 0.5 (neutral) — the agent didn't add value but isn't punished.
        return 0.5
    ratio = killed_by_new_only / baseline_surviving
    return max(0.0, min(1.0, ratio))
