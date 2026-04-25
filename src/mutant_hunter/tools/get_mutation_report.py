"""`get_mutation_report` tool: line-anchored summary of surviving mutants.

Critical anti-hack constraint: the agent must NEVER see the mutated source.
We surface only ``mutant_id``, ``line``, ``original``, and ``mutated`` so
the agent can't grep against patches.
"""

from __future__ import annotations

from mutant_hunter.models import State

_DEFAULT_CAP = 30


def get_mutation_report(state: State, *, cap: int = _DEFAULT_CAP) -> str:
    survivors = state.surviving_mutants
    if not survivors:
        return "No surviving baseline mutants — the existing suite already kills every one."
    cap = max(1, int(cap))
    shown = survivors[:cap]
    lines = [
        f"{m.mutant_id}, line {m.line}, {m.original} -> {m.mutated} ({m.operator})"
        for m in shown
    ]
    if len(survivors) > cap:
        lines.append(f"...<truncated: {len(survivors) - cap} more>...")
    return "\n".join(lines)
