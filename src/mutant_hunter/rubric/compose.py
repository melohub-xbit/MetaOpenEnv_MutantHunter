"""Reward composition: weighted sum, gated by no-regression.

    final = no_regression * (
        0.60 * mutation_kill +
        0.20 * coverage_delta +
        0.15 * format +
        0.05 * parsimony
    )

The no-regression term is multiplicative — if any submitted test fails on
the unmodified module, the final reward collapses to 0. This is the
single most important shield against reward hacking.
"""

from __future__ import annotations

from dataclasses import dataclass

from .reward_coverage_delta import reward_coverage_delta
from .reward_format import reward_format
from .reward_mutation_kill import reward_mutation_kill
from .reward_no_regression import reward_no_regression
from .reward_parsimony import reward_parsimony

WEIGHTS = {
    "mutation_kill": 0.60,
    "coverage_delta": 0.20,
    "format": 0.15,
    "parsimony": 0.05,
}


@dataclass(frozen=True)
class RewardBreakdown:
    final: float
    components: dict[str, float]
    no_regression_gate: float


def compose_reward(
    *,
    baseline_surviving: int,
    killed_by_new_only: int,
    baseline_coverage: float,
    new_coverage: float,
    parses: bool,
    contains_forbidden: bool,
    new_tests_pass_clean: bool,
    test_code: str,
) -> RewardBreakdown:
    components = {
        "mutation_kill": reward_mutation_kill(
            baseline_surviving=baseline_surviving,
            killed_by_new_only=killed_by_new_only,
        ),
        "coverage_delta": reward_coverage_delta(
            baseline_coverage=baseline_coverage,
            new_coverage=new_coverage,
        ),
        "format": reward_format(parses=parses, contains_forbidden=contains_forbidden),
        "parsimony": reward_parsimony(test_code),
    }
    gate = reward_no_regression(new_tests_pass_clean=new_tests_pass_clean)
    weighted = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    final = max(0.0, min(1.0, gate * weighted))
    return RewardBreakdown(final=final, components=components, no_regression_gate=gate)
