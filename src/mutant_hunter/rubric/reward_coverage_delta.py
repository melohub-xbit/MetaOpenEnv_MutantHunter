from __future__ import annotations


def reward_coverage_delta(*, baseline_coverage: float, new_coverage: float) -> float:
    # Coverage in [0,100]
    delta = new_coverage - baseline_coverage
    headroom = max(100.0 - baseline_coverage, 1.0)
    v = delta / headroom
    return max(0.0, min(1.0, v))

