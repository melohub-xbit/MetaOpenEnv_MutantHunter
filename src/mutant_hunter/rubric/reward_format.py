from __future__ import annotations


def reward_format(*, parses: bool, contains_forbidden: bool) -> float:
    if not parses:
        return 0.0
    if contains_forbidden:
        return 0.0
    return 1.0

