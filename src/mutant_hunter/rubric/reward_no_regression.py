from __future__ import annotations


def reward_no_regression(*, new_tests_pass_clean: bool) -> float:
    return 1.0 if new_tests_pass_clean else 0.0

