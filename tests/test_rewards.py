from __future__ import annotations

import pytest

from mutant_hunter.rubric.compose import WEIGHTS, compose_reward
from mutant_hunter.rubric.reward_coverage_delta import reward_coverage_delta
from mutant_hunter.rubric.reward_format import reward_format
from mutant_hunter.rubric.reward_mutation_kill import reward_mutation_kill
from mutant_hunter.rubric.reward_no_regression import reward_no_regression
from mutant_hunter.rubric.reward_parsimony import reward_parsimony


def test_mutation_kill_zero_baseline_neutral() -> None:
    assert reward_mutation_kill(baseline_surviving=0, killed_by_new_only=0) == 0.5


def test_mutation_kill_full() -> None:
    assert reward_mutation_kill(baseline_surviving=10, killed_by_new_only=10) == 1.0


def test_mutation_kill_partial() -> None:
    v = reward_mutation_kill(baseline_surviving=10, killed_by_new_only=4)
    assert v == pytest.approx(0.4)


def test_coverage_delta_positive() -> None:
    v = reward_coverage_delta(baseline_coverage=50.0, new_coverage=75.0)
    # delta = 25, headroom = 50 -> 0.5
    assert v == pytest.approx(0.5)


def test_coverage_delta_negative_clamped() -> None:
    v = reward_coverage_delta(baseline_coverage=80.0, new_coverage=70.0)
    assert v == 0.0


def test_format_blocks_on_forbidden() -> None:
    assert reward_format(parses=True, contains_forbidden=True) == 0.0


def test_format_blocks_on_unparseable() -> None:
    assert reward_format(parses=False, contains_forbidden=False) == 0.0


def test_format_pass() -> None:
    assert reward_format(parses=True, contains_forbidden=False) == 1.0


def test_no_regression_binary() -> None:
    assert reward_no_regression(new_tests_pass_clean=True) == 1.0
    assert reward_no_regression(new_tests_pass_clean=False) == 0.0


def test_parsimony_short_test_full_credit() -> None:
    src = "def test_a():\n    assert 1 == 1\n"
    assert reward_parsimony(src) == 1.0


def test_parsimony_no_tests_zero() -> None:
    src = "x = 1\n"
    assert reward_parsimony(src) == 0.0


def test_parsimony_long_test_decays() -> None:
    body = "    " + "x = 1\n    " * 30 + "    assert True\n"
    src = "def test_long():\n" + body
    v = reward_parsimony(src)
    assert v <= 1.0
    assert v < 1.0


def test_compose_no_regression_zeros_everything() -> None:
    rb = compose_reward(
        baseline_surviving=10,
        killed_by_new_only=10,
        baseline_coverage=50.0,
        new_coverage=100.0,
        parses=True,
        contains_forbidden=False,
        new_tests_pass_clean=False,
        test_code="def test_x():\n    assert True\n",
    )
    assert rb.no_regression_gate == 0.0
    assert rb.final == 0.0
    # Components are still computed for telemetry.
    assert rb.components["mutation_kill"] == 1.0


def test_compose_perfect_submission() -> None:
    rb = compose_reward(
        baseline_surviving=10,
        killed_by_new_only=10,
        baseline_coverage=50.0,
        new_coverage=100.0,
        parses=True,
        contains_forbidden=False,
        new_tests_pass_clean=True,
        test_code="def test_x():\n    assert 1 + 1 == 2\n",
    )
    expected = sum(WEIGHTS.values())
    assert rb.final == pytest.approx(expected)
    assert rb.final == pytest.approx(1.0)


def test_compose_format_failure_blocks_format_only() -> None:
    rb = compose_reward(
        baseline_surviving=10,
        killed_by_new_only=10,
        baseline_coverage=50.0,
        new_coverage=100.0,
        parses=True,
        contains_forbidden=True,
        new_tests_pass_clean=True,
        test_code="def test_x():\n    assert True\n",
    )
    # mutation=1 cov=1 format=0 parsimony=1
    expected = WEIGHTS["mutation_kill"] + WEIGHTS["coverage_delta"] + WEIGHTS["parsimony"]
    assert rb.final == pytest.approx(expected)


def test_weights_sum_to_one() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
