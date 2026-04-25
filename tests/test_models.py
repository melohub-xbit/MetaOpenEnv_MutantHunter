from __future__ import annotations

import pytest
from pydantic import ValidationError

from mutant_hunter.models import Action, Mutant, Observation, State, ToolCall, ToolResult


def test_action_tool_call_valid() -> None:
    a = Action(kind="tool_call", tool_call=ToolCall(name="run_tests", args={}))
    assert a.kind == "tool_call"


def test_action_tool_call_with_test_code_rejected() -> None:
    with pytest.raises(ValidationError):
        Action(
            kind="tool_call",
            tool_call=ToolCall(name="run_tests", args={}),
            test_code="def test_x(): pass\n",
        )


def test_action_submit_requires_test_code() -> None:
    with pytest.raises(ValidationError):
        Action(kind="submit_tests")


def test_action_submit_empty_test_code_rejected() -> None:
    with pytest.raises(ValidationError):
        Action(kind="submit_tests", test_code="   \n")


def test_action_submit_with_tool_call_rejected() -> None:
    with pytest.raises(ValidationError):
        Action(
            kind="submit_tests",
            tool_call=ToolCall(name="run_tests", args={}),
            test_code="def test_x(): pass\n",
        )


def test_observation_defaults_round_trip() -> None:
    obs = Observation(reward=0.0)
    dumped = obs.model_dump()
    restored = Observation(**dumped)
    assert restored.budget_remaining == 0
    assert restored.history == []


def test_state_surviving_le_total_invariant() -> None:
    m = Mutant(
        mutant_id="ROR-1-0-0",
        operator="ROR",
        line=1,
        column=0,
        original="Eq",
        mutated="NotEq",
    )
    with pytest.raises(ValidationError):
        State(
            repo_path="x",
            module_path="m",
            full_source="",
            full_test_suite="",
            surviving_mutants=[m, m, m],
            total_mutants=2,
            coverage_baseline=0.0,
        )


def test_tool_result_truncated_default_false() -> None:
    r = ToolResult(tool="read_file", output="hi")
    assert r.truncated is False
