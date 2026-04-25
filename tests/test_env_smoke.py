from __future__ import annotations

import pytest

from mutant_hunter.models import Action, ToolCall
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment


@pytest.fixture()
def env() -> MutantHunterEnvironment:
    e = MutantHunterEnvironment()
    yield e
    e.close()


def test_reset_returns_observation_with_baseline_loaded(env: MutantHunterEnvironment) -> None:
    obs = env.reset(seed=0)
    assert obs.done is False
    assert obs.budget_remaining == 5
    assert obs.module_path
    assert obs.repo_name
    # Pulled from precomputed baseline file:
    assert 0.0 <= obs.baseline_mutation_score <= 1.0


def test_tool_call_decrements_budget(env: MutantHunterEnvironment) -> None:
    env.reset(seed=0)
    obs = env.step(
        Action(kind="tool_call", tool_call=ToolCall(name="get_mutation_report", args={}))
    )
    assert obs.done is False
    assert obs.budget_remaining == 4
    assert len(obs.history) == 1


def test_unknown_tool_records_error_message(env: MutantHunterEnvironment) -> None:
    env.reset(seed=0)
    # ToolCall enforces the literal whitelist, so we can't construct an
    # Action with an unknown tool. Instead exercise a known tool with bad args.
    obs = env.step(
        Action(
            kind="tool_call",
            tool_call=ToolCall(name="read_file", args={"path": "../../etc/passwd"}),
        )
    )
    assert obs.history[-1].output.startswith("Tool error")


def test_submit_terminates_episode(env: MutantHunterEnvironment) -> None:
    env.reset(seed=0)
    obs = env.step(
        Action(kind="submit_tests", test_code="def test_a():\n    assert True\n")
    )
    assert obs.done is True
    assert obs.reward is not None
    assert 0.0 <= obs.reward <= 1.0


def test_step_after_done_raises(env: MutantHunterEnvironment) -> None:
    env.reset(seed=0)
    env.step(Action(kind="submit_tests", test_code="def test_a():\n    assert True\n"))
    with pytest.raises(RuntimeError):
        env.step(Action(kind="tool_call", tool_call=ToolCall(name="run_tests", args={})))


def test_budget_exhaustion_terminates(env: MutantHunterEnvironment) -> None:
    env.reset(seed=0)
    for _ in range(5):
        obs = env.step(
            Action(kind="tool_call", tool_call=ToolCall(name="get_mutation_report", args={}))
        )
        assert obs.done is False
    # Sixth tool call: budget=0 → terminal with reward=0.
    obs = env.step(
        Action(kind="tool_call", tool_call=ToolCall(name="get_mutation_report", args={}))
    )
    assert obs.done is True
    assert obs.reward == 0.0
