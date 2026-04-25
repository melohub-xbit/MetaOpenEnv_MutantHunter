"""Tools the agent can invoke mid-episode.

Each tool has signature ``(state: State, **args: Any) -> str``. The
environment dispatches `Action.tool_call.name` against `TOOL_REGISTRY`,
then wraps the return string in a `ToolResult` and appends it to the
episode history.
"""

from __future__ import annotations

from typing import Callable

from mutant_hunter.models import State

from .get_coverage import get_coverage
from .get_mutation_report import get_mutation_report
from .list_tests import list_tests
from .read_file import read_file
from .run_tests import run_tests

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "read_file": read_file,
    "list_tests": list_tests,
    "run_tests": run_tests,
    "get_coverage": get_coverage,
    "get_mutation_report": get_mutation_report,
}

__all__ = [
    "TOOL_REGISTRY",
    "get_coverage",
    "get_mutation_report",
    "list_tests",
    "read_file",
    "run_tests",
]
