"""Pydantic v2 contract for the MutantHunter OpenEnv environment.

Wire format between the env server (FastAPI + WebSocket) and any client
(trainer, eval harness, ad-hoc curl). All custom types subclass the
corresponding OpenEnv base (`openenv.core.env_server.interfaces.{Action,
Observation, State}`) so the server's serialization, schema endpoints, and
session machinery work without further wiring.

Design rules
------------
* `Action` carries a discriminator (`kind`) and a model-level validator that
  enforces "exactly the right payload for the chosen kind". This is what
  prevents, e.g., a `submit_tests` action that quietly omits `test_code`.
* `Observation` is everything the agent is allowed to see. `State` is the
  hidden full picture and must NEVER cross the wire.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation
from openenv.core.env_server.types import State as OpenEnvState
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- Type aliases -----------------------------------------------------------

ToolName = Literal[
    "read_file",
    "list_tests",
    "run_tests",
    "get_coverage",
    "get_mutation_report",
]

ActionKind = Literal["tool_call", "submit_tests"]


# --- Action / tool call -----------------------------------------------------


class ToolCall(BaseModel):
    """A single tool invocation issued by the agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ToolName
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("args")
    @classmethod
    def _args_keys_are_strings(cls, v: dict[str, Any]) -> dict[str, Any]:
        for k in v:
            if not isinstance(k, str):
                raise ValueError(f"tool call arg keys must be strings, got {type(k).__name__}")
        return v


class Action(OpenEnvAction):
    """The agent's per-turn action.

    Exactly one of two shapes is valid:

    * `kind == "tool_call"`: `tool_call` set, `test_code` unset.
    * `kind == "submit_tests"`: `test_code` set (non-empty), `tool_call` unset.
    """

    kind: ActionKind
    tool_call: ToolCall | None = None
    test_code: str | None = Field(
        default=None,
        description="Final pytest file content. Required when kind == 'submit_tests'.",
    )

    @model_validator(mode="after")
    def _validate_payload_matches_kind(self) -> "Action":
        if self.kind == "tool_call":
            if self.tool_call is None:
                raise ValueError("kind == 'tool_call' requires `tool_call` to be set")
            if self.test_code is not None:
                raise ValueError("kind == 'tool_call' must not set `test_code`")
        elif self.kind == "submit_tests":
            if self.tool_call is not None:
                raise ValueError("kind == 'submit_tests' must not set `tool_call`")
            if self.test_code is None or self.test_code.strip() == "":
                raise ValueError("kind == 'submit_tests' requires non-empty `test_code`")
        return self


# --- Tool result ------------------------------------------------------------


class ToolResult(BaseModel):
    """One tool invocation result, appended to `Observation.history`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: ToolName
    output: str
    truncated: bool = False


# --- Observation ------------------------------------------------------------


class Observation(OpenEnvObservation):
    """The agent-visible view of the episode at the current turn.

    Anything the agent might use to reward-hack must NOT live here — most
    notably the surviving-mutant patch text. `get_mutation_report` returns a
    line-anchored, deduplicated *summary* of mutants, not the patches
    themselves.
    """

    repo_name: str = Field(default="", description="Corpus repo identifier.")
    module_path: str = Field(default="", description="Dotted module path under test.")
    module_summary: str = Field(default="", description="AST-derived module summary.")
    existing_tests: list[str] = Field(default_factory=list)
    baseline_mutation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_remaining: int = Field(default=0, ge=0)
    history: list[ToolResult] = Field(default_factory=list)
    turn: int = Field(default=0, ge=0)


# --- Hidden state (server-side only) ----------------------------------------


class Mutant(BaseModel):
    """A single mutant produced by the mutation engine.

    Lives only inside `State.surviving_mutants`. The agent NEVER sees the
    `mutated_source` field — `get_mutation_report` returns only the
    line-anchored summary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mutant_id: str = Field(..., min_length=1)
    operator: str
    line: int = Field(..., ge=1)
    column: int = Field(..., ge=0)
    original: str
    mutated: str
    mutated_source: str | None = Field(
        default=None,
        description="Full mutated source. Hidden from the agent.",
    )


class State(OpenEnvState):
    """Full hidden episode state. Held only inside the env server process."""

    SCHEMA_VERSION: ClassVar[int] = 1

    repo_path: str = Field(default="")
    module_path: str = Field(default="")
    full_source: str = Field(default="")
    full_test_suite: str = Field(default="")
    surviving_mutants: list[Mutant] = Field(default_factory=list)
    total_mutants: int = Field(default=0, ge=0)
    coverage_baseline: float = Field(default=0.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _surviving_le_total(self) -> "State":
        if len(self.surviving_mutants) > self.total_mutants:
            raise ValueError(
                f"surviving_mutants ({len(self.surviving_mutants)}) cannot exceed "
                f"total_mutants ({self.total_mutants})"
            )
        return self


__all__ = [
    "Action",
    "ActionKind",
    "Mutant",
    "Observation",
    "State",
    "ToolCall",
    "ToolName",
    "ToolResult",
]
