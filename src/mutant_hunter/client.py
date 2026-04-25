"""MutantHunterEnv — WebSocket client wrapper for the MutantHunter env server.

Subclasses `openenv.core.env_client.EnvClient` so the trainer / eval harness
can drive the env over a persistent WebSocket session, mirroring the
`coding_env` / `echo_env` patterns in OpenEnv.
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from mutant_hunter.models import Action, Observation, State


class MutantHunterEnv(EnvClient[Action, Observation, State]):
    """WebSocket client for `MutantHunterEnvironment`."""

    def _step_payload(self, action: Action) -> dict:
        # Wire format: the env server's deserialize_action runs
        # Action.model_validate(action_data), so we send the full action dict.
        return action.model_dump(exclude_none=True, exclude={"metadata"})

    def _parse_result(self, payload: dict) -> StepResult[Observation]:
        obs = Observation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(**payload)


__all__ = ["MutantHunterEnv"]
