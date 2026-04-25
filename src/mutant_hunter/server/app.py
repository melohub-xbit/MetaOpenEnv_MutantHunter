"""FastAPI entry point for the MutantHunter env server.

Mirrors the OpenEnv `coding_env` / `echo_env` shape:

    app = create_app(EnvCls, ActionCls, ObservationCls, env_name=...)

`uvicorn mutant_hunter.server.app:app --host 0.0.0.0 --port 8000` exposes
the full HTTP + WebSocket surface that `MutantHunterEnv` consumes.
"""

from __future__ import annotations

import os

from openenv.core.env_server import create_app

from mutant_hunter.models import Action, Observation
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment

max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))

app = create_app(
    MutantHunterEnvironment,
    Action,
    Observation,
    env_name="mutant_hunter",
    max_concurrent_envs=max_concurrent,
)


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
