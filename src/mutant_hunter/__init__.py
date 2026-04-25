"""MutantHunter: an OpenEnv-compliant RL environment for training LLMs to
write high-quality unit tests, with mutation score as the verifiable reward.

Public surface:

    from mutant_hunter.models import (
        Action,
        Observation,
        State,
        StepResult,
        ToolCall,
        ToolResult,
    )

The environment, server, client, tools, mutation engine, rubric, and safety
sandbox are implemented in their respective subpackages and become importable
as those subpackages land during Phase 1.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
