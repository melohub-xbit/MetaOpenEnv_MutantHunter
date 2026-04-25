"""Mutation engine package.

Will host:

    engine.py    — wrapper over mutmut (and an optional in-tree AST injector)
    operators.py — operator definitions (AOR, COR, ROR, LCR, NCR, BCR, BOUNDARY)
    injector.py  — deterministic mutation application
    runner.py    — sandboxed test execution per surviving mutant

Public types (`Mutant`, `MutationReport`, `MutationEngine`) are exported here
once the modules land in their dedicated Phase 1 step.
"""

from __future__ import annotations

from .engine import MutationEngine, MutationReport

__all__ = ["MutationEngine", "MutationReport"]
