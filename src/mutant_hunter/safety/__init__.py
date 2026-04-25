"""Sandboxing + anti-hack package.

Modules
-------

* `sandbox.py` — subprocess execution with CPU/AS/file/proc rlimits and
  optional `unshare -n` network namespace dropout.
* `forbidden_patterns.py` — AST + regex blocks on agent-submitted test code.
* `validators.py` — pre-flight structural checks before any execution.
"""

from __future__ import annotations

from .forbidden_patterns import ForbiddenFinding, scan_forbidden_patterns
from .sandbox import (
    MutationWorkspaceError,
    Sandbox,
    SandboxLimits,
    SandboxResult,
    is_posix,
    platform_summary,
)
from .validators import ValidationResult, validate_test_code

__all__ = [
    "ForbiddenFinding",
    "MutationWorkspaceError",
    "Sandbox",
    "SandboxLimits",
    "SandboxResult",
    "ValidationResult",
    "is_posix",
    "platform_summary",
    "scan_forbidden_patterns",
    "validate_test_code",
]
