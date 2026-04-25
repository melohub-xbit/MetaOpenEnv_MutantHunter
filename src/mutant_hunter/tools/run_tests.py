"""`run_tests` tool: run the existing pytest suite on the unmodified module."""

from __future__ import annotations

from pathlib import Path

from mutant_hunter.models import State
from mutant_hunter.safety.sandbox import Sandbox

_DEFAULT_TIMEOUT = 30.0
_MAX_LINES = 30


def run_tests(state: State, *, timeout_s: float = _DEFAULT_TIMEOUT) -> str:
    sb = Sandbox(timeout_s=timeout_s)
    res = sb.run_pytest(Path(state.repo_path), test_path="tests", timeout_s=timeout_s)
    if res.timed_out:
        return f"Timed out running pytest after {timeout_s:.1f}s."
    if res.returncode == 0:
        return "PASS: all existing tests passed."
    blob = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    snippet = "\n".join(blob.splitlines()[:_MAX_LINES])
    return f"FAIL: pytest exit={res.returncode}\n{snippet}"
