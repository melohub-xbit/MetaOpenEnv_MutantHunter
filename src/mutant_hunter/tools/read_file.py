"""`read_file` tool: read a slice of a source file inside the sandboxed repo."""

from __future__ import annotations

from pathlib import Path

from mutant_hunter.models import State

_DEFAULT_CAP = 4000


def read_file(
    state: State,
    *,
    path: str,
    start_line: int = 0,
    end_line: int = -1,
    cap_chars: int = _DEFAULT_CAP,
) -> str:
    """Return ``[start_line, end_line)`` of ``path``, capped at ``cap_chars``.

    The caller's ``path`` is treated as relative to the sandbox repo root.
    Absolute paths and parent-traversal escapes raise ``ValueError``.
    """
    repo_root = Path(state.repo_path).resolve()
    requested = (repo_root / path).resolve()
    try:
        requested.relative_to(repo_root)
    except ValueError as e:
        raise ValueError(f"path '{path}' escapes repo root") from e
    if not requested.exists() or not requested.is_file():
        raise FileNotFoundError(f"file not found: {path}")

    text = requested.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if end_line == -1:
        end_line = len(lines)
    start_line = max(0, int(start_line))
    end_line = max(start_line, min(len(lines), int(end_line)))
    sliced = "\n".join(lines[start_line:end_line])
    if len(sliced) > cap_chars:
        return sliced[:cap_chars] + "\n...<truncated>..."
    return sliced
