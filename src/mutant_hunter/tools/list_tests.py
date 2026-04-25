"""`list_tests` tool: enumerate test functions + their first docstring line."""

from __future__ import annotations

import ast
from pathlib import Path

from mutant_hunter.models import State

_MAX_ENTRIES = 100


def list_tests(state: State) -> str:
    repo = Path(state.repo_path)
    test_root = repo / "tests"
    if not test_root.exists():
        return "No tests/ directory found."

    out: list[str] = []
    for path in sorted(test_root.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rel = path.relative_to(repo).as_posix()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = ast.get_docstring(node) or ""
                first = doc.strip().splitlines()[0] if doc.strip() else ""
                out.append(f"{rel}::{node.name}" + (f" — {first}" if first else ""))
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test_"):
                        doc = ast.get_docstring(sub) or ""
                        first = doc.strip().splitlines()[0] if doc.strip() else ""
                        out.append(f"{rel}::{node.name}::{sub.name}" + (f" — {first}" if first else ""))
        if len(out) >= _MAX_ENTRIES:
            out = out[:_MAX_ENTRIES]
            out.append(f"...<truncated at {_MAX_ENTRIES}>...")
            break
    return "\n".join(out) if out else "No test_* functions found."
