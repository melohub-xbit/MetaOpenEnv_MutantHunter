"""`get_coverage` tool: terse uncovered-line summary of the module under test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mutant_hunter.corpus.baselines import module_to_relpath
from mutant_hunter.models import State
from mutant_hunter.safety.sandbox import Sandbox

_DEFAULT_TIMEOUT = 45.0


def get_coverage(state: State, *, timeout_s: float = _DEFAULT_TIMEOUT) -> str:
    repo = Path(state.repo_path)
    repo_name = state.module_path.split(".")[0]
    relpath = module_to_relpath(repo_name, state.module_path)
    module_file = repo / relpath
    if not module_file.exists():
        return f"Module file not found: {relpath.as_posix()}"

    sb = Sandbox()
    cov_run = sb.run_coverage(repo, test_path="tests", timeout_s=timeout_s)
    if cov_run.timed_out:
        return "Timed out running coverage."

    cov_json = sb.run(
        [sys.executable, "-m", "coverage", "json", "-o", "coverage.json"],
        cwd=repo,
        timeout_s=timeout_s,
    )
    if cov_json.returncode != 0:
        return f"coverage json failed:\n{cov_json.stderr or cov_json.stdout}"

    try:
        data = json.loads((repo / "coverage.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return "Coverage report missing or unreadable."

    target_name = relpath.as_posix()
    entry = None
    for k, v in data.get("files", {}).items():
        if k.replace("\\", "/").endswith(target_name):
            entry = v
            break
    if entry is None:
        return "Coverage did not include target module."

    missing = sorted(int(x) for x in entry.get("missing_lines", []))
    if not missing:
        return "No missing lines (100% line coverage on existing suite)."
    ranges = _compress_ranges(missing)
    rendered = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)
    pct = float(entry.get("summary", {}).get("percent_covered", 0.0))
    return f"Coverage {pct:.1f}%. Missing line ranges: {rendered}"


def _compress_ranges(nums: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not nums:
        return out
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append((start, prev))
        start = prev = n
    out.append((start, prev))
    return out
