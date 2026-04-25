"""Baseline cache: per-module precomputed mutation/coverage signal.

Each `<repo>__<module>.json` file under `corpus/_baselines/` holds:

    {
      "repo": "...",
      "module": "...",
      "module_relpath": "...",
      "total_mutants": int,
      "killed": int,
      "surviving_mutants": [Mutant.model_dump(), ...],   # includes mutated_source
      "baseline_mutation_score": float,                   # killed / total
      "coverage_baseline": float,                         # 0..100
      "module_summary": str,
      "existing_test_names": [str, ...]
    }

The env's `reset()` looks up this file by (repo, module). If missing, it
errors out — episodes must NOT silently degrade to "no surviving mutants",
since that would let the agent score 0.5 on an empty problem.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from mutant_hunter.models import Mutant

CORPUS_ROOT = Path(__file__).resolve().parent
LOCAL_LIBS_ROOT = CORPUS_ROOT / "_local"
CACHE_ROOT = CORPUS_ROOT / "_cache"
BASELINE_ROOT = CORPUS_ROOT / "_baselines"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"


@dataclass(frozen=True)
class Baseline:
    repo: str
    module: str
    module_relpath: str
    total_mutants: int
    surviving_mutants: list[Mutant]
    baseline_mutation_score: float
    coverage_baseline: float
    module_summary: str
    existing_test_names: list[str]


def baseline_path(repo: str, module: str) -> Path:
    return BASELINE_ROOT / f"{repo}__{module}.json"


def load_baseline(repo: str, module: str) -> Baseline:
    path = baseline_path(repo, module)
    if not path.exists():
        raise FileNotFoundError(
            f"baseline cache missing for {repo}:{module} ({path}). "
            f"Run scripts/precompute_baselines.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return Baseline(
        repo=data["repo"],
        module=data["module"],
        module_relpath=data["module_relpath"],
        total_mutants=int(data["total_mutants"]),
        surviving_mutants=[Mutant.model_validate(m) for m in data["surviving_mutants"]],
        baseline_mutation_score=float(data["baseline_mutation_score"]),
        coverage_baseline=float(data["coverage_baseline"]),
        module_summary=str(data.get("module_summary", "")),
        existing_test_names=list(data.get("existing_test_names", [])),
    )


def save_baseline(b: Baseline) -> Path:
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    path = baseline_path(b.repo, b.module)
    payload = {
        "repo": b.repo,
        "module": b.module,
        "module_relpath": b.module_relpath,
        "total_mutants": b.total_mutants,
        "killed": b.total_mutants - len(b.surviving_mutants),
        "baseline_mutation_score": round(b.baseline_mutation_score, 6),
        "coverage_baseline": round(b.coverage_baseline, 4),
        "surviving_mutants": [m.model_dump() for m in b.surviving_mutants],
        "module_summary": b.module_summary,
        "existing_test_names": b.existing_test_names,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def repo_dir(repo: str, source: str = "local") -> Path:
    if source == "local":
        return LOCAL_LIBS_ROOT / repo
    return CACHE_ROOT / repo


def module_to_relpath(repo: str, dotted: str) -> Path:
    parts = dotted.split(".")
    if not parts or parts[0] != repo:
        raise ValueError(f"module '{dotted}' must start with repo name '{repo}'")
    rest = parts[1:]
    if not rest:
        return Path("__init__.py")
    return Path(*rest).with_suffix(".py")


def dotted_to_workspace_relpath(dotted: str) -> Path:
    """For ``mini_calendar.parser`` -> ``Path('mini_calendar/parser.py')``.

    Unlike :func:`module_to_relpath`, the leading package segment is kept,
    because the per-mutant workspace lays the package under
    ``<workspace>/<package_name>/``.
    """
    parts = dotted.split(".")
    if not parts:
        raise ValueError(f"empty dotted module path: {dotted!r}")
    return Path(*parts).with_suffix(".py")


def summarize_module(source: str) -> str:
    """AST-derived summary: top-level signatures + first docstring line."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "<unparsable module>"
    out: list[str] = []
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        out.append(mod_doc.splitlines()[0])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"def {node.name}{_arg_signature(node)}: ...")
        elif isinstance(node, ast.ClassDef):
            out.append(f"class {node.name}: ...")
    return "\n".join(out)


def _arg_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in node.args.args]
    return "(" + ", ".join(args) + ")"


def list_existing_tests(repo_dir_path: Path, test_dir: str = "tests") -> list[str]:
    """Return `relative_path::test_name` strings for every test_* function."""
    names: list[str] = []
    test_root = repo_dir_path / test_dir
    if not test_root.exists():
        return names
    for path in sorted(test_root.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                rel = path.relative_to(repo_dir_path).as_posix()
                names.append(f"{rel}::{node.name}")
    return names


__all__ = [
    "BASELINE_ROOT",
    "Baseline",
    "CACHE_ROOT",
    "CORPUS_ROOT",
    "LOCAL_LIBS_ROOT",
    "MANIFEST_PATH",
    "baseline_path",
    "dotted_to_workspace_relpath",
    "list_existing_tests",
    "load_baseline",
    "module_to_relpath",
    "repo_dir",
    "save_baseline",
    "summarize_module",
]
