"""Precompute per-module baselines for the corpus.

For every (repo, module) entry in `corpus/manifest.json` whose `source ==
"local"`, this script:

1. Generates the full mutant set for the module (deterministic, AST-based).
2. Runs the existing test suite against each mutant — killed = exit != 0.
3. Computes line+branch coverage of the module under the existing suite.
4. Writes `corpus/_baselines/<repo>__<module>.json` with the surviving
   mutants (incl. mutated_source), coverage, and a module summary.
5. Updates `manifest.json` in place with the summary stats.

Run:

    python scripts/precompute_baselines.py
    python scripts/precompute_baselines.py --only-repo mini_calendar
    python scripts/precompute_baselines.py --only-module csv_normalizer.normalizer
    python scripts/precompute_baselines.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Make `src/` importable when running directly from repo root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mutant_hunter.corpus.baselines import (  # noqa: E402
    BASELINE_ROOT,
    Baseline,
    LOCAL_LIBS_ROOT,
    MANIFEST_PATH,
    list_existing_tests,
    module_to_relpath,
    save_baseline,
    summarize_module,
)
from mutant_hunter.mutation.engine import MutationEngine  # noqa: E402


def _count_loc(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for ln in lines if ln.strip())


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _validate_manifest(manifest: dict) -> None:
    if manifest.get("version") != "v1":
        raise ValueError("manifest.json: expected version == 'v1'")
    if not isinstance(manifest.get("repos"), list):
        raise ValueError("manifest.json: missing 'repos' list")
    for repo in manifest["repos"]:
        for k in ("name", "source", "path", "license", "modules"):
            if k not in repo:
                raise ValueError(f"manifest.json: repo missing key '{k}'")


def _compute_coverage(
    repo_dir: Path,
    module_relpath: Path,
    test_dir: str,
    timeout_s: float,
    package_name: str | None = None,
) -> float:
    """Run coverage in a fresh option-B workspace. Return percent_covered or 0.0."""
    from mutant_hunter.safety.sandbox import Sandbox  # local import to avoid cycle

    workspace = Sandbox.make_workspace(
        repo_dir,
        package_name=package_name,
        test_dir=test_dir,
    )
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(workspace) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            subprocess.run(
                [sys.executable, "-m", "coverage", "run", "--branch", "-m", "pytest", "-q", test_dir],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                env=env,
            )
            rpt = subprocess.run(
                [sys.executable, "-m", "coverage", "json", "-o", "coverage.json"],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                env=env,
            )
            if rpt.returncode != 0:
                return 0.0
            data = json.loads((workspace / "coverage.json").read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            return 0.0

        files = data.get("files", {})
        if package_name is None:
            target_name = module_relpath.as_posix()
        else:
            target_name = (Path(package_name) / module_relpath).as_posix()
        for k, v in files.items():
            kp = k.replace("\\", "/")
            if kp.endswith(target_name):
                return float(v.get("summary", {}).get("percent_covered", 0.0))
        return 0.0
    finally:
        shutil.rmtree(workspace.parent, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute baseline mutation scores + coverage.")
    ap.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    ap.add_argument("--timeout-s", type=float, default=8.0, help="Per-mutant pytest timeout")
    ap.add_argument("--cov-timeout-s", type=float, default=60.0)
    ap.add_argument("--only-repo", type=str, default=None)
    ap.add_argument("--only-module", type=str, default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip modules with a baseline file already on disk.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)

    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    engine = MutationEngine()

    for repo in manifest["repos"]:
        if repo.get("source") != "local":
            continue
        repo_name = repo["name"]
        if args.only_repo and repo_name != args.only_repo:
            continue
        repo_path = LOCAL_LIBS_ROOT / repo_name
        if not repo_path.exists():
            raise FileNotFoundError(f"missing local repo dir: {repo_path}")

        for mod in repo["modules"]:
            dotted = mod["module"]
            if args.only_module and dotted != args.only_module:
                continue

            cache_path = BASELINE_ROOT / f"{repo_name}__{dotted}.json"
            if args.skip_existing and cache_path.exists():
                print(f"Skipping existing baseline for {repo_name}:{dotted}", flush=True)
                continue

            relpath = module_to_relpath(repo_name, dotted)
            source_path = repo_path / relpath
            if not source_path.exists():
                raise FileNotFoundError(f"module source not found: {dotted} -> {source_path}")

            test_dir = mod.get("test_dir", "tests")
            loc = _count_loc(source_path)
            print(f"=== {repo_name}:{dotted} (loc={loc}) ===", flush=True)

            cov = _compute_coverage(
                repo_path, relpath, test_dir, args.cov_timeout_s, package_name=repo_name
            )
            print(f"  coverage_baseline = {cov:.2f}%", flush=True)

            print("  running mutation baseline ...", flush=True)
            report = engine.baseline_report(
                repo_dir=repo_path,
                module_relpath=relpath,
                test_dir=test_dir,
                package_name=repo_name,
                timeout_s=args.timeout_s,
            )
            print(
                f"  total={report.total_mutants} killed={report.killed} "
                f"survived={report.survived} score={report.mutation_score:.3f}",
                flush=True,
            )

            module_summary = summarize_module(source_path.read_text(encoding="utf-8"))
            existing_tests = list_existing_tests(repo_path, test_dir)

            baseline = Baseline(
                repo=repo_name,
                module=dotted,
                module_relpath=str(relpath).replace("\\", "/"),
                total_mutants=report.total_mutants,
                surviving_mutants=report.survived_mutants,
                baseline_mutation_score=report.mutation_score,
                coverage_baseline=cov,
                module_summary=module_summary,
                existing_test_names=existing_tests,
            )
            written = save_baseline(baseline)
            print(f"  wrote {written}", flush=True)

            mod["loc"] = loc
            mod["total_mutants"] = report.total_mutants
            mod["surviving_mutants"] = report.survived
            mod["baseline_mutation_score"] = round(report.mutation_score, 6)
            mod["coverage_baseline"] = round(cov, 4)

    _atomic_write_json(manifest_path, manifest)
    print(f"Updated {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
