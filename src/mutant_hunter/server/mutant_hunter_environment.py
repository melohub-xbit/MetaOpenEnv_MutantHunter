"""MutantHunter OpenEnv environment.

Per-episode flow
----------------

1. `reset()` picks a (repo, module) from the manifest, loads the
   precomputed baseline (surviving mutants, coverage, summary), copies the
   target repo into a sandbox workspace, and returns the initial
   `Observation`.
2. `step(Action.tool_call)` dispatches to one of 5 tools, decrements the
   tool-call budget, appends a `ToolResult` to history. `done=False`,
   `reward=0.0`.
3. `step(Action.submit_tests)` validates the test code, writes it into the
   sandbox workspace as `tests/test_agent_submission.py`, runs:

   * pytest on the unmodified module (no-regression gate)
   * coverage on the same suite (delta vs baseline)
   * pytest on each baseline-surviving mutant in turn (kill count)

   then composes the final reward via `rubric.compose_reward`.
   `done=True`.

Tool-call budget exhaustion *without* a submit returns a terminal
observation with reward 0.0.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from mutant_hunter.corpus import (
    LOCAL_LIBS_ROOT,
    MANIFEST_PATH,
    Baseline,
    dotted_to_workspace_relpath,
    load_baseline,
    module_to_relpath,
    repo_dir as corpus_repo_dir,
)
from mutant_hunter.models import Action, Mutant, Observation, State, ToolResult
from mutant_hunter.rubric.compose import compose_reward
from mutant_hunter.safety.sandbox import MutationWorkspaceError, Sandbox
from mutant_hunter.safety.validators import validate_test_code
from mutant_hunter.tools import TOOL_REGISTRY

DEFAULT_TOOL_BUDGET = 5
DEFAULT_MAX_TURNS = 8
DEFAULT_NO_REGRESSION_TIMEOUT = 30.0
DEFAULT_COVERAGE_TIMEOUT = 60.0
DEFAULT_PER_MUTANT_TIMEOUT = 15.0


class MutantHunterEnvironment(Environment):
    """Phase 1 OpenEnv environment for mutation-test RL."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *,
        manifest_path: Path | None = None,
        local_libs_root: Path | None = None,
        tool_budget: int = DEFAULT_TOOL_BUDGET,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        super().__init__()
        self._manifest_path = manifest_path or MANIFEST_PATH
        self._local_libs_root = local_libs_root or LOCAL_LIBS_ROOT
        self._tool_budget = tool_budget
        self._max_turns = max_turns
        self._sandbox = Sandbox()

        self._workspace: Path | None = None
        self._baseline: Baseline | None = None
        self._test_dir: str = "tests"
        self._budget_remaining = tool_budget
        self._history: list[ToolResult] = []
        self._turn = 0
        self._done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # --- Helpers -----------------------------------------------------------

    def _load_manifest(self) -> dict:
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def _pick_module(self, seed: int | None) -> tuple[str, str, str]:
        manifest = self._load_manifest()
        local_repos = [r for r in manifest["repos"] if r.get("source") == "local"]
        if not local_repos:
            raise RuntimeError("manifest contains no local repos")
        rng = random.Random(seed)
        repo = rng.choice(local_repos)
        module = rng.choice(repo["modules"])
        return repo["name"], module["module"], module.get("test_dir", "tests")

    def _make_workspace(self, repo_name: str, test_dir: str) -> Path:
        src = self._local_libs_root / repo_name
        if not src.exists():
            raise FileNotFoundError(f"local repo not found: {src}")
        return self._sandbox.make_workspace(
            src,
            package_name=repo_name,
            test_dir=test_dir,
        )

    def _cleanup_workspace(self) -> None:
        if self._workspace is not None and self._workspace.exists():
            try:
                shutil.rmtree(self._workspace.parent, ignore_errors=True)
            finally:
                self._workspace = None

    def _make_observation(
        self,
        *,
        done: bool,
        reward: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        b = self._baseline
        return Observation(
            done=done,
            reward=reward,
            metadata=metadata or {},
            repo_name=b.repo if b else "",
            module_path=b.module if b else "",
            module_summary=b.module_summary if b else "",
            existing_tests=list(b.existing_test_names) if b else [],
            baseline_mutation_score=float(b.baseline_mutation_score) if b else 0.0,
            budget_remaining=self._budget_remaining,
            history=list(self._history),
            turn=self._turn,
        )

    # --- Required Environment API ------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        self._cleanup_workspace()
        repo_name, dotted, _test_dir = self._pick_module(seed)
        self._baseline = load_baseline(repo_name, dotted)
        self._workspace = self._make_workspace(repo_name, _test_dir)
        self._test_dir = _test_dir

        # In the new option-B layout the package source lives under
        # <workspace>/<repo_name>/<...>, not at the workspace root.
        ws_relpath = dotted_to_workspace_relpath(dotted)
        full_source = (self._workspace / ws_relpath).read_text(encoding="utf-8")
        suite_text = self._read_suite(self._workspace, _test_dir)

        self._budget_remaining = self._tool_budget
        self._history = []
        self._turn = 0
        self._done = False
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            repo_path=str(self._workspace),
            module_path=dotted,
            full_source=full_source,
            full_test_suite=suite_text,
            surviving_mutants=list(self._baseline.surviving_mutants),
            total_mutants=self._baseline.total_mutants,
            coverage_baseline=self._baseline.coverage_baseline,
        )

        return self._make_observation(
            done=False,
            reward=0.0,
            metadata={"status": "ready"},
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if self._baseline is None:
            raise RuntimeError("step() called before reset()")
        if self._done:
            raise RuntimeError("step() called on a terminated episode; call reset() first")

        self._state.step_count += 1
        self._turn += 1

        if action.kind == "tool_call":
            return self._handle_tool_call(action)
        return self._handle_submit(action)

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._cleanup_workspace()

    # --- Step handlers -----------------------------------------------------

    def _handle_tool_call(self, action: Action) -> Observation:
        assert action.tool_call is not None

        if self._budget_remaining <= 0 or self._turn > self._max_turns:
            self._done = True
            return self._make_observation(
                done=True,
                reward=0.0,
                metadata={"status": "budget_exhausted"},
            )

        self._budget_remaining -= 1
        tool_name = action.tool_call.name
        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            output = f"Unknown tool: {tool_name}"
            truncated = False
        else:
            try:
                output = str(tool_fn(self._state, **action.tool_call.args))
            except Exception as e:
                output = f"Tool error: {type(e).__name__}: {e}"
            truncated = output.endswith("...<truncated>...")

        self._history.append(ToolResult(tool=tool_name, output=output, truncated=truncated))
        return self._make_observation(
            done=False,
            reward=0.0,
            metadata={"status": "ok", "kind": "tool_call", "tool": tool_name},
        )

    def _handle_submit(self, action: Action) -> Observation:
        assert action.test_code is not None
        assert self._workspace is not None
        assert self._baseline is not None
        self._done = True

        validation = validate_test_code(action.test_code)
        parses = not any(f.kind == "syntax" for f in validation.findings)
        contains_forbidden = (not validation.ok) and any(
            f.kind in {"import", "call", "attr", "fs", "regex", "structure", "size"}
            for f in validation.findings
        )

        if not validation.ok:
            breakdown = compose_reward(
                baseline_surviving=len(self._baseline.surviving_mutants),
                killed_by_new_only=0,
                baseline_coverage=self._baseline.coverage_baseline,
                new_coverage=self._baseline.coverage_baseline,
                parses=parses,
                contains_forbidden=True,
                new_tests_pass_clean=False,
                test_code=action.test_code,
            )
            return self._make_observation(
                done=True,
                reward=breakdown.final,
                metadata={
                    "status": "rejected",
                    "findings": [f.message for f in validation.findings][:10],
                    "components": breakdown.components,
                    "no_regression_gate": breakdown.no_regression_gate,
                    "baseline_surviving": len(self._baseline.surviving_mutants),
                    "killed_by_new_only": 0,
                    "new_coverage": self._baseline.coverage_baseline,
                },
            )

        ws_relpath = dotted_to_workspace_relpath(self._baseline.module)
        new_tests_pass_clean, new_cov, killed_by_new_only = self._evaluate_submission(
            test_code=action.test_code,
            workspace_module_relpath=ws_relpath,
            surviving_mutants=self._baseline.surviving_mutants,
        )

        breakdown = compose_reward(
            baseline_surviving=len(self._baseline.surviving_mutants),
            killed_by_new_only=killed_by_new_only,
            baseline_coverage=self._baseline.coverage_baseline,
            new_coverage=new_cov,
            parses=True,
            contains_forbidden=False,
            new_tests_pass_clean=new_tests_pass_clean,
            test_code=action.test_code,
        )

        return self._make_observation(
            done=True,
            reward=breakdown.final,
            metadata={
                "status": "ok",
                "kind": "submit_tests",
                "components": breakdown.components,
                "no_regression_gate": breakdown.no_regression_gate,
                "killed_by_new_only": killed_by_new_only,
                "baseline_surviving": len(self._baseline.surviving_mutants),
                "new_coverage": new_cov,
                "new_tests_pass_clean": new_tests_pass_clean,
            },
        )

    # --- Submission evaluation --------------------------------------------

    def _evaluate_submission(
        self,
        *,
        test_code: str,
        workspace_module_relpath: Path,
        surviving_mutants: list[Mutant],
    ) -> tuple[bool, float, int]:
        """Return (new_tests_pass_clean, new_coverage, killed_by_new_only).

        Builds a per-submission sandbox copy of ``self._workspace``, drops
        the agent's tests into ``<tmp_workspace>/<test_dir>/test_agent_submission.py``,
        and runs the no-regression check, coverage, and per-mutant pytest
        all under the same workspace layout. Before any of those, an
        import-path probe verifies the mutated module imports out of
        ``<tmp_workspace>/<package_name>/`` rather than a site-packages
        editable install — if it doesn't, this method raises
        :class:`MutationWorkspaceError` instead of silently producing
        zero-kill rewards.
        """
        assert self._workspace is not None
        with tempfile.TemporaryDirectory(prefix="mutant_hunter_submit_") as td:
            tmp_workspace = Path(td) / "workspace"
            shutil.copytree(self._workspace, tmp_workspace, dirs_exist_ok=False)
            new_test = tmp_workspace / self._test_dir / "test_agent_submission.py"
            new_test.parent.mkdir(parents=True, exist_ok=True)
            new_test.write_text(test_code, encoding="utf-8")

            self._verify_import_resolves(tmp_workspace, workspace_module_relpath)

            new_tests_pass_clean = self._run_no_regression(tmp_workspace)
            new_coverage = self._run_coverage(tmp_workspace, workspace_module_relpath)
            killed = self._run_per_mutant(
                tmp_workspace, workspace_module_relpath, surviving_mutants
            )
        return new_tests_pass_clean, new_coverage, killed

    def _verify_import_resolves(
        self,
        workspace: Path,
        workspace_module_relpath: Path,
    ) -> None:
        """Probe: assert ``import <module>`` resolves under ``workspace``.

        Catches the editable-install-shadows-the-workspace bug at the top of
        every per-mutant evaluation rather than after wasting minutes on a
        full mutant pass that silently no-ops.
        """
        assert self._baseline is not None
        dotted = self._baseline.module
        expected_file = (workspace / workspace_module_relpath).resolve()
        # Build the probe source line-by-line so the substituted values
        # (which can contain backslashes on Windows) cannot break out of
        # any quoting.
        probe_lines = [
            "import os, sys",
            f"import {dotted} as _m",
            "actual = os.path.realpath(_m.__file__)",
            f"expected = os.path.realpath({str(expected_file)!r})",
            "if actual != expected:",
            "    sys.stderr.write("
            f"'PROBE FAIL: import {dotted} resolved to ' + repr(actual) + "
            "' not ' + repr(expected))",
            "    sys.exit(2)",
        ]
        probe = "\n".join(probe_lines) + "\n"
        probe_path = workspace / "_probe_import.py"
        probe_path.write_text(probe, encoding="utf-8")
        try:
            res = self._sandbox.run(
                [sys.executable, str(probe_path)],
                cwd=workspace,
                timeout_s=15.0,
                pythonpath_prepend=workspace,
            )
        finally:
            try:
                probe_path.unlink()
            except OSError:
                pass
        if res.returncode != 0:
            raise MutationWorkspaceError(
                f"import-path probe failed for {dotted!r}; "
                f"workspace={workspace}; "
                f"returncode={res.returncode}; "
                f"stdout={res.stdout!r}; stderr={res.stderr!r}"
            )

    def _run_no_regression(self, workspace: Path) -> bool:
        res = self._sandbox.run_pytest(
            workspace,
            test_path=self._test_dir,
            timeout_s=DEFAULT_NO_REGRESSION_TIMEOUT,
            pythonpath_prepend=workspace,
        )
        return (not res.timed_out) and res.returncode == 0

    def _run_coverage(self, workspace: Path, workspace_module_relpath: Path) -> float:
        try:
            cov_run = self._sandbox.run_coverage(
                workspace,
                test_path=self._test_dir,
                timeout_s=DEFAULT_COVERAGE_TIMEOUT,
                pythonpath_prepend=workspace,
            )
            if cov_run.timed_out:
                return self._baseline.coverage_baseline if self._baseline else 0.0
            cov_json = self._sandbox.run(
                [sys.executable, "-m", "coverage", "json", "-o", "coverage.json"],
                cwd=workspace,
                timeout_s=DEFAULT_COVERAGE_TIMEOUT,
                pythonpath_prepend=workspace,
            )
            if cov_json.returncode != 0:
                return self._baseline.coverage_baseline if self._baseline else 0.0
            data = json.loads((workspace / "coverage.json").read_text(encoding="utf-8"))
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            return self._baseline.coverage_baseline if self._baseline else 0.0

        target_name = workspace_module_relpath.as_posix()
        for k, v in data.get("files", {}).items():
            kp = k.replace("\\", "/")
            if kp.endswith(target_name):
                return float(v.get("summary", {}).get("percent_covered", 0.0))
        return self._baseline.coverage_baseline if self._baseline else 0.0

    def _run_per_mutant(
        self,
        workspace: Path,
        workspace_module_relpath: Path,
        surviving_mutants: list[Mutant],
    ) -> int:
        """For each baseline-surviving mutant, write the mutant source then run
        the agent's submission. A non-zero exit means the new tests killed it."""
        target_path = workspace / workspace_module_relpath
        original_text = target_path.read_text(encoding="utf-8")
        killed = 0
        agent_test_relpath = f"{self._test_dir}/test_agent_submission.py"
        try:
            for m in surviving_mutants:
                if m.mutated_source is None:
                    continue
                target_path.write_text(m.mutated_source, encoding="utf-8")
                res = self._sandbox.run_pytest(
                    workspace,
                    test_path=agent_test_relpath,
                    timeout_s=DEFAULT_PER_MUTANT_TIMEOUT,
                    pythonpath_prepend=workspace,
                )
                if res.timed_out or res.returncode != 0:
                    killed += 1
        finally:
            target_path.write_text(original_text, encoding="utf-8")
        return killed

    # --- Misc helpers ------------------------------------------------------

    @staticmethod
    def _read_suite(repo_dir: Path, test_dir: str) -> str:
        root = repo_dir / test_dir
        if not root.exists():
            return ""
        chunks: list[str] = []
        for p in sorted(root.rglob("test_*.py")):
            try:
                chunks.append(f"# {p.relative_to(repo_dir).as_posix()}\n" + p.read_text(encoding="utf-8"))
            except Exception:
                continue
        return "\n\n".join(chunks)
