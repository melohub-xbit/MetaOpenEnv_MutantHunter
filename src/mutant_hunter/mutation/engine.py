"""Mutation engine.

Pure-Python AST-based mutation operators (no mutmut dependency at runtime
because mutmut doesn't run cleanly on native Windows). Five operator
classes are produced: NCR (numeric constant), BCR (boolean constant), ROR
(relational op), LCR (logical connector), AOR (arithmetic op replacement).

The engine is deterministic: identical input source produces identical
mutant ordering and identical mutant ids. Each `Mutant.mutated_source` is
the full mutated module source ready to be written to disk and re-tested.

``baseline_report`` runs the existing test suite against every mutant. It
delegates to :meth:`mutant_hunter.safety.sandbox.Sandbox.make_workspace`
so the per-mutant copy uses the same option-B layout as the live env: the
package source lands at ``<workspace>/<package_name>/``, tests are hoisted
to ``<workspace>/<test_dir>/``, and ``PYTHONPATH=<workspace>`` is prepended
on every subprocess so the workspace beats any editable install.
"""

from __future__ import annotations

import ast
import copy
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mutant_hunter.models import Mutant
from mutant_hunter.safety.sandbox import Sandbox


@dataclass(frozen=True)
class MutationReport:
    total_mutants: int
    killed_mutants: list[str]
    survived_mutants: list[Mutant]

    @property
    def killed(self) -> int:
        return len(self.killed_mutants)

    @property
    def survived(self) -> int:
        return len(self.survived_mutants)

    @property
    def mutation_score(self) -> float:
        if self.total_mutants <= 0:
            return 0.0
        return self.killed / self.total_mutants


# --- Operator helpers -------------------------------------------------------

_COMPARE_FLIP: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}

_ARITH_FLIP: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.Div: ast.Mult,
    ast.Mod: ast.Mult,
}


@dataclass(frozen=True)
class _Candidate:
    operator: str
    original: str
    mutated: str
    line: int
    column: int
    apply: Callable[[ast.AST], None]
    """Function applied to the equivalent node in a fresh deep-copy of the tree."""


def _apply_constant(new_value: object) -> Callable[[ast.AST], None]:
    def _f(node: ast.AST) -> None:
        assert isinstance(node, ast.Constant)
        node.value = new_value
    return _f


def _apply_compare_op(new_op_cls: type[ast.cmpop]) -> Callable[[ast.AST], None]:
    def _f(node: ast.AST) -> None:
        assert isinstance(node, ast.Compare)
        node.ops[0] = new_op_cls()
    return _f


def _apply_boolop(new_op_cls: type[ast.boolop]) -> Callable[[ast.AST], None]:
    def _f(node: ast.AST) -> None:
        assert isinstance(node, ast.BoolOp)
        node.op = new_op_cls()
    return _f


def _apply_binop(new_op_cls: type[ast.operator]) -> Callable[[ast.AST], None]:
    def _f(node: ast.AST) -> None:
        assert isinstance(node, ast.BinOp)
        node.op = new_op_cls()
    return _f


# --- Engine -----------------------------------------------------------------


class MutationEngine:
    """Deterministic AST mutation engine."""

    def generate_mutants(self, source: str) -> list[Mutant]:
        """Return one `Mutant` per single-point mutation.

        Each mutant's `mutated_source` is a complete, syntactically-valid
        Python source string ready to drop in place of the original module.
        """
        original_tree = ast.parse(source)
        nodes_in_order = list(ast.walk(original_tree))

        candidates: list[tuple[int, _Candidate]] = []
        for idx, node in enumerate(nodes_in_order):
            for cand in self._candidates_for_node(node):
                candidates.append((idx, cand))

        mutants: list[Mutant] = []
        for mid, (idx, cand) in enumerate(candidates):
            new_tree = copy.deepcopy(original_tree)
            target = list(ast.walk(new_tree))[idx]
            cand.apply(target)
            ast.fix_missing_locations(new_tree)
            mutated_source = ast.unparse(new_tree)
            if mutated_source == source:
                # Defensive: if the mutation collapsed (e.g. no observable
                # change), skip it rather than emit a useless duplicate.
                continue
            mutants.append(
                Mutant(
                    mutant_id=f"{cand.operator}-{cand.line}-{cand.column}-{mid}",
                    operator=cand.operator,
                    line=cand.line,
                    column=cand.column,
                    original=cand.original,
                    mutated=cand.mutated,
                    mutated_source=mutated_source,
                )
            )
        return mutants

    def _candidates_for_node(self, node: ast.AST) -> list[_Candidate]:
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        out: list[_Candidate] = []

        if isinstance(node, ast.Constant):
            v = node.value
            # bool is a subclass of int, so check it first
            if isinstance(v, bool):
                out.append(
                    _Candidate(
                        operator="BCR",
                        original=str(v),
                        mutated=str(not v),
                        line=line,
                        column=col,
                        apply=_apply_constant(not v),
                    )
                )
            elif isinstance(v, int):
                out.append(
                    _Candidate(
                        operator="NCR",
                        original=repr(v),
                        mutated=repr(v + 1),
                        line=line,
                        column=col,
                        apply=_apply_constant(v + 1),
                    )
                )
                if v != 0:
                    out.append(
                        _Candidate(
                            operator="NCR",
                            original=repr(v),
                            mutated="0",
                            line=line,
                            column=col,
                            apply=_apply_constant(0),
                        )
                    )
            elif isinstance(v, float):
                out.append(
                    _Candidate(
                        operator="NCR",
                        original=repr(v),
                        mutated=repr(v + 1.0),
                        line=line,
                        column=col,
                        apply=_apply_constant(v + 1.0),
                    )
                )
                if v != 0.0:
                    out.append(
                        _Candidate(
                            operator="NCR",
                            original=repr(v),
                            mutated="0.0",
                            line=line,
                            column=col,
                            apply=_apply_constant(0.0),
                        )
                    )

        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            op_cls = type(node.ops[0])
            new_cls = _COMPARE_FLIP.get(op_cls)
            if new_cls is not None:
                out.append(
                    _Candidate(
                        operator="ROR",
                        original=op_cls.__name__,
                        mutated=new_cls.__name__,
                        line=line,
                        column=col,
                        apply=_apply_compare_op(new_cls),
                    )
                )

        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                out.append(
                    _Candidate(
                        operator="LCR",
                        original="and",
                        mutated="or",
                        line=line,
                        column=col,
                        apply=_apply_boolop(ast.Or),
                    )
                )
            elif isinstance(node.op, ast.Or):
                out.append(
                    _Candidate(
                        operator="LCR",
                        original="or",
                        mutated="and",
                        line=line,
                        column=col,
                        apply=_apply_boolop(ast.And),
                    )
                )

        elif isinstance(node, ast.BinOp):
            new_cls = _ARITH_FLIP.get(type(node.op))
            if new_cls is not None:
                out.append(
                    _Candidate(
                        operator="AOR",
                        original=type(node.op).__name__,
                        mutated=new_cls.__name__,
                        line=line,
                        column=col,
                        apply=_apply_binop(new_cls),
                    )
                )

        return out

    # --- Test runners -------------------------------------------------------

    def run_pytest(
        self,
        repo_dir: Path,
        *,
        timeout_s: float = 20.0,
        extra_pytest_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        args = [sys.executable, "-m", "pytest", "-q", "--no-header", "-x"]
        if extra_pytest_args:
            args.extend(extra_pytest_args)
        return subprocess.run(
            args,
            cwd=str(repo_dir),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )

    def baseline_report(
        self,
        *,
        repo_dir: Path,
        module_relpath: Path,
        test_dir: str = "tests",
        package_name: str | None = None,
        timeout_s: float = 20.0,
        progress_every: int = 10,
    ) -> MutationReport:
        """Run the existing suite against every mutant. Killed = exit != 0.

        ``package_name`` controls the workspace layout (see
        :meth:`Sandbox.make_workspace`). Pass the corpus repo name (which
        matches the dotted module's first segment) for real corpus modules;
        leave it ``None`` for the synthetic flat-``m.py`` fixtures used by
        the unit tests.
        """
        source_path = repo_dir / module_relpath
        source = source_path.read_text(encoding="utf-8")
        mutants = self.generate_mutants(source)

        killed: list[str] = []
        survived: list[Mutant] = []

        workspace = Sandbox.make_workspace(
            repo_dir,
            package_name=package_name,
            test_dir=test_dir,
        )
        try:
            if package_name is None:
                tmp_source = workspace / module_relpath
            else:
                tmp_source = workspace / package_name / module_relpath
            original_text = tmp_source.read_text(encoding="utf-8")

            base_env = os.environ.copy()
            base_env["PYTHONPATH"] = str(workspace) + (
                os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else ""
            )
            base_env["PYTHONNOUSERSITE"] = "1"
            base_env["PYTHONDONTWRITEBYTECODE"] = "1"

            for i, m in enumerate(mutants, start=1):
                assert m.mutated_source is not None
                tmp_source.write_text(m.mutated_source, encoding="utf-8")
                try:
                    res = self.run_pytest(
                        workspace,
                        timeout_s=timeout_s,
                        extra_pytest_args=[test_dir],
                        env=base_env,
                    )
                    is_killed = res.returncode != 0
                except subprocess.TimeoutExpired:
                    is_killed = True
                finally:
                    tmp_source.write_text(original_text, encoding="utf-8")

                if is_killed:
                    killed.append(m.mutant_id)
                else:
                    survived.append(m)

                if progress_every and i % progress_every == 0:
                    print(
                        f"  mutants: {i}/{len(mutants)} killed={len(killed)} survived={len(survived)}",
                        flush=True,
                    )
        finally:
            import shutil as _shutil

            _shutil.rmtree(workspace.parent, ignore_errors=True)

        return MutationReport(
            total_mutants=len(mutants),
            killed_mutants=killed,
            survived_mutants=survived,
        )
