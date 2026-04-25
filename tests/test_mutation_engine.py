from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mutant_hunter.mutation.engine import MutationEngine


def test_generate_mutants_produces_distinct_sources() -> None:
    src = "def f(x):\n    return x + 1 if x == 0 else 0\n"
    eng = MutationEngine()
    muts = eng.generate_mutants(src)
    assert len(muts) >= 4
    # Every mutant's source must differ from the original.
    for m in muts:
        assert m.mutated_source is not None
        assert m.mutated_source != src
    # mutant ids are unique.
    assert len({m.mutant_id for m in muts}) == len(muts)


def test_generate_mutants_covers_all_operators() -> None:
    src = textwrap.dedent(
        """
        def g(a, b):
            x = 1 + 2
            if a == b and a > 0:
                return True
            return False
        """
    )
    eng = MutationEngine()
    ops = {m.operator for m in eng.generate_mutants(src)}
    assert {"NCR", "ROR", "LCR", "AOR", "BCR"}.issubset(ops)


def test_generate_mutants_deterministic() -> None:
    src = "def f(): return 1 + 2\n"
    e1 = MutationEngine().generate_mutants(src)
    e2 = MutationEngine().generate_mutants(src)
    assert [m.mutant_id for m in e1] == [m.mutant_id for m in e2]
    assert [m.mutated_source for m in e1] == [m.mutated_source for m in e2]


def test_baseline_report_kills_what_strong_tests_should(tmp_path: Path) -> None:
    """A strict suite must kill basic NCR mutations on a constant-returning fn."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    tdir = repo / "tests"
    tdir.mkdir()
    (tdir / "test_m.py").write_text(
        "from m import f\n\ndef test_f_returns_one():\n    assert f() == 1\n",
        encoding="utf-8",
    )
    eng = MutationEngine()
    rpt = eng.baseline_report(
        repo_dir=repo,
        module_relpath=Path("m.py"),
        test_dir="tests",
        timeout_s=10.0,
        progress_every=0,
    )
    assert rpt.total_mutants >= 1
    # Every NCR mutant on `1` (1->2 and 1->0) must be killed.
    assert rpt.killed >= 2
