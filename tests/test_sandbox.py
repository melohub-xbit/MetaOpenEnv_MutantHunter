from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mutant_hunter.safety.sandbox import Sandbox, SandboxLimits


def test_make_workspace_copies_repo(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "hello.py").write_text("x = 1\n", encoding="utf-8")
    ws = Sandbox.make_workspace(src)
    try:
        assert (ws / "hello.py").read_text(encoding="utf-8") == "x = 1\n"
    finally:
        # Cleanup the parent tempdir
        import shutil

        shutil.rmtree(ws.parent, ignore_errors=True)


def test_run_pytest_passes_on_trivial_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def f(): return 1\n", encoding="utf-8")
    tdir = repo / "tests"
    tdir.mkdir()
    (tdir / "test_m.py").write_text(
        "from m import f\n\ndef test_f():\n    assert f() == 1\n",
        encoding="utf-8",
    )
    sb = Sandbox(timeout_s=15.0)
    res = sb.run_pytest(repo, test_path="tests", timeout_s=15.0)
    assert not res.timed_out
    assert res.returncode == 0


def test_run_pytest_fails_on_failing_test(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def f(): return 1\n", encoding="utf-8")
    tdir = repo / "tests"
    tdir.mkdir()
    (tdir / "test_m.py").write_text(
        "from m import f\n\ndef test_f():\n    assert f() == 999\n",
        encoding="utf-8",
    )
    sb = Sandbox(timeout_s=15.0)
    res = sb.run_pytest(repo, test_path="tests", timeout_s=15.0)
    assert res.returncode != 0


def test_timeout_is_enforced(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("import time\n", encoding="utf-8")
    tdir = repo / "tests"
    tdir.mkdir()
    (tdir / "test_m.py").write_text(
        "import time\n\ndef test_slow():\n    time.sleep(60)\n    assert True\n",
        encoding="utf-8",
    )
    sb = Sandbox(timeout_s=3.0)
    res = sb.run_pytest(repo, test_path="tests", timeout_s=3.0)
    assert res.timed_out


def test_sandbox_limits_default_values() -> None:
    lim = SandboxLimits()
    assert lim.cpu_seconds > 0
    assert lim.address_space_bytes > 0
    assert lim.network_disabled is True
