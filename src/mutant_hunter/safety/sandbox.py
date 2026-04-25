"""Subprocess sandbox for running pytest/coverage on agent-supplied test code.

Layered defenses (defense-in-depth):

1. Static checks before execution (validators.py + forbidden_patterns.py).
2. Throwaway working directory (a fresh temp copy of the target repo per
   sandbox call). The mutated module file is the only thing we ever touch.
3. Scrubbed environment: only a small allow-list of env vars survives,
   ``PYTHONNOUSERSITE=1`` is always set so editable installs in the user
   site-packages can never re-shadow a workspace's package source.
4. POSIX-only kernel-enforced limits via `resource.setrlimit` set in a
   `preexec_fn`: CPU time, address space, file size, file descriptors,
   process count.
5. POSIX-only network isolation by wrapping the child in `unshare -n`
   when the binary is available.
6. Hard wall-clock timeout via `subprocess.run(timeout=...)`.

Windows is the developer-machine fallback path: kernel limits and namespace
isolation aren't portable, so we still get (1)-(3) and (6). Production
deployment runs in the Linux Docker image where the full set fires.

Workspace layout
----------------

``Sandbox.make_workspace`` builds an *option-B* layout for per-mutant
isolation:

    <workspace>/<package_name>/...   the package source (mutated in place)
    <workspace>/<test_dir>/...       the tests, hoisted out of the package
    <workspace>/conftest.py          empty marker, forces pytest rootdir = workspace

Subprocess invocations under this sandbox prepend ``<workspace>`` to
``PYTHONPATH`` so the package source in the workspace beats any editable
install in site-packages — the exact bug that caused every mutation_kill
component to be 0 in Phase 1's baselines.

When ``package_name=None`` (used by the unit tests, where the "module"
is a flat top-level ``m.py``), the workspace contents are flat instead of
nested.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_IS_POSIX = os.name == "posix"

# Resource limits (POSIX only). Conservative defaults; tunable via constructor.
_DEFAULT_CPU_SECONDS = 30
_DEFAULT_AS_BYTES = 512 * 1024 * 1024  # 512 MB
_DEFAULT_FSIZE_BYTES = 16 * 1024 * 1024  # 16 MB written per file
_DEFAULT_NOFILE = 256
_DEFAULT_NPROC = 256

_KEEP_ENV = {
    "PATH",
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "WINDIR",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
}


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = _DEFAULT_CPU_SECONDS
    address_space_bytes: int = _DEFAULT_AS_BYTES
    file_size_bytes: int = _DEFAULT_FSIZE_BYTES
    nofile: int = _DEFAULT_NOFILE
    nproc: int = _DEFAULT_NPROC
    network_disabled: bool = True


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _make_preexec(limits: SandboxLimits):
    """Build a preexec_fn that applies POSIX rlimits in the child process.

    Returns None on non-POSIX where setrlimit isn't available.
    """
    if not _IS_POSIX:
        return None

    try:
        import resource
    except ImportError:
        return None

    def _set_limits() -> None:
        # CPU seconds
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (limits.cpu_seconds, limits.cpu_seconds),
        )
        # Virtual memory / address space — pytest+import is heavy, give a
        # generous cap rather than tight.
        resource.setrlimit(
            resource.RLIMIT_AS,
            (limits.address_space_bytes, limits.address_space_bytes),
        )
        # Max single-file write size
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (limits.file_size_bytes, limits.file_size_bytes),
        )
        # Open file descriptor cap
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (limits.nofile, limits.nofile),
        )
        # Subprocess / fork count cap (Linux only — RLIMIT_NPROC may not
        # exist on macOS for the per-process semantics, so guard it).
        if hasattr(resource, "RLIMIT_NPROC"):
            try:
                resource.setrlimit(
                    resource.RLIMIT_NPROC,
                    (limits.nproc, limits.nproc),
                )
            except (ValueError, OSError):
                # macOS sometimes refuses to lower NPROC; non-fatal.
                pass
        # Detach from controlling terminal so the child can't read tty.
        try:
            os.setsid()
        except OSError:
            pass

    return _set_limits


def _scrubbed_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in _KEEP_ENV or k.startswith("MUTANT_HUNTER_"):
            env[k] = v
    # Don't write .pyc into the read-only sandbox dir; harmless on dev too.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Disable pip / network attempts via env, even if the kernel namespace
    # isolation isn't available.
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    env["HTTP_PROXY"] = "http://127.0.0.1:1"
    env["HTTPS_PROXY"] = "http://127.0.0.1:1"
    # ALWAYS block user site-packages. With it enabled, an editable install
    # of a corpus package in ~/.local can re-shadow a workspace's mutated
    # source — exactly the bug that left every Phase 1 baseline at 0.0.
    env["PYTHONNOUSERSITE"] = "1"
    return env


class MutationWorkspaceError(RuntimeError):
    """Raised when the per-mutant workspace cannot resolve its target import.

    This is what catches the "editable install shadows the workspace" class
    of bug — if the probe at the top of an evaluation run cannot prove that
    ``import <module>`` resolves to a file inside the workspace, the env
    raises this rather than silently producing zero-kill rewards.
    """


def _wrap_with_unshare(args: list[str]) -> list[str]:
    """Prepend `unshare -n` when available to drop the network namespace."""
    if not _IS_POSIX:
        return args
    unshare = shutil.which("unshare")
    if unshare is None:
        return args
    # `--net` works without root in user namespaces; if it fails we still
    # fall back to the scrubbed-proxy env above.
    return [unshare, "--net", "--map-root-user", "--", *args]


class Sandbox:
    """Subprocess sandbox.

    Use `make_workspace()` to copy the target repo into a fresh tempdir,
    then `run_pytest()` / `run()` to execute commands inside it under the
    layered restrictions described in the module docstring.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        limits: Optional[SandboxLimits] = None,
    ):
        self._timeout_s = timeout_s
        self._limits = limits or SandboxLimits()

    @property
    def limits(self) -> SandboxLimits:
        return self._limits

    @staticmethod
    def make_workspace(
        source_repo_dir: Path,
        *,
        package_name: str | None = None,
        test_dir: str = "tests",
    ) -> Path:
        """Build an isolated per-episode workspace under a fresh temp dir.

        ``package_name`` controls the layout:

        * ``package_name=None`` (the unit-test fallback) — the source repo's
          contents are copied flat into ``<workspace>/`` so ``import <module>``
          resolves via cwd. This matches the synthetic ``m.py`` fixtures used
          by ``tests/test_mutation_engine.py``.
        * ``package_name=<name>`` — the source repo is treated as the package
          named ``<name>``: its contents land at ``<workspace>/<name>/``, and
          the inner ``<test_dir>/`` is hoisted to ``<workspace>/<test_dir>/``
          so pytest collection happens at the workspace root. This is the
          layout the corpus modules need.

        In both layouts the function:
          * strips every ``__pycache__`` from the workspace (stale ``.pyc``
            could otherwise serve up the unmutated bytes),
          * writes an empty ``conftest.py`` at the workspace root so
            ``pytest`` settles on the workspace as ``rootdir`` and so that
            ``cwd`` is always on ``sys.path`` ahead of site-packages.

        Returns the workspace root (the directory pytest should run inside),
        not the package directory. Callers also pass this path to
        ``run_pytest(pythonpath_prepend=workspace)`` so the workspace's
        package source wins over any editable install.
        """
        td = tempfile.mkdtemp(prefix="mutant_hunter_episode_")
        workspace = Path(td) / "workspace"
        workspace.mkdir(parents=True)

        if package_name is None:
            for child in source_repo_dir.iterdir():
                dst = workspace / child.name
                if child.is_dir():
                    shutil.copytree(child, dst, dirs_exist_ok=False)
                else:
                    shutil.copy2(child, dst)
        else:
            pkg_dst = workspace / package_name
            shutil.copytree(source_repo_dir, pkg_dst, dirs_exist_ok=False)
            inside_tests = pkg_dst / test_dir
            outside_tests = workspace / test_dir
            if inside_tests.exists():
                shutil.move(str(inside_tests), str(outside_tests))

        for p in list(workspace.rglob("__pycache__")):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

        (workspace / "conftest.py").write_text("", encoding="utf-8")
        return workspace

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path,
        timeout_s: float | None = None,
        pythonpath_prepend: Path | None = None,
    ) -> SandboxResult:
        env = _scrubbed_env()
        if pythonpath_prepend is not None:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                str(pythonpath_prepend) + (os.pathsep + existing if existing else "")
            )
        preexec = _make_preexec(self._limits)
        cmd = _wrap_with_unshare(argv) if self._limits.network_disabled else argv
        # On Windows, preexec_fn isn't available; subprocess will reject it.
        kwargs: dict[str, object] = {
            "cwd": str(cwd),
            "text": True,
            "capture_output": True,
            "env": env,
            "timeout": timeout_s if timeout_s is not None else self._timeout_s,
        }
        if preexec is not None:
            kwargs["preexec_fn"] = preexec
        try:
            cp = subprocess.run(cmd, **kwargs)
            return SandboxResult(cp.returncode, cp.stdout, cp.stderr, timed_out=False)
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                returncode=124,
                stdout=e.stdout if isinstance(e.stdout, str) else (e.stdout.decode("utf-8", "replace") if e.stdout else ""),
                stderr=e.stderr if isinstance(e.stderr, str) else (e.stderr.decode("utf-8", "replace") if e.stderr else ""),
                timed_out=True,
            )

    def run_pytest(
        self,
        repo_dir: Path,
        *,
        test_path: str = "tests",
        timeout_s: float | None = None,
        extra_args: list[str] | None = None,
        pythonpath_prepend: Path | None = None,
    ) -> SandboxResult:
        argv = [sys.executable, "-m", "pytest", "-q", "--no-header", test_path]
        if extra_args:
            argv.extend(extra_args)
        return self.run(
            argv,
            cwd=repo_dir,
            timeout_s=timeout_s,
            pythonpath_prepend=pythonpath_prepend,
        )

    def run_coverage(
        self,
        repo_dir: Path,
        *,
        test_path: str = "tests",
        timeout_s: float | None = None,
        pythonpath_prepend: Path | None = None,
    ) -> SandboxResult:
        argv = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "-m",
            "pytest",
            "-q",
            "--no-header",
            test_path,
        ]
        return self.run(
            argv,
            cwd=repo_dir,
            timeout_s=timeout_s,
            pythonpath_prepend=pythonpath_prepend,
        )


def is_posix() -> bool:
    return _IS_POSIX


def platform_summary() -> str:
    return f"{platform.system()} {platform.release()} ({platform.python_implementation()} {sys.version.split()[0]})"
