# MutantHunter Safety Sandbox — Threat Model & Design

The sandbox is what stops the agent from making the reward go up by cheating
instead of writing better tests. Every defence here exists because there is a
specific reward-hack vector it is built to refuse.

## Threat model

The agent is **untrusted code**. We assume a competent adversary that will:

1. Try to read the surviving-mutant list from disk and reflect it as an
   `assert` to fake mutation kills without writing real tests.
2. Try to patch `pytest`, `mutmut`, or `coverage` internals at runtime to
   forge pass/fail counts.
3. Try to network-exfiltrate the corpus or environment secrets.
4. Try to write arbitrary files on the host (CI poisoning, code injection
   into the next episode).
5. Try to spawn subprocesses (`os.system`, `subprocess`, shelled-out git).
6. Try to burn budget — infinite loops, memory bombs — until the host kills
   the run and the env returns a zero-reward "done".
7. Try to shadow the module under test by injecting a same-named module
   with a forged truth table.
8. Try `sys.exit(0)` to short-circuit the test runner.
9. Try `pytest.skip` everywhere so that "no failures" looks like success.
10. Try to mutate `__builtins__` / `globals()` / `sys.modules` to neuter
    assertion helpers.

## Defence layers (defence in depth)

### 1. Pre-flight static validation (`validators.py`)

Before any execution:

- Parse the submitted test file with `ast.parse`. Reject syntax errors.
- Reject if the file is over 50 KB.
- Reject if it contains more than 50 `test_*` functions.
- Reject if any top-level statement is not one of: `Import`, `ImportFrom`,
  `FunctionDef` (named `test_*` or `_*` helper), `ClassDef` (named
  `Test*`), `Assign` (allow simple constants used as fixtures).

### 2. Forbidden-pattern AST + regex pass (`forbidden_patterns.py`)

Walks the AST and matches against a regex denylist. Either match → reject:

- Imports of `os`, `sys`, `subprocess`, `socket`, `urllib`, `urllib2`,
  `urllib3`, `requests`, `httpx`, `aiohttp`, `ftplib`, `telnetlib`,
  `smtplib`, `pickle`, `marshal`, `shelve`, `ctypes`, `cffi`, `mmap`,
  `multiprocessing`, `threading.Thread`, `concurrent.futures`,
  `pathlib.Path.write_*`, `tempfile.mkstemp`.
- Calls to `eval`, `exec`, `compile`, `__import__`, `globals`, `locals`,
  `vars`, `getattr` with a non-literal attribute name, `setattr` on
  imported modules, `delattr`, `breakpoint`.
- Any attribute access on `pytest._*`, `mutmut._*`, `coverage._*`,
  `_pytest.*`, `sys.modules`.
- Any `open(...)` whose mode contains `'w'`, `'a'`, `'x'`, `'+'`.
- Calls to `time.sleep`, `signal.*`, `os.kill`, `sys.exit`, `os._exit`,
  `pytest.exit`, `pytest.skip` (we treat skip as a hack vector).
- Direct mutation of `__builtins__`, `globals()`, `sys.modules`,
  `sys.path`, environment variables.

The denylist is conservative-by-design. Legitimate uses (e.g. `tempfile`)
that we want to allow can be added explicitly with a justification comment.

### 3. Subprocess sandbox (`sandbox.py`)

Even after the static checks, every test execution runs in a child
process with hard limits applied via `resource.setrlimit` and Linux
namespaces:

| Limit / isolation                | Value                                  |
| -------------------------------- | -------------------------------------- |
| `RLIMIT_CPU`                     | 30 seconds                             |
| `RLIMIT_AS` (address space)      | 512 MiB                                |
| `RLIMIT_NPROC`                   | 32                                     |
| `RLIMIT_NOFILE`                  | 64                                     |
| `RLIMIT_FSIZE`                   | 16 MiB                                 |
| `signal.alarm` wall-clock guard  | 45 seconds                             |
| Network                          | `unshare -n` (Linux) — no NICs visible |
| Filesystem                       | `/` read-only; only `/tmp/mutant_hunter_episode_<uuid>/` is RW |
| User                             | non-root (uid 1000)                    |
| Environment                      | scrubbed: only `PATH`, `HOME`, `LANG`, `MUTANT_HUNTER_*` survive |

The subprocess return code is checked **separately** from pytest's
exit reporting. A `sys.exit(0)` from inside the test does not register
as "all tests passed" — pytest's structured JSON report (via
`--report-log`) is the source of truth.

### 4. Mutant truth-table integrity

The list of surviving mutants from the baseline run is **never** placed on
the read-only filesystem the agent can see. It is held in the
`State.surviving_mutants` field, which lives only in the parent process
and is consulted post-hoc to compute the reward. The agent's
`get_mutation_report` tool returns a redacted, line-anchored summary —
not the AST-level patch — which prevents the agent from copy-pasting
the patched code into an `assert`.

### 5. Module-shadow protection

The episode workspace under `/tmp/mutant_hunter_episode_<uuid>/` contains
**only** the test file the agent submitted. The module under test is
imported from a separate path that is earlier on `sys.path` and is
read-only. Even if the agent could write into the episode workspace
(it cannot, after the static checks), it could not shadow the real
module.

### 6. Budget

Tool calls are budgeted at 5 per episode (configurable via
`MutantHunterEnvironment(max_turns=...)`). `submit_tests` does not
consume budget — but it ends the episode immediately, so there is no
"submit, observe reward, retry" loop.

## What this sandbox does NOT defend against

- A genuine kernel exploit. We assume the host kernel and Python interpreter
  are not vulnerable; if they are, this sandbox is the wrong layer.
- Side-channels (timing, cache). The reward signal is too coarse for these
  to matter at training scale.
- A reward function that is itself wrong. The sandbox guards execution; the
  rubric is what makes the reward meaningful, and lives in
  `src/mutant_hunter/rubric/`.

## Verification

Every claim in this document is exercised by an adversarial test in
`evaluation/reward_hacking_tests.py`. If you change the sandbox, those
tests must still pass — and you should add a new one for any new vector
you considered.
