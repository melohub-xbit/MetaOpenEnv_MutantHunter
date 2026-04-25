from __future__ import annotations

from mutant_hunter.safety.forbidden_patterns import scan_forbidden_patterns
from mutant_hunter.safety.validators import validate_test_code


def test_accept_minimal_valid_test() -> None:
    r = validate_test_code("def test_a():\n    assert 1 + 1 == 2\n")
    assert r.ok, r.findings


def test_reject_no_test_functions() -> None:
    r = validate_test_code("x = 1\n")
    assert not r.ok
    assert any(f.kind == "structure" for f in r.findings)


def test_reject_forbidden_import() -> None:
    r = validate_test_code("import os\n\ndef test_a():\n    assert True\n")
    assert not r.ok
    assert any(f.kind == "import" for f in r.findings)


def test_reject_top_level_statement() -> None:
    r = validate_test_code("print('x')\n\ndef test_a():\n    assert True\n")
    assert not r.ok


def test_reject_eval_call() -> None:
    r = validate_test_code("def test_a():\n    eval('1')\n    assert True\n")
    assert not r.ok
    assert any(f.kind in ("call", "regex") for f in r.findings)


def test_reject_open_write_mode() -> None:
    r = validate_test_code(
        "def test_a():\n    open('x','w').write('y')\n    assert True\n"
    )
    assert not r.ok
    assert any(f.kind in ("fs", "regex") for f in r.findings)


def test_reject_size_overflow() -> None:
    big = "# " + "X" * 60_000 + "\n\ndef test_a():\n    assert True\n"
    r = validate_test_code(big)
    assert not r.ok
    assert any(f.kind == "size" for f in r.findings)


def test_reject_too_many_tests() -> None:
    src = "\n".join(f"def test_{i}():\n    assert True\n" for i in range(60))
    r = validate_test_code(src)
    assert not r.ok
    assert any("too many" in f.message for f in r.findings)


def test_reject_subscript_assignment() -> None:
    r = validate_test_code(
        "__builtins__['eval'] = lambda x: 1\n\ndef test_a():\n    assert True\n"
    )
    assert not r.ok


def test_scan_forbidden_patterns_independent_of_validator() -> None:
    findings = scan_forbidden_patterns("import socket\n\ndef test_a(): pass\n")
    assert any(f.kind == "import" for f in findings)
