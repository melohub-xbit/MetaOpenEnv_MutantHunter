"""AST + regex denylist on agent-submitted test code.

Two layers of defense:

1. Regex pass over the raw source — catches anything the AST pass might
   miss (string-of-import tricks, reflection via getattr-on-string, etc.).
2. AST pass that flags forbidden imports, calls, attribute chains, and
   ``open(...)`` write/append/exclusive modes.

The list is intentionally generous; the rubric uses
``contains_forbidden = True`` to collapse the format reward to 0, which
combined with the no-regression gate produces a final reward of 0.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ForbiddenFinding:
    kind: str
    message: str
    lineno: int | None = None
    col: int | None = None


_FORBIDDEN_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "urllib",
    "urllib2",
    "urllib3",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "telnetlib",
    "smtplib",
    "pickle",
    "marshal",
    "shelve",
    "ctypes",
    "cffi",
    "mmap",
    "multiprocessing",
    "threading",
    "concurrent",
    "asyncio",
    "signal",
    "shutil",
    "pathlib",
    "tempfile",
    "fcntl",
    "platform",
    "gc",
    "weakref",
    "importlib",
}

_FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "input",
    "exit",
    "quit",
    "memoryview",
}

_FORBIDDEN_ATTR_REGEXES = [
    re.compile(r"^pytest\._"),
    re.compile(r"^_pytest\."),
    re.compile(r"^coverage\._"),
    re.compile(r"^mutant_hunter\."),
    re.compile(r"^sys\.modules$"),
    re.compile(r"^sys\.path"),
]

_FORBIDDEN_TEXT_REGEXES = [
    re.compile(r"\b(os\.system|os\.popen|os\.remove|os\.unlink|subprocess\.)\b"),
    re.compile(r"\b(requests|httpx|aiohttp)\b"),
    re.compile(r"\b(socket|ftplib|telnetlib|smtplib)\b"),
    re.compile(r"\b(__builtins__|__import__|sys\.path|sys\.modules|sys\.exit)\b"),
    re.compile(r"\b(pytest\.skip|pytest\.exit|pytest\.fail|os\._exit)\b"),
    re.compile(r"\bopen\s*\(.*?,\s*['\"][wax\+]", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bcompile\s*\("),
    re.compile(r"\bgetattr\s*\(\s*__builtins__"),
    re.compile(r"\bsetattr\s*\(\s*__builtins__"),
]


def scan_forbidden_patterns(source: str) -> list[ForbiddenFinding]:
    findings: list[ForbiddenFinding] = []

    for rx in _FORBIDDEN_TEXT_REGEXES:
        if rx.search(source):
            findings.append(
                ForbiddenFinding(
                    kind="regex",
                    message=f"forbidden pattern matched: {rx.pattern}",
                )
            )

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        findings.append(
            ForbiddenFinding(
                kind="syntax",
                message=f"syntax error: {e.msg}",
                lineno=getattr(e, "lineno", None),
                col=getattr(e, "offset", None),
            )
        )
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FORBIDDEN_IMPORTS:
                    findings.append(
                        ForbiddenFinding(
                            kind="import",
                            message=f"forbidden import: {alias.name}",
                            lineno=getattr(node, "lineno", None),
                            col=getattr(node, "col_offset", None),
                        )
                    )

        if isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _FORBIDDEN_IMPORTS:
                    findings.append(
                        ForbiddenFinding(
                            kind="import",
                            message=f"forbidden import: {node.module}",
                            lineno=getattr(node, "lineno", None),
                            col=getattr(node, "col_offset", None),
                        )
                    )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                findings.append(
                    ForbiddenFinding(
                        kind="call",
                        message=f"forbidden call: {node.func.id}(...)",
                        lineno=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                    )
                )
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                    if isinstance(mode, str) and any(ch in mode for ch in ("w", "a", "x", "+")):
                        findings.append(
                            ForbiddenFinding(
                                kind="fs",
                                message=f"forbidden open() mode: {mode!r}",
                                lineno=getattr(node, "lineno", None),
                                col=getattr(node, "col_offset", None),
                            )
                        )
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                        if isinstance(mode, str) and any(ch in mode for ch in ("w", "a", "x", "+")):
                            findings.append(
                                ForbiddenFinding(
                                    kind="fs",
                                    message=f"forbidden open() mode: {mode!r}",
                                    lineno=getattr(node, "lineno", None),
                                    col=getattr(node, "col_offset", None),
                                )
                            )

        if isinstance(node, ast.Attribute):
            full = _attr_chain(node)
            if full and any(rx.match(full) for rx in _FORBIDDEN_ATTR_REGEXES):
                findings.append(
                    ForbiddenFinding(
                        kind="attr",
                        message=f"forbidden attribute access: {full}",
                        lineno=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                    )
                )

    return findings


def _attr_chain(node: ast.Attribute) -> str | None:
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return None
    return ".".join(reversed(parts))
