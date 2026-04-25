"""Pre-flight structural checks on agent-submitted pytest code.

Run BEFORE any execution. Anything that returns ``ok=False`` is rejected
by the env without invoking pytest. The composition of static checks is:

* size cap on source bytes
* forbidden imports / calls / attribute chains / open() write modes (AST)
* forbidden text-level patterns (regex denylist)
* top-level statement allow-list (imports, assigns, docstring, defs)
* test-function naming + count cap
* requires at least one ``test_*`` function (an empty file is a hack)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from mutant_hunter.safety.forbidden_patterns import ForbiddenFinding, scan_forbidden_patterns


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    findings: list[ForbiddenFinding]


def validate_test_code(
    test_code: str,
    *,
    max_bytes: int = 50_000,
    max_tests: int = 50,
) -> ValidationResult:
    findings: list[ForbiddenFinding] = []

    if len(test_code.encode("utf-8", errors="ignore")) > max_bytes:
        findings.append(
            ForbiddenFinding(kind="size", message=f"test file exceeds {max_bytes} bytes")
        )

    findings.extend(scan_forbidden_patterns(test_code))
    if any(f.kind == "syntax" for f in findings):
        return ValidationResult(ok=False, findings=findings)

    tree = ast.parse(test_code)

    test_fn_count = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("test_"):
                test_fn_count += 1
            elif not node.name.startswith("_"):
                findings.append(
                    ForbiddenFinding(
                        kind="structure",
                        message=f"top-level function must be test_* or _helper, got {node.name}",
                        lineno=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                    )
                )
        elif isinstance(node, ast.AsyncFunctionDef):
            findings.append(
                ForbiddenFinding(
                    kind="structure",
                    message="async def is not allowed at module top level",
                    lineno=getattr(node, "lineno", None),
                )
            )
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                # Count test_* methods inside the class too.
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test_"):
                        test_fn_count += 1
            else:
                findings.append(
                    ForbiddenFinding(
                        kind="structure",
                        message=f"top-level class must be Test*, got {node.name}",
                        lineno=getattr(node, "lineno", None),
                        col=getattr(node, "col_offset", None),
                    )
                )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            pass
        elif isinstance(node, ast.Assign):
            # Allow bare module-level assignments (constants, fixtures-by-name)
            # but reject assignment to subscripts/attributes (a common
            # __builtins__ patch hack).
            for tgt in node.targets:
                if isinstance(tgt, (ast.Subscript, ast.Attribute)):
                    findings.append(
                        ForbiddenFinding(
                            kind="structure",
                            message=f"top-level subscript/attribute assignment is forbidden",
                            lineno=getattr(node, "lineno", None),
                        )
                    )
                    break
        elif isinstance(node, ast.AnnAssign):
            pass
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            # module docstring
            pass
        else:
            findings.append(
                ForbiddenFinding(
                    kind="structure",
                    message=f"forbidden top-level statement: {type(node).__name__}",
                    lineno=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                )
            )

    if test_fn_count == 0:
        findings.append(
            ForbiddenFinding(
                kind="structure",
                message="submission contains no test_* functions",
            )
        )
    if test_fn_count > max_tests:
        findings.append(
            ForbiddenFinding(
                kind="structure",
                message=f"too many tests: {test_fn_count} > {max_tests}",
            )
        )

    return ValidationResult(ok=len(findings) == 0, findings=findings)
