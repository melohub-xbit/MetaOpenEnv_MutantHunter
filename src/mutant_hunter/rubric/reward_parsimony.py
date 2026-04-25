from __future__ import annotations

import ast


def reward_parsimony(test_code: str, *, soft_limit_lines: int = 20, hard_limit_lines: int = 50) -> float:
    """
    Mild penalty for very long tests.
    1.0 when avg test length <= soft_limit_lines.
    Linearly down to 0.0 at hard_limit_lines.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return 0.0

    lengths: list[int] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            if node.body:
                start = getattr(node, "lineno", 1)
                end = getattr(node.body[-1], "end_lineno", getattr(node.body[-1], "lineno", start))
                lengths.append(max(1, int(end) - int(start) + 1))
    if not lengths:
        return 0.0

    avg = sum(lengths) / len(lengths)
    if avg <= soft_limit_lines:
        return 1.0
    if avg >= hard_limit_lines:
        return 0.0
    return max(0.0, 1.0 - (avg - soft_limit_lines) / (hard_limit_lines - soft_limit_lines))

