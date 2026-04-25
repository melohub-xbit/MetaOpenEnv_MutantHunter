"""Utility helpers shared across the env, rubric, and tools.

Modules:

    logging.py        — structured logger setup (rich + JSONL fallback)
    tracing.py        — per-step trace capture for inspect_rollouts
    pytest_helpers.py — discover/parse/run pytest collections programmatically
"""

from __future__ import annotations

__all__: list[str] = []
