"""interval_tree — small half-open ``[start, end)`` interval container.

Self-contained mini-library used as a mutation-testing target. The module of
record for the corpus manifest is ``interval_tree.tree``.
"""

from interval_tree.tree import (
    Interval,
    IntervalError,
    IntervalTree,
)

__all__ = [
    "Interval",
    "IntervalError",
    "IntervalTree",
]
