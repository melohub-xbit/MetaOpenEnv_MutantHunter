"""Half-open ``[start, end)`` interval container.

A flat, list-backed implementation — *not* a balanced augmented tree. The
shape is wrong for asymptotic performance, but right for mutation testing:
every comparison, boundary, and overlap predicate sits in plain Python where
mutmut can flip it.

Conventions:
    * Intervals are half-open: ``start`` is inclusive, ``end`` is exclusive.
    * ``start < end`` is required; zero-width intervals are rejected.
    * ``start`` and ``end`` may be ``int`` or ``float``; mixing is fine as
      long as the comparison is well defined.
    * "Overlap" means strictly positive intersection length.
      ``[1, 3)`` and ``[3, 5)`` do **not** overlap.
    * Containers are ordered by ``(start, end, payload-id)`` for stable
      iteration; equal intervals with different payloads are kept distinct.
"""

from __future__ import annotations

from typing import Any, Iterator, Tuple

# An ``Interval`` is the public 3-tuple shape returned to callers.
Interval = Tuple[float, float, Any]


class IntervalError(ValueError):
    """Raised on invalid interval arguments (e.g. ``start >= end``)."""


def _validate(start: float, end: float) -> None:
    if start >= end:
        raise IntervalError(f"start must be < end, got [{start}, {end})")


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """True iff two half-open intervals overlap by a strictly positive length."""
    return a_start < b_end and b_start < a_end


def _contains_point(start: float, end: float, point: float) -> bool:
    """True iff ``point`` lies in the half-open interval ``[start, end)``."""
    return start <= point < end


class IntervalTree:
    """Container of half-open intervals with overlap / point queries.

    The implementation stores intervals in a list kept sorted by
    ``(start, end)``. ``add`` is ``O(n)`` (insertion + shift); queries are
    ``O(n)``. Stability is intentional for mutation-testing leverage —
    callers see deterministic iteration order.
    """

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[Interval] = []

    # ----- mutation -------------------------------------------------------- #

    def add(self, start: float, end: float, payload: Any = None) -> None:
        """Insert ``[start, end) -> payload`` into the tree.

        Duplicates (same ``start``, ``end``, and ``payload``) are allowed
        and stored as separate entries. Insertion preserves the
        ``(start, end)`` sort order.
        """
        _validate(start, end)
        new_item: Interval = (start, end, payload)
        # Linear scan to find the first item that should sort after `new_item`.
        idx = 0
        while idx < len(self._items):
            cur = self._items[idx]
            if (cur[0], cur[1]) > (start, end):
                break
            idx += 1
        self._items.insert(idx, new_item)

    def remove(self, start: float, end: float, payload: Any = None) -> bool:
        """Remove the first matching ``[start, end)`` (with matching payload).

        Returns True if an item was removed, False otherwise. Only one
        matching entry is removed per call, even if duplicates exist.
        """
        _validate(start, end)
        for idx, (s, e, p) in enumerate(self._items):
            if s == start and e == end and p == payload:
                del self._items[idx]
                return True
        return False

    def clear(self) -> None:
        """Drop all stored intervals."""
        self._items = []

    # ----- queries --------------------------------------------------------- #

    def query_point(self, point: float) -> list[Interval]:
        """Return all intervals that contain ``point``, in insertion-sort order."""
        return [
            (s, e, p)
            for (s, e, p) in self._items
            if _contains_point(s, e, point)
        ]

    def query_range(self, start: float, end: float) -> list[Interval]:
        """Return all intervals that overlap ``[start, end)`` (positive intersection)."""
        _validate(start, end)
        return [
            (s, e, p)
            for (s, e, p) in self._items
            if _overlaps(s, e, start, end)
        ]

    def overlaps(self, start: float, end: float) -> bool:
        """True iff any stored interval overlaps ``[start, end)``."""
        _validate(start, end)
        for (s, e, _) in self._items:
            if _overlaps(s, e, start, end):
                return True
        return False

    def total_length(self) -> float:
        """Sum of ``end - start`` across all stored intervals (with duplicates)."""
        return sum(e - s for (s, e, _) in self._items)

    # ----- bulk ops -------------------------------------------------------- #

    def merge_overlapping(self) -> int:
        """Merge contiguous / overlapping intervals in place. Payloads are dropped.

        Two intervals are considered mergeable when they overlap *or* touch
        (``a.end == b.start``). The merged interval keeps payload ``None``.
        Returns the number of intervals removed by merging (i.e. the
        reduction in ``len(self)``).
        """
        if not self._items:
            return 0
        before = len(self._items)
        merged: list[Interval] = []
        items_sorted = sorted(self._items, key=lambda it: (it[0], it[1]))
        cur_start, cur_end, _ = items_sorted[0]
        for (s, e, _p) in items_sorted[1:]:
            if s <= cur_end:
                if e > cur_end:
                    cur_end = e
            else:
                merged.append((cur_start, cur_end, None))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end, None))
        self._items = merged
        return before - len(self._items)

    # ----- dunder methods -------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Interval]:
        return iter(list(self._items))

    def __contains__(self, item: object) -> bool:
        """Membership test.

        * ``(start, end)`` checks for any stored interval with that exact span.
        * ``(start, end, payload)`` checks the full triple.
        * Anything else returns ``False``.
        """
        if not isinstance(item, tuple):
            return False
        if len(item) == 2:
            start, end = item  # type: ignore[misc]
            for (s, e, _p) in self._items:
                if s == start and e == end:
                    return True
            return False
        if len(item) == 3:
            start, end, payload = item  # type: ignore[misc]
            for (s, e, p) in self._items:
                if s == start and e == end and p == payload:
                    return True
            return False
        return False

    def __repr__(self) -> str:
        return f"IntervalTree({self._items!r})"


__all__ = [
    "Interval",
    "IntervalError",
    "IntervalTree",
]
