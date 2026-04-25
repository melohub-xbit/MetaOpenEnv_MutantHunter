"""Existing test suite for ``interval_tree.tree``.

DELIBERATELY WEAK. Mutation-testing target — the suite below covers only
direct happy paths and skips, among other things:

    * Half-open boundary semantics (``[1, 3)`` does NOT contain ``3``;
      ``[1, 3)`` and ``[3, 5)`` do NOT overlap).
    * Duplicate insertion and per-call single-removal of duplicates.
    * Stable sort order across equal ``start`` keys.
    * ``query_range`` boundary edges and zero-length query rejection.
    * ``merge_overlapping`` of touching intervals (``[1, 3)`` + ``[3, 5)``).
    * Payload preservation vs. payload dropping in ``merge_overlapping``.
    * ``__contains__`` with the 3-tuple ``(start, end, payload)`` form
      and with non-tuple inputs.
    * ``IntervalError`` paths (e.g. ``start >= end``, zero-width).
    * ``total_length`` and ``overlaps`` predicates.
"""

from interval_tree.tree import IntervalTree


def test_add_then_len():
    t = IntervalTree()
    t.add(1, 5)
    t.add(10, 20)
    assert len(t) == 2


def test_query_point_inside():
    t = IntervalTree()
    t.add(1, 5, payload="A")
    hits = t.query_point(3)
    assert hits == [(1, 5, "A")]


def test_query_point_outside():
    t = IntervalTree()
    t.add(1, 5, payload="A")
    assert t.query_point(10) == []


def test_query_range_overlap():
    t = IntervalTree()
    t.add(1, 5, payload="A")
    t.add(10, 20, payload="B")
    hits = t.query_range(4, 12)
    assert hits == [(1, 5, "A"), (10, 20, "B")]


def test_overlaps_true():
    t = IntervalTree()
    t.add(1, 5)
    assert t.overlaps(3, 7) is True


def test_overlaps_false():
    t = IntervalTree()
    t.add(1, 5)
    assert t.overlaps(10, 20) is False


def test_remove_existing_returns_true():
    t = IntervalTree()
    t.add(1, 5, payload="A")
    assert t.remove(1, 5, payload="A") is True
    assert len(t) == 0


def test_remove_missing_returns_false():
    t = IntervalTree()
    t.add(1, 5, payload="A")
    assert t.remove(7, 9, payload="A") is False


def test_clear():
    t = IntervalTree()
    t.add(1, 5)
    t.add(10, 20)
    t.clear()
    assert len(t) == 0


def test_iter_yields_all():
    t = IntervalTree()
    t.add(1, 5, payload="A")
    t.add(10, 20, payload="B")
    items = list(iter(t))
    assert items == [(1, 5, "A"), (10, 20, "B")]


def test_contains_two_tuple():
    t = IntervalTree()
    t.add(1, 5, payload="A")
    assert (1, 5) in t


def test_total_length_simple():
    t = IntervalTree()
    t.add(1, 5)
    t.add(10, 20)
    assert t.total_length() == 14


def test_merge_overlapping_two_overlap():
    t = IntervalTree()
    t.add(1, 5)
    t.add(3, 8)
    removed = t.merge_overlapping()
    assert removed == 1
    assert list(iter(t)) == [(1, 8, None)]


def test_merge_overlapping_disjoint_unchanged():
    t = IntervalTree()
    t.add(1, 3)
    t.add(10, 20)
    removed = t.merge_overlapping()
    assert removed == 0
    assert len(t) == 2
