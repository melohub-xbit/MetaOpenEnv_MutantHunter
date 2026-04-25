"""Existing test suite for ``bloom_filter_lite.bloom``.

DELIBERATELY WEAK. Mutation-testing target — the suite below covers only
direct happy paths and skips, among other things:

    * Off-by-one bounds on ``capacity`` and ``error_rate`` validation.
    * The "no false negatives" invariant under many inserts.
    * Saturation / FPR estimates after a known number of inserts.
    * Insertion-count semantics (duplicates are still counted).
    * ``merge`` shape-mismatch error paths and ``other`` type checks.
    * ``copy`` independence (mutating a copy must not affect the original).
    * ``clear`` resetting bits AND count.
    * ``optimal_parameters`` rounding behaviour at very small / large inputs.
    * Bytes / int / bool / str encoding consistency in :func:`_encode`.
"""

from bloom_filter_lite.bloom import (
    BloomFilter,
    BloomFilterError,
    optimal_parameters,
)


def test_optimal_parameters_returns_two_positive_ints():
    m, k = optimal_parameters(1000, 0.01)
    assert isinstance(m, int)
    assert isinstance(k, int)
    assert m > 0
    assert k > 0


def test_add_then_contains_true():
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.add("hello")
    assert "hello" in bf


def test_missing_item_not_contained():
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.add("hello")
    # "definitely-not-inserted" is unlikely to collide at this small load.
    assert "definitely-not-inserted-xyz" not in bf


def test_len_after_two_adds():
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.add("a")
    bf.add("b")
    assert len(bf) == 2


def test_capacity_property():
    bf = BloomFilter(capacity=500, error_rate=0.01)
    assert bf.capacity == 500


def test_target_error_rate_property():
    bf = BloomFilter(capacity=500, error_rate=0.01)
    assert bf.target_error_rate == 0.01


def test_bit_size_positive():
    bf = BloomFilter(capacity=500, error_rate=0.01)
    assert bf.bit_size > 0


def test_num_hashes_positive():
    bf = BloomFilter(capacity=500, error_rate=0.01)
    assert bf.num_hashes > 0


def test_saturation_zero_when_empty():
    bf = BloomFilter(capacity=100, error_rate=0.01)
    assert bf.saturation == 0.0


def test_saturation_increases_after_add():
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.add("hello")
    assert bf.saturation > 0.0


def test_copy_contains_same_item():
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.add("hello")
    other = bf.copy()
    assert "hello" in other


def test_merge_union_membership():
    a = BloomFilter(capacity=100, error_rate=0.01)
    b = BloomFilter(capacity=100, error_rate=0.01)
    a.add("alpha")
    b.add("beta")
    merged = a.merge(b)
    assert "alpha" in merged
    assert "beta" in merged


def test_clear_removes_membership():
    bf = BloomFilter(capacity=100, error_rate=0.01)
    bf.add("hello")
    bf.clear()
    assert "hello" not in bf


def test_invalid_capacity_raises():
    try:
        BloomFilter(capacity=0, error_rate=0.01)
    except BloomFilterError:
        return
    raise AssertionError("expected BloomFilterError")
