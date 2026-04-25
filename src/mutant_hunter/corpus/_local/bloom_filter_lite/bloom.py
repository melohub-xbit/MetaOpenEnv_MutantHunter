"""Small bit-array Bloom filter.

Hand-rolled — no third-party Bloom library, no ``bitarray`` dependency. The
arithmetic for ``m`` (bits) and ``k`` (hash functions) is laid out in plain
Python so mutation testing can bite into every constant and operator.

References:
    * Burton H. Bloom, "Space/Time Trade-offs in Hash Coding with Allowable
      Errors" (1970).
    * Standard sizing formulas:
          m = ceil(-n * ln(p) / (ln 2)^2)
          k = round((m / n) * ln 2)
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

LN2 = math.log(2)
LN2_SQUARED = LN2 * LN2


class BloomFilterError(ValueError):
    """Raised on invalid Bloom-filter arguments or shape-mismatched merges."""


def optimal_parameters(capacity: int, error_rate: float) -> tuple[int, int]:
    """Return ``(m, k)`` — bit-array size and number of hash functions.

    Args:
        capacity: Expected number of distinct elements ``n``.
        error_rate: Target false-positive probability ``p`` in ``(0, 1)``.

    Both bounds are exclusive. ``m`` is at least 1 and ``k`` is at least 1
    even for tiny inputs that would otherwise round to zero.
    """
    if not isinstance(capacity, int) or isinstance(capacity, bool):
        raise BloomFilterError(f"capacity must be int, got {type(capacity).__name__}")
    if capacity <= 0:
        raise BloomFilterError(f"capacity must be positive, got {capacity}")
    if not isinstance(error_rate, (int, float)) or isinstance(error_rate, bool):
        raise BloomFilterError(
            f"error_rate must be a real number, got {type(error_rate).__name__}"
        )
    if error_rate <= 0.0 or error_rate >= 1.0:
        raise BloomFilterError(f"error_rate must be in (0, 1), got {error_rate}")
    m = int(math.ceil(-capacity * math.log(error_rate) / LN2_SQUARED))
    if m < 1:
        m = 1
    k = int(round((m / capacity) * LN2))
    if k < 1:
        k = 1
    return m, k


class BloomFilter:
    """A space-efficient probabilistic set.

    Membership queries are exact for negatives but allow false positives —
    never false negatives. The filter is sized at construction from a
    target ``capacity`` and ``error_rate`` and never resizes.

    Two filters are *shape-compatible* when their bit count ``m`` and hash
    count ``k`` are equal; only shape-compatible filters can be merged.
    """

    __slots__ = ("_capacity", "_error_rate", "_m", "_k", "_bits", "_count")

    def __init__(self, capacity: int, error_rate: float = 0.01) -> None:
        m, k = optimal_parameters(capacity, error_rate)
        self._capacity = capacity
        self._error_rate = float(error_rate)
        self._m = m
        self._k = k
        self._bits = bytearray((m + 7) // 8)
        self._count = 0

    # ----- shape introspection -------------------------------------------- #

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def target_error_rate(self) -> float:
        return self._error_rate

    @property
    def bit_size(self) -> int:
        return self._m

    @property
    def num_hashes(self) -> int:
        return self._k

    # ----- runtime metrics ------------------------------------------------- #

    @property
    def saturation(self) -> float:
        """Fraction of bits in ``[0, m)`` that are currently set."""
        if self._m <= 0:
            return 0.0
        ones = 0
        # Count set bits across full bytes; the trailing bits past index m-1
        # are never set (we always index modulo m), so a plain popcount
        # over the whole bytearray is correct.
        for byte in self._bits:
            ones += bin(byte).count("1")
        return ones / self._m

    @property
    def false_positive_rate(self) -> float:
        """Estimated FPR given current saturation: ``saturation ** k``."""
        return self.saturation ** self._k

    def __len__(self) -> int:
        """Number of times :meth:`add` has been called.

        Note this counts duplicates — a Bloom filter cannot tell them apart
        from genuinely new elements, so the value is an upper bound on the
        true number of distinct items inserted.
        """
        return self._count

    # ----- hashing --------------------------------------------------------- #

    @staticmethod
    def _encode(item: Any) -> bytes:
        if isinstance(item, bytes):
            return item
        if isinstance(item, str):
            return item.encode("utf-8")
        if isinstance(item, bool):
            return b"\x01" if item else b"\x00"
        if isinstance(item, int):
            # Two's-complement signed encoding so negatives are distinguishable.
            length = max(1, (item.bit_length() + 8) // 8)
            return item.to_bytes(length, "big", signed=True)
        return repr(item).encode("utf-8")

    def _hash_indices(self, item: Any) -> list[int]:
        data = self._encode(item)
        # Two independent hash digests, combined by the Kirsch-Mitzenmacher
        # double-hashing trick: g_i(x) = (h1(x) + i * h2(x)) mod m.
        digest1 = hashlib.blake2b(data, digest_size=8, person=b"blf-h1--").digest()
        digest2 = hashlib.blake2b(data, digest_size=8, person=b"blf-h2--").digest()
        h1 = int.from_bytes(digest1, "big")
        h2 = int.from_bytes(digest2, "big")
        if h2 == 0:
            h2 = 1
        m = self._m
        return [(h1 + i * h2) % m for i in range(self._k)]

    # ----- core operations ------------------------------------------------- #

    def add(self, item: Any) -> None:
        """Insert ``item`` into the filter."""
        for idx in self._hash_indices(item):
            byte_idx, bit_idx = divmod(idx, 8)
            self._bits[byte_idx] |= 1 << bit_idx
        self._count += 1

    def __contains__(self, item: Any) -> bool:
        """Membership test. May return false positives, never false negatives."""
        for idx in self._hash_indices(item):
            byte_idx, bit_idx = divmod(idx, 8)
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    # ----- bulk / structural ops ------------------------------------------ #

    def merge(self, other: "BloomFilter") -> "BloomFilter":
        """Return a new filter representing the union of ``self`` and ``other``.

        Raises :class:`BloomFilterError` if the two filters have different
        bit counts ``m`` or hash counts ``k``.
        """
        if not isinstance(other, BloomFilter):
            raise BloomFilterError(
                f"can only merge with BloomFilter, got {type(other).__name__}"
            )
        if self._m != other._m or self._k != other._k:
            raise BloomFilterError(
                "cannot merge: shape mismatch "
                f"(m={self._m}, k={self._k}) vs (m={other._m}, k={other._k})"
            )
        merged = BloomFilter.__new__(BloomFilter)
        merged._capacity = max(self._capacity, other._capacity)
        merged._error_rate = max(self._error_rate, other._error_rate)
        merged._m = self._m
        merged._k = self._k
        merged._bits = bytearray(a | b for a, b in zip(self._bits, other._bits))
        merged._count = self._count + other._count
        return merged

    def copy(self) -> "BloomFilter":
        """Return an independent copy of this filter."""
        c = BloomFilter.__new__(BloomFilter)
        c._capacity = self._capacity
        c._error_rate = self._error_rate
        c._m = self._m
        c._k = self._k
        c._bits = bytearray(self._bits)
        c._count = self._count
        return c

    def clear(self) -> None:
        """Reset all bits to zero and the insertion counter to zero."""
        self._bits = bytearray((self._m + 7) // 8)
        self._count = 0

    def __repr__(self) -> str:
        return (
            f"BloomFilter(capacity={self._capacity}, "
            f"error_rate={self._error_rate}, m={self._m}, k={self._k}, "
            f"count={self._count})"
        )


__all__ = [
    "BloomFilter",
    "BloomFilterError",
    "optimal_parameters",
]
