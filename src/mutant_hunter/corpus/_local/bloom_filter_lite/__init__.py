"""bloom_filter_lite — small bit-array Bloom filter implementation.

Self-contained mini-library used as a mutation-testing target. The module of
record for the corpus manifest is ``bloom_filter_lite.bloom``.
"""

from bloom_filter_lite.bloom import (
    BloomFilter,
    BloomFilterError,
    optimal_parameters,
)

__all__ = [
    "BloomFilter",
    "BloomFilterError",
    "optimal_parameters",
]
