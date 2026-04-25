"""csv_normalizer — small CSV parsing / normalisation helpers.

Self-contained mini-library used as a mutation-testing target. The module of
record for the corpus manifest is ``csv_normalizer.normalizer``.
"""

from csv_normalizer.normalizer import (
    NormalizationError,
    coerce_value,
    detect_delimiter,
    is_numeric,
    normalize_header,
    parse_csv,
    parse_row,
    strip_bom,
    write_csv,
    write_row,
)

__all__ = [
    "NormalizationError",
    "coerce_value",
    "detect_delimiter",
    "is_numeric",
    "normalize_header",
    "parse_csv",
    "parse_row",
    "strip_bom",
    "write_csv",
    "write_row",
]
