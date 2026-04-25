"""Existing test suite for ``csv_normalizer.normalizer``.

DELIBERATELY WEAK. This is a mutation-testing target: the suite below
covers only the most direct happy paths and skips, among other things:

    * Quoted fields containing the delimiter (``"a,b",c``).
    * Escaped double-quotes inside quoted fields (``""`` → literal ``"``).
    * BOM-prefixed input round-tripping through :func:`parse_csv`.
    * Empty input and whitespace-only input.
    * Tab, semicolon, and pipe delimiters (auto-detection on each).
    * Header de-duplication via the ``_2`` / ``_3`` suffix machinery.
    * Padding short rows with ``None``.
    * Rejecting rows that are wider than the header.
    * Negative numbers, exponential floats, and ``NaN`` / ``Inf`` rejection
      in :func:`coerce_value` / :func:`is_numeric`.
    * Boolean ``False`` and the ``yes`` / ``no`` aliases.
    * ``write_row`` quoting fields that contain delimiters, quotes, or
      newlines, and the round-trip with :func:`parse_row`.
    * Error paths (e.g. unterminated quoted field, multi-character
      delimiter, ``"`` as delimiter).
"""

from csv_normalizer.normalizer import (
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


def test_parse_row_simple():
    assert parse_row("a,b,c") == ["a", "b", "c"]


def test_parse_row_quoted_simple():
    assert parse_row('"a","b","c"') == ["a", "b", "c"]


def test_normalize_header_simple():
    assert normalize_header("First Name") == "first_name"


def test_strip_bom_no_bom():
    assert strip_bom("hello") == "hello"


def test_detect_delimiter_comma():
    assert detect_delimiter("a,b,c\n1,2,3") == ","


def test_coerce_value_int():
    assert coerce_value("42") == 42


def test_coerce_value_float():
    assert coerce_value("3.14") == 3.14


def test_coerce_value_true():
    assert coerce_value("true") is True


def test_coerce_value_empty_is_none():
    assert coerce_value("") is None


def test_is_numeric_int():
    assert is_numeric("42") is True


def test_is_numeric_letters():
    assert is_numeric("abc") is False


def test_parse_csv_basic():
    text = "name,age\nAlice,30\nBob,25"
    rows = parse_csv(text)
    assert rows == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]


def test_write_row_simple():
    assert write_row(["a", "b", "c"]) == "a,b,c"


def test_write_csv_basic():
    rows = [{"name": "Alice", "age": 30}]
    out = write_csv(rows)
    assert "name,age" in out
    assert "Alice,30" in out
