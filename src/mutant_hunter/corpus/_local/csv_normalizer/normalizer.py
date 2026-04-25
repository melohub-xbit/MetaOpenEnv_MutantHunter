"""Tiny CSV parsing / normalisation helpers.

Hand-rolled (no ``csv`` stdlib delegation) so mutation testing has a wide
surface of branches, comparisons, and string handling to bite into. The
parser handles RFC-4180-flavoured CSV: a single-character delimiter,
double-quoted fields, ``""`` as an escaped quote inside a quoted field. It
is line-oriented — newlines inside quoted fields are not supported.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

# Order matters: the first matching candidate wins on a tie.
DEFAULT_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")

UTF8_BOM = "﻿"

_TRUE_LITERALS = frozenset({"true", "yes"})
_FALSE_LITERALS = frozenset({"false", "no"})


class NormalizationError(ValueError):
    """Raised on irrecoverable CSV-shape errors (e.g. unbalanced quotes)."""


# --------------------------------------------------------------------------- #
# Header normalisation                                                        #
# --------------------------------------------------------------------------- #


def normalize_header(header: str) -> str:
    """Return a snake_case-ish identifier for a header label.

    Rules:
        * Lower-case the input.
        * Replace any maximal run of non-alphanumeric characters with a
          single ``_``.
        * Strip leading and trailing underscores.
        * Return ``"col"`` if the result would otherwise be empty (so the
          caller always gets a non-empty identifier-ish string).
    """
    if not isinstance(header, str):
        raise TypeError(f"normalize_header expects str, got {type(header).__name__}")
    out_chars: list[str] = []
    last_underscore = True  # leading punctuation is collapsed to nothing
    for ch in header.lower():
        if ch.isalnum():
            out_chars.append(ch)
            last_underscore = False
        else:
            if not last_underscore:
                out_chars.append("_")
                last_underscore = True
    while out_chars and out_chars[-1] == "_":
        out_chars.pop()
    if not out_chars:
        return "col"
    return "".join(out_chars)


# --------------------------------------------------------------------------- #
# BOM handling                                                                #
# --------------------------------------------------------------------------- #


def strip_bom(text: str) -> str:
    """Remove a leading UTF-8 BOM if present, returning the rest."""
    if text.startswith(UTF8_BOM):
        return text[len(UTF8_BOM):]
    return text


# --------------------------------------------------------------------------- #
# Delimiter detection                                                         #
# --------------------------------------------------------------------------- #


def detect_delimiter(text: str, candidates: Sequence[str] = DEFAULT_DELIMITERS) -> str:
    """Pick the delimiter that splits the first few lines most consistently.

    Strategy: for each candidate, count its occurrences on each of up to
    the first five non-empty lines. A candidate's score is the modal count
    minus a half-weight variance penalty; the highest-scoring candidate
    wins, with ties broken by ``candidates`` order.
    """
    if not text:
        raise NormalizationError("cannot detect delimiter from empty input")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""][:5]
    if not lines:
        raise NormalizationError("cannot detect delimiter: no non-empty lines")
    best_score = -1.0
    best_candidate = candidates[0]
    for cand in candidates:
        if len(cand) != 1:
            raise ValueError(f"delimiter candidate must be a single char, got {cand!r}")
        counts = [ln.count(cand) for ln in lines]
        modal = max(counts)
        if modal == 0:
            continue
        variance = sum((c - modal) ** 2 for c in counts) / len(counts)
        score = float(modal) - 0.5 * variance
        if score > best_score:
            best_score = score
            best_candidate = cand
    if best_score < 0:
        raise NormalizationError(
            f"could not detect delimiter from candidates {candidates!r}"
        )
    return best_candidate


# --------------------------------------------------------------------------- #
# Row / CSV parsing                                                           #
# --------------------------------------------------------------------------- #


def parse_row(line: str, delimiter: str = ",") -> list[str]:
    """Parse a single CSV line into a list of field strings.

    Recognises double-quoted fields and ``""`` as an escaped quote inside a
    quoted field. Whitespace adjacent to delimiters is preserved exactly —
    callers that want trimming should ``.strip()`` the result themselves.
    Raises :class:`NormalizationError` on an unterminated quoted field.
    """
    if len(delimiter) != 1:
        raise ValueError(f"delimiter must be a single character, got {delimiter!r}")
    if delimiter == '"':
        raise ValueError("delimiter must not be the double-quote character")

    fields: list[str] = []
    buf: list[str] = []
    in_quotes = False
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < n and line[i + 1] == '"':
                    # Escaped quote inside quoted field.
                    buf.append('"')
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue
        # Outside a quoted field.
        if ch == '"' and not buf:
            in_quotes = True
            i += 1
            continue
        if ch == delimiter:
            fields.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if in_quotes:
        raise NormalizationError(f"unterminated quoted field in row: {line!r}")
    fields.append("".join(buf))
    return fields


def parse_csv(text: str, delimiter: str | None = None) -> list[dict[str, Any]]:
    """Parse a full CSV document into a list of dicts using row 1 as the header.

    * Strips a leading UTF-8 BOM if present.
    * Auto-detects the delimiter when ``delimiter is None``.
    * Normalises header labels via :func:`normalize_header` and disambiguates
      duplicates by suffixing ``_2``, ``_3``, ...
    * Coerces values via :func:`coerce_value`.
    * Skips fully blank lines.
    * Pads short rows with ``None``; rejects rows wider than the header.
    """
    text = strip_bom(text)
    if not text.strip():
        return []
    if delimiter is None:
        delimiter = detect_delimiter(text)
    raw_lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    header_fields = parse_row(raw_lines[0], delimiter)
    headers: list[str] = []
    seen: dict[str, int] = {}
    for field in header_fields:
        base = normalize_header(field)
        if base not in seen:
            seen[base] = 1
            headers.append(base)
        else:
            seen[base] += 1
            headers.append(f"{base}_{seen[base]}")

    rows: list[dict[str, Any]] = []
    for line in raw_lines[1:]:
        values = parse_row(line, delimiter)
        if len(values) > len(headers):
            raise NormalizationError(
                f"row has {len(values)} fields but header has {len(headers)}: {line!r}"
            )
        padded: list[Any] = list(values) + [None] * (len(headers) - len(values))
        record: dict[str, Any] = {}
        for h, v in zip(headers, padded):
            record[h] = v if v is None else coerce_value(v)
        rows.append(record)
    return rows


# --------------------------------------------------------------------------- #
# Value coercion                                                              #
# --------------------------------------------------------------------------- #


def is_numeric(value: str) -> bool:
    """True iff ``value`` parses as a finite Python int or float.

    Treats ``NaN``, ``+Inf``, ``-Inf`` (and string variants) as non-numeric.
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    if s == "":
        return False
    try:
        int(s)
        return True
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        return False
    if f != f:
        return False
    return f not in (float("inf"), float("-inf"))


def coerce_value(value: str) -> Any:
    """Coerce a raw CSV cell string to ``None`` / ``bool`` / ``int`` / ``float`` / ``str``.

    Rules, in order:

    1. Empty cell after ``.strip()`` → ``None``.
    2. Lower-cased value matches ``true|yes`` / ``false|no`` → ``bool``.
    3. Parses as ``int`` → ``int``.
    4. Parses as a finite ``float`` → ``float``.
    5. Otherwise the original string (whitespace preserved).
    """
    if not isinstance(value, str):
        raise TypeError(f"coerce_value expects str, got {type(value).__name__}")
    stripped = value.strip()
    if stripped == "":
        return None
    lowered = stripped.lower()
    if lowered in _TRUE_LITERALS:
        return True
    if lowered in _FALSE_LITERALS:
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        f = float(stripped)
    except ValueError:
        return value
    if f != f or f in (float("inf"), float("-inf")):
        return value
    return f


# --------------------------------------------------------------------------- #
# Serialisation                                                               #
# --------------------------------------------------------------------------- #


def _needs_quoting(field: str, delimiter: str) -> bool:
    return delimiter in field or '"' in field or "\n" in field or "\r" in field


def _quote(field: str) -> str:
    return '"' + field.replace('"', '""') + '"'


def write_row(values: Iterable[Any], delimiter: str = ",") -> str:
    """Format a single row as a CSV line. Does NOT terminate with a newline.

    ``None`` is rendered as the empty string. ``bool`` is rendered as the
    lowercase literal ``"true"`` / ``"false"`` to round-trip with
    :func:`coerce_value`. Other values are passed through ``str()`` and
    quoted if they contain the delimiter, a quote, or a newline.
    """
    if len(delimiter) != 1:
        raise ValueError(f"delimiter must be a single character, got {delimiter!r}")
    parts: list[str] = []
    for v in values:
        if v is None:
            parts.append("")
            continue
        if isinstance(v, bool):
            parts.append("true" if v else "false")
            continue
        s = str(v)
        if _needs_quoting(s, delimiter):
            parts.append(_quote(s))
        else:
            parts.append(s)
    return delimiter.join(parts)


def write_csv(
    rows: Iterable[dict[str, Any]],
    fields: Sequence[str] | None = None,
    delimiter: str = ",",
) -> str:
    """Serialise an iterable of dict rows to a CSV string.

    The output uses ``"\\n"`` as the line separator and ends with a single
    trailing newline.

    ``fields`` controls header order. When omitted, the union of keys
    across ``rows`` is used in first-seen order.
    """
    rows_list = list(rows)
    if fields is None:
        seen: dict[str, None] = {}
        for row in rows_list:
            for k in row.keys():
                if k not in seen:
                    seen[k] = None
        fields = list(seen.keys())
    out_lines: list[str] = [write_row(fields, delimiter)]
    for row in rows_list:
        values = [row.get(f) for f in fields]
        out_lines.append(write_row(values, delimiter))
    return "\n".join(out_lines) + "\n"


__all__ = [
    "DEFAULT_DELIMITERS",
    "NormalizationError",
    "UTF8_BOM",
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
