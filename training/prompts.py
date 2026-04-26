"""System prompt + few-shot examples for the MutantHunter policy.

Tuned for Qwen2.5-Coder-1.5B-Instruct. The env is one-shot: the model sees
module summary + existing tests + tool-history, and must emit a single
pytest file. The blueprint's tool-loop wording is intentionally NOT used
here because the env does not expose those tools to the model — surfacing
fake tool names would just teach the policy to hallucinate calls and waste
token budget.

Public symbols:
    SYSTEM_PROMPT       — instruction prelude
    FEW_SHOT_EXAMPLES   — list of {"prompt": str, "completion": str}
    render_few_shot()   — concatenate examples into a single string block
"""

from __future__ import annotations

from typing import TypedDict


class FewShot(TypedDict):
    prompt: str
    completion: str


# Phrased entirely in positive form. A previous version listed forbidden
# imports (os/sys/subprocess/eval/exec); small Coder-instruct models read
# that list as a recommended import list and regurgitated it. Negative
# rules are now expressed as positive constraints.
SYSTEM_PROMPT = """\
You are MutantHunter, an expert at writing pytest unit tests that catch bugs.

You will be given the FULL source of a Python module and the names of its
existing tests. Your job is to write ADDITIONAL pytest tests that catch
behaviors the existing suite misses — the kinds of small, plausible code
mutations (off-by-one, flipped comparison, swapped operator, dropped
early-return, constant changes) that a mutation tester would otherwise
leave alive.

Grounding (the most important rule):
- Tests MUST pass on the unmodified source code shown in the prompt.
- READ THE ACTUAL IMPLEMENTATION before deciding what to assert. Match the
  existing behavior exactly. DO NOT guess what a function should return
  from its name or signature.
- If the source raises ValueError on bad input, your test must expect
  ValueError — not TypeError, not RuntimeError, not "no exception".
- If the source returns 0 for an empty case, your test must assert == 0,
  not > 0, not None, not raises.
- A test that fails on the unmodified source is worse than no test.

Output format:
- Output ONLY the contents of a single pytest file.
- No markdown fences, no prose, no commentary.
- Imports first, then test functions named `test_*`.

Imports policy:
- Import only from the target module shown in the prompt.
- `pytest` itself is allowed (use `pytest.raises` for exception assertions).
- Standard-library helpers are allowed only when they are pure data types
  used inside an assertion (e.g. `math.isclose` for float comparison).
- Do not perform file, network, environment, or subprocess access.

Test design:
- Each test asserts one focused behavior; keep tests under ~20 lines.
- Prefer exact return values, exact exception types, and boundary inputs
  (0, 1, -1, empty, max-sized) over loose `isinstance` checks.
- Use the public API exactly as the source defines it. Examples:
    * If the class defines `__contains__`, write `x in obj`, not `obj.contains(x)`.
    * If the class defines `__len__`, write `len(obj)`, not `obj.length()`.
    * Use the exact constructor argument names shown in `__init__`.
- Cover documented edge cases (errors, defaults, branches) — at least one
  test per branch.
"""


# A single class-based example that matches the corpus's shape: constructor
# with kwargs, public methods, a property, an exception type, and the
# `in` operator via __contains__. Function-only examples were poisoning
# small models — they would copy the example verbatim instead of
# generalizing the *style*.
FEW_SHOT_EXAMPLES: list[FewShot] = [
    {
        "prompt": (
            "Module: ratelimit.bucket\n"
            "## Module source\n"
            "```python\n"
            "class RateLimitError(ValueError):\n"
            "    pass\n"
            "\n"
            "class TokenBucket:\n"
            "    '''A capacity-limited bucket of tokens.'''\n"
            "    def __init__(self, capacity: int, refill_per_sec: float = 1.0) -> None:\n"
            "        if capacity <= 0:\n"
            "            raise RateLimitError('capacity must be positive')\n"
            "        if refill_per_sec <= 0:\n"
            "            raise RateLimitError('refill rate must be positive')\n"
            "        self._capacity = capacity\n"
            "        self._refill = float(refill_per_sec)\n"
            "        self._tokens = capacity\n"
            "    @property\n"
            "    def capacity(self) -> int:\n"
            "        return self._capacity\n"
            "    @property\n"
            "    def available(self) -> int:\n"
            "        return self._tokens\n"
            "    def __contains__(self, count: int) -> bool:\n"
            "        return count <= self._tokens\n"
            "    def __len__(self) -> int:\n"
            "        return self._tokens\n"
            "    def consume(self, count: int = 1) -> None:\n"
            "        if count <= 0:\n"
            "            raise RateLimitError('count must be positive')\n"
            "        if count > self._tokens:\n"
            "            raise RateLimitError('insufficient tokens')\n"
            "        self._tokens -= count\n"
            "    def refill(self, count: int) -> None:\n"
            "        self._tokens = min(self._capacity, self._tokens + count)\n"
            "```\n"
            "## Existing tests\n"
            "  - tests/test_bucket.py::test_consume_one\n"
            "  - tests/test_bucket.py::test_capacity_property\n"
        ),
        "completion": (
            "import pytest\n"
            "from ratelimit.bucket import TokenBucket, RateLimitError\n"
            "\n"
            "def test_new_bucket_starts_full():\n"
            "    b = TokenBucket(capacity=5)\n"
            "    assert b.available == 5\n"
            "    assert len(b) == 5\n"
            "\n"
            "def test_in_operator_reflects_availability():\n"
            "    b = TokenBucket(capacity=5)\n"
            "    assert 5 in b\n"
            "    assert 6 not in b\n"
            "\n"
            "def test_consume_reduces_tokens_exactly():\n"
            "    b = TokenBucket(capacity=10)\n"
            "    b.consume(3)\n"
            "    assert b.available == 7\n"
            "\n"
            "def test_consume_default_count_is_one():\n"
            "    b = TokenBucket(capacity=4)\n"
            "    b.consume()\n"
            "    assert b.available == 3\n"
            "\n"
            "def test_consume_zero_raises():\n"
            "    b = TokenBucket(capacity=4)\n"
            "    with pytest.raises(RateLimitError):\n"
            "        b.consume(0)\n"
            "\n"
            "def test_consume_more_than_available_raises():\n"
            "    b = TokenBucket(capacity=2)\n"
            "    with pytest.raises(RateLimitError):\n"
            "        b.consume(3)\n"
            "\n"
            "def test_refill_capped_at_capacity():\n"
            "    b = TokenBucket(capacity=5)\n"
            "    b.consume(2)\n"
            "    b.refill(100)\n"
            "    assert b.available == 5\n"
            "\n"
            "def test_zero_capacity_rejected_at_construction():\n"
            "    with pytest.raises(RateLimitError):\n"
            "        TokenBucket(capacity=0)\n"
            "\n"
            "def test_negative_refill_rate_rejected():\n"
            "    with pytest.raises(RateLimitError):\n"
            "        TokenBucket(capacity=5, refill_per_sec=-1.0)\n"
        ),
    },
    # Second example: one of the actual corpus modules (mini_calendar). The
    # point is to show the policy that when full source is shown, every
    # assertion in the test mirrors a concrete behavior in the source —
    # exact exception type (ValueError), exact return value (a tuple, not a
    # date object), exact bounds (1..9999), exact day-of-week convention
    # (Mon=0). The test passes on this source verbatim.
    {
        "prompt": (
            "Module: mini_calendar.parser\n"
            "## Module source\n"
            "```python\n"
            "from typing import Tuple\n"
            "Date = Tuple[int, int, int]\n"
            "MIN_YEAR = 1\n"
            "MAX_YEAR = 9999\n"
            "_MONTH_DAYS_NORMAL = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)\n"
            "\n"
            "def is_leap_year(year: int) -> bool:\n"
            "    if year % 4 != 0:\n"
            "        return False\n"
            "    if year % 100 != 0:\n"
            "        return True\n"
            "    return year % 400 == 0\n"
            "\n"
            "def days_in_month(year: int, month: int) -> int:\n"
            "    if month < 1 or month > 12:\n"
            "        raise ValueError(f'month must be in 1..12, got {month}')\n"
            "    if month == 2 and is_leap_year(year):\n"
            "        return 29\n"
            "    return _MONTH_DAYS_NORMAL[month - 1]\n"
            "\n"
            "def is_valid_date(year: int, month: int, day: int) -> bool:\n"
            "    if year < MIN_YEAR or year > MAX_YEAR:\n"
            "        return False\n"
            "    if month < 1 or month > 12:\n"
            "        return False\n"
            "    if day < 1 or day > days_in_month(year, month):\n"
            "        return False\n"
            "    return True\n"
            "\n"
            "def parse_iso_date(s: str) -> Date:\n"
            "    if not isinstance(s, str):\n"
            "        raise TypeError(f'parse_iso_date expects str, got {type(s).__name__}')\n"
            "    if len(s) != 10:\n"
            "        raise ValueError(f'expected 10 chars (YYYY-MM-DD), got {len(s)}')\n"
            "    if s[4] != '-' or s[7] != '-':\n"
            "        raise ValueError(f'expected dashes at positions 4 and 7, got {s!r}')\n"
            "    year, month, day = int(s[0:4]), int(s[5:7]), int(s[8:10])\n"
            "    if not is_valid_date(year, month, day):\n"
            "        raise ValueError(f'invalid date: {year:04d}-{month:02d}-{day:02d}')\n"
            "    return (year, month, day)\n"
            "```\n"
            "## Existing tests\n"
            "  - tests/test_parser.py::test_is_leap_year_2000\n"
            "  - tests/test_parser.py::test_parse_iso_date_happy_path\n"
        ),
        "completion": (
            "import pytest\n"
            "from mini_calendar.parser import (\n"
            "    is_leap_year, days_in_month, is_valid_date, parse_iso_date,\n"
            "    MIN_YEAR, MAX_YEAR,\n"
            ")\n"
            "\n"
            "def test_century_not_div_400_is_not_leap():\n"
            "    assert is_leap_year(1900) is False\n"
            "\n"
            "def test_year_div_4_not_div_100_is_leap():\n"
            "    assert is_leap_year(2024) is True\n"
            "\n"
            "def test_days_in_february_leap_is_29():\n"
            "    assert days_in_month(2024, 2) == 29\n"
            "\n"
            "def test_days_in_february_non_leap_is_28():\n"
            "    assert days_in_month(2023, 2) == 28\n"
            "\n"
            "def test_days_in_month_zero_raises_value_error():\n"
            "    with pytest.raises(ValueError):\n"
            "        days_in_month(2024, 0)\n"
            "\n"
            "def test_days_in_month_thirteen_raises_value_error():\n"
            "    with pytest.raises(ValueError):\n"
            "        days_in_month(2024, 13)\n"
            "\n"
            "def test_is_valid_date_year_below_min_is_false():\n"
            "    assert is_valid_date(MIN_YEAR - 1, 1, 1) is False\n"
            "\n"
            "def test_is_valid_date_year_above_max_is_false():\n"
            "    assert is_valid_date(MAX_YEAR + 1, 1, 1) is False\n"
            "\n"
            "def test_parse_iso_date_returns_tuple_not_date_object():\n"
            "    result = parse_iso_date('2024-02-29')\n"
            "    assert result == (2024, 2, 29)\n"
            "    assert isinstance(result, tuple)\n"
            "\n"
            "def test_parse_iso_date_non_string_raises_type_error():\n"
            "    with pytest.raises(TypeError):\n"
            "        parse_iso_date(20240229)\n"
            "\n"
            "def test_parse_iso_date_wrong_length_raises_value_error():\n"
            "    with pytest.raises(ValueError):\n"
            "        parse_iso_date('2024-2-29')\n"
            "\n"
            "def test_parse_iso_date_missing_dash_raises_value_error():\n"
            "    with pytest.raises(ValueError):\n"
            "        parse_iso_date('2024/02/29')\n"
            "\n"
            "def test_parse_iso_date_invalid_calendar_date_raises_value_error():\n"
            "    with pytest.raises(ValueError):\n"
            "        parse_iso_date('2023-02-29')\n"
        ),
    },
]


def render_few_shot(examples: list[FewShot] | None = None) -> str:
    """Render few-shot examples as a single string block.

    Format mirrors the live prompt: prompt body, then a clearly-delimited
    example completion. The 'EXAMPLE N' headers are there so the policy
    learns the boundary between exemplar and the live task it must answer.
    """
    if examples is None:
        examples = FEW_SHOT_EXAMPLES
    chunks: list[str] = []
    for i, ex in enumerate(examples, 1):
        chunks.append(f"### EXAMPLE {i} — INPUT")
        chunks.append(ex["prompt"].rstrip())
        chunks.append(f"### EXAMPLE {i} — OUTPUT")
        chunks.append(ex["completion"].rstrip())
    return "\n\n".join(chunks)


__all__ = ["SYSTEM_PROMPT", "FEW_SHOT_EXAMPLES", "FewShot", "render_few_shot"]
