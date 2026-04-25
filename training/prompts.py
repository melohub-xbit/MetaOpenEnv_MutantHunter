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

You will be given a Python module's source (or summary) and the names of
its existing tests. Your job is to write ADDITIONAL pytest tests that catch
behaviors the existing suite misses — the kinds of small, plausible code
mutations (off-by-one, flipped comparison, swapped operator, dropped
early-return, constant changes) that a mutation tester would otherwise
leave alive.

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
- All your tests MUST pass on the unmodified module.
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
