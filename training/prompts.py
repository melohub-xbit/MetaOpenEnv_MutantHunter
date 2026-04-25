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


SYSTEM_PROMPT = """\
You are MutantHunter, an expert at writing pytest unit tests that catch bugs.

You will be given a Python module summary and the names of its existing
tests. Your job is to write ADDITIONAL pytest tests that catch behaviors
the existing suite misses — the kinds of small, plausible code mutations
(off-by-one, flipped comparison, swapped operator, dropped early-return,
constant changes) that a mutation tester would otherwise leave alive.

Output format (strict):
- Output ONLY the contents of a single pytest file.
- Do NOT include markdown fences, prose, or commentary before/after.
- Imports first, then test functions named ``test_*``.

Rules:
- All your tests MUST pass on the unmodified module (no regressions).
- Do NOT import or use ``os``, ``sys``, ``subprocess``, ``eval``, ``exec``.
- Do NOT touch the filesystem, network, or environment variables.
- Each test should be a focused assertion on one behavior; keep tests
  short (≤ 20 lines).
- Prefer asserting exact return values, exact exception types, and
  boundary inputs (0, 1, -1, empty, max-sized) over loose ``isinstance``
  checks. Loose checks rarely kill mutants.
- When a function has documented edge cases (errors, defaults, branches),
  cover each branch with at least one targeted test.
"""


FEW_SHOT_EXAMPLES: list[FewShot] = [
    {
        "prompt": (
            "Module: shop.discount\n"
            "## Module summary\n"
            "def apply_discount(price: float, pct: float) -> float:\n"
            "    '''Return price reduced by pct percent. Raises ValueError\n"
            "    if pct is outside [0, 100]. price < 0 also raises.'''\n"
            "## Existing tests\n"
            "  - tests/test_discount.py::test_apply_discount_basic\n"
        ),
        "completion": (
            "import pytest\n"
            "from shop.discount import apply_discount\n"
            "\n"
            "def test_zero_percent_returns_input_price():\n"
            "    assert apply_discount(100.0, 0.0) == 100.0\n"
            "\n"
            "def test_full_hundred_percent_returns_zero():\n"
            "    assert apply_discount(50.0, 100.0) == 0.0\n"
            "\n"
            "def test_negative_pct_raises_value_error():\n"
            "    with pytest.raises(ValueError):\n"
            "        apply_discount(10.0, -0.0001)\n"
            "\n"
            "def test_pct_above_one_hundred_raises_value_error():\n"
            "    with pytest.raises(ValueError):\n"
            "        apply_discount(10.0, 100.0001)\n"
            "\n"
            "def test_negative_price_raises_value_error():\n"
            "    with pytest.raises(ValueError):\n"
            "        apply_discount(-1.0, 10.0)\n"
            "\n"
            "def test_half_off_exact_value():\n"
            "    assert apply_discount(80.0, 50.0) == 40.0\n"
        ),
    },
    {
        "prompt": (
            "Module: text.tokens\n"
            "## Module summary\n"
            "def count_words(s: str) -> int:\n"
            "    '''Return number of whitespace-separated tokens. Empty\n"
            "    string and whitespace-only strings return 0.'''\n"
            "## Existing tests\n"
            "  - tests/test_tokens.py::test_count_words_simple_sentence\n"
        ),
        "completion": (
            "from text.tokens import count_words\n"
            "\n"
            "def test_empty_string_is_zero_words():\n"
            "    assert count_words('') == 0\n"
            "\n"
            "def test_whitespace_only_is_zero_words():\n"
            "    assert count_words('   \\t \\n ') == 0\n"
            "\n"
            "def test_single_word_no_padding():\n"
            "    assert count_words('hello') == 1\n"
            "\n"
            "def test_leading_and_trailing_whitespace_ignored():\n"
            "    assert count_words('  one two  ') == 2\n"
            "\n"
            "def test_multiple_internal_spaces_collapse():\n"
            "    assert count_words('a    b    c') == 3\n"
            "\n"
            "def test_tabs_and_newlines_are_separators():\n"
            "    assert count_words('a\\tb\\nc') == 3\n"
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
