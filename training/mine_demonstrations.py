"""Mine good/bad pytest demonstrations for in-context demonstration learning.

Reads ``baseline_zeroshot.json`` (locally or from HF Hub) to label each
library by which seeds the zero-shot Qwen2.5-Coder-7B run scored well or
poorly on. The JSON only stores reward metadata, not the generated test
code — so the canonical GOOD source is the corpus's own
``corpus/_local/<lib>/tests/`` file (known to pass on clean code), augmented
with 2-3 hand-written assertions targeting the kinds of mutations the
baseline test suite leaves alive (off-by-one, flipped comparison, branch
collapse). BAD examples are hand-written failure modes (empty file,
``assert True`` only, broken import).

Output: ``training/data/demonstrations.json`` keyed by library, each entry
holding ``{module, good: [...], bad: [...]}`` with raw pytest code per
example. Each GOOD example is validated by running pytest against the
clean source.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mutant_hunter.corpus import LOCAL_LIBS_ROOT  # noqa: E402

DATA_DIR = ROOT / "training" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LIBRARIES = {
    "bloom_filter_lite": "bloom_filter_lite.bloom",
    "mini_calendar": "mini_calendar.parser",
    "csv_normalizer": "csv_normalizer.normalizer",
    "interval_tree": "interval_tree.tree",
}

GOOD_REWARD_THRESHOLD = 0.20

# --- Hand-written mutation-killer suffixes (appended to existing tests) ----

MUTATION_KILLER_SUFFIX: dict[str, str] = {
    "bloom_filter_lite": (
        "\n\n"
        "def test_optimal_parameters_zero_capacity_raises():\n"
        "    import pytest\n"
        "    with pytest.raises(BloomFilterError):\n"
        "        optimal_parameters(0, 0.01)\n"
        "\n"
        "def test_optimal_parameters_error_rate_one_raises():\n"
        "    import pytest\n"
        "    with pytest.raises(BloomFilterError):\n"
        "        optimal_parameters(100, 1.0)\n"
        "\n"
        "def test_no_false_negatives_under_load():\n"
        "    bf = BloomFilter(capacity=200, error_rate=0.01)\n"
        "    items = [f'item-{i}' for i in range(50)]\n"
        "    for x in items:\n"
        "        bf.add(x)\n"
        "    for x in items:\n"
        "        assert x in bf\n"
        "\n"
        "def test_clear_resets_count_and_bits():\n"
        "    bf = BloomFilter(capacity=100, error_rate=0.01)\n"
        "    bf.add('x')\n"
        "    bf.add('y')\n"
        "    bf.clear()\n"
        "    assert len(bf) == 0\n"
        "    assert bf.saturation == 0.0\n"
        "\n"
        "def test_merge_shape_mismatch_raises():\n"
        "    import pytest\n"
        "    a = BloomFilter(capacity=100, error_rate=0.01)\n"
        "    b = BloomFilter(capacity=200, error_rate=0.01)\n"
        "    with pytest.raises(BloomFilterError):\n"
        "        a.merge(b)\n"
    ),
    "mini_calendar": (
        "\n\n"
        "def test_is_leap_year_1900_is_false():\n"
        "    assert is_leap_year(1900) is False\n"
        "\n"
        "def test_is_leap_year_2000_is_true():\n"
        "    assert is_leap_year(2000) is True\n"
        "\n"
        "def test_days_in_february_leap_is_29():\n"
        "    assert days_in_month(2024, 2) == 29\n"
        "\n"
        "def test_days_in_month_zero_raises_value_error():\n"
        "    import pytest\n"
        "    with pytest.raises(ValueError):\n"
        "        days_in_month(2024, 0)\n"
        "\n"
        "def test_days_in_month_thirteen_raises_value_error():\n"
        "    import pytest\n"
        "    with pytest.raises(ValueError):\n"
        "        days_in_month(2024, 13)\n"
        "\n"
        "def test_parse_iso_date_returns_tuple():\n"
        "    result = parse_iso_date('2024-02-29')\n"
        "    assert result == (2024, 2, 29)\n"
        "    assert isinstance(result, tuple)\n"
    ),
    "csv_normalizer": (
        "\n\n"
        "def test_normalize_header_collapses_runs():\n"
        "    assert normalize_header('A  B__C') == 'a_b_c'\n"
        "\n"
        "def test_normalize_header_strips_trailing_underscore():\n"
        "    assert normalize_header('hello!') == 'hello'\n"
        "\n"
        "def test_normalize_header_empty_returns_col():\n"
        "    assert normalize_header('!!!') == 'col'\n"
        "\n"
        "def test_coerce_value_false_literal():\n"
        "    assert coerce_value('false') is False\n"
        "\n"
        "def test_coerce_value_no_alias_is_false():\n"
        "    assert coerce_value('no') is False\n"
        "\n"
        "def test_is_numeric_negative_int():\n"
        "    assert is_numeric('-7') is True\n"
        "\n"
        "def test_parse_row_quoted_with_comma():\n"
        "    assert parse_row('\"a,b\",c') == ['a,b', 'c']\n"
    ),
    "interval_tree": (
        "\n\n"
        "def test_query_point_at_end_excluded():\n"
        "    t = IntervalTree()\n"
        "    t.add(1, 5, payload='A')\n"
        "    assert t.query_point(5) == []\n"
        "\n"
        "def test_query_point_at_start_included():\n"
        "    t = IntervalTree()\n"
        "    t.add(1, 5, payload='A')\n"
        "    assert t.query_point(1) == [(1, 5, 'A')]\n"
        "\n"
        "def test_touching_intervals_do_not_overlap():\n"
        "    t = IntervalTree()\n"
        "    t.add(1, 3)\n"
        "    assert t.overlaps(3, 5) is False\n"
        "\n"
        "def test_zero_width_interval_rejected():\n"
        "    import pytest\n"
        "    from interval_tree.tree import IntervalError\n"
        "    t = IntervalTree()\n"
        "    with pytest.raises(IntervalError):\n"
        "        t.add(5, 5)\n"
        "\n"
        "def test_start_after_end_rejected():\n"
        "    import pytest\n"
        "    from interval_tree.tree import IntervalError\n"
        "    t = IntervalTree()\n"
        "    with pytest.raises(IntervalError):\n"
        "        t.add(7, 3)\n"
    ),
}


# --- Hand-written BAD examples --------------------------------------------

BAD_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "bloom_filter_lite": [
        {
            "name": "empty_test",
            "code": (
                "from bloom_filter_lite.bloom import BloomFilter\n"
                "\n"
                "def test_nothing():\n"
                "    pass\n"
            ),
        },
        {
            "name": "vacuous_assert",
            "code": (
                "from bloom_filter_lite.bloom import BloomFilter\n"
                "\n"
                "def test_bloom_works():\n"
                "    assert True\n"
                "\n"
                "def test_bloom_again():\n"
                "    bf = BloomFilter(capacity=100, error_rate=0.01)\n"
                "    assert bf is not None\n"
            ),
        },
        {
            "name": "broken_import",
            "code": (
                "from bloom_filter_lite.bloom import NonExistentClass\n"
                "\n"
                "def test_broken():\n"
                "    obj = NonExistentClass()\n"
                "    assert obj is not None\n"
            ),
        },
    ],
    "mini_calendar": [
        {
            "name": "empty_test",
            "code": (
                "from mini_calendar.parser import is_leap_year\n"
                "\n"
                "def test_nothing():\n"
                "    pass\n"
            ),
        },
        {
            "name": "wrong_exception_type",
            "code": (
                "import pytest\n"
                "from mini_calendar.parser import days_in_month\n"
                "\n"
                "def test_days_in_month_bad_month_raises_runtime_error():\n"
                "    with pytest.raises(RuntimeError):\n"
                "        days_in_month(2024, 0)\n"
            ),
        },
        {
            "name": "vacuous_assert",
            "code": (
                "from mini_calendar.parser import is_leap_year\n"
                "\n"
                "def test_leap_year_returns_something():\n"
                "    result = is_leap_year(2024)\n"
                "    assert result is not None\n"
            ),
        },
    ],
    "csv_normalizer": [
        {
            "name": "broken_import",
            "code": (
                "from csv_normalizer.normalizer import does_not_exist\n"
                "\n"
                "def test_broken():\n"
                "    assert does_not_exist('x') == 'x'\n"
            ),
        },
        {
            "name": "vacuous_assert",
            "code": (
                "from csv_normalizer.normalizer import parse_row\n"
                "\n"
                "def test_parse_row_returns_truthy():\n"
                "    result = parse_row('a,b,c')\n"
                "    assert result\n"
            ),
        },
        {
            "name": "wrong_expected_value",
            "code": (
                "from csv_normalizer.normalizer import normalize_header\n"
                "\n"
                "def test_normalize_header_keeps_spaces():\n"
                "    assert normalize_header('First Name') == 'First Name'\n"
            ),
        },
    ],
    "interval_tree": [
        {
            "name": "empty_test",
            "code": (
                "from interval_tree.tree import IntervalTree\n"
                "\n"
                "def test_nothing():\n"
                "    pass\n"
            ),
        },
        {
            "name": "vacuous_assert",
            "code": (
                "from interval_tree.tree import IntervalTree\n"
                "\n"
                "def test_tree_constructible():\n"
                "    t = IntervalTree()\n"
                "    assert t is not None\n"
            ),
        },
        {
            "name": "wrong_boundary_assumption",
            "code": (
                "from interval_tree.tree import IntervalTree\n"
                "\n"
                "def test_query_point_at_end_included():\n"
                "    t = IntervalTree()\n"
                "    t.add(1, 5, payload='A')\n"
                "    # WRONG: half-open [1, 5) does NOT contain 5.\n"
                "    assert t.query_point(5) == [(1, 5, 'A')]\n"
            ),
        },
    ],
}


# --- Loading baseline_zeroshot.json ---------------------------------------


def load_baseline(local_path: Path | None) -> dict | None:
    """Return the parsed baseline_zeroshot.json. Try local first, then HF Hub."""
    if local_path and local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))
    fallback_local = ROOT / "final_results" / "baseline_zeroshot.json"
    if fallback_local.exists():
        print(f"[mine] using local fallback: {fallback_local}")
        return json.loads(fallback_local.read_text(encoding="utf-8"))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[mine] huggingface_hub not installed and no local baseline; "
              "continuing with empty zero-shot stats.")
        return None
    try:
        path = hf_hub_download(
            repo_id="jester1177/mutant-hunter-results",
            filename="baseline_zeroshot.json",
            repo_type="dataset",
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[mine] HF Hub fetch failed: {exc}; continuing without baseline data.")
        return None


def label_episodes(baseline: dict | None) -> dict[str, dict[str, list[int]]]:
    """Group episode seeds by library into good (>= 0.20) / bad (=0 + gate=0)."""
    out: dict[str, dict[str, list[int]]] = {
        lib: {"good": [], "bad": []} for lib in LIBRARIES
    }
    if not baseline:
        return out
    for ep in baseline.get("episodes", []):
        repo = ep.get("repo")
        if repo not in out:
            continue
        reward = float(ep.get("final_reward") or 0.0)
        gate = ep.get("no_regression_gate")
        seed = int(ep.get("seed", -1))
        if reward >= GOOD_REWARD_THRESHOLD:
            out[repo]["good"].append(seed)
        elif reward == 0.0 and (gate == 0 or gate == 0.0):
            out[repo]["bad"].append(seed)
    return out


# --- Validation: run pytest against clean source ---------------------------


def _existing_test_path(repo: str) -> Path:
    test_dir = LOCAL_LIBS_ROOT / repo / "tests"
    candidates = sorted(test_dir.glob("test_*.py"))
    if not candidates:
        raise FileNotFoundError(f"no test_*.py in {test_dir}")
    return candidates[0]


def _validate_good_passes(repo: str, code: str) -> tuple[bool, str]:
    """Run the candidate test code against the clean repo via pytest in a
    temp workspace. Returns (passed, stdout_tail)."""
    repo_root = LOCAL_LIBS_ROOT / repo
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Mirror the package layout the repo expects: copy package source
        # plus an empty tests/ dir, then drop the candidate file in.
        shutil.copytree(repo_root, tmp / repo,
                        ignore=shutil.ignore_patterns("__pycache__",
                                                       ".pytest_cache",
                                                       "*.egg-info"))
        # Wipe the existing tests dir so we only run the candidate file.
        candidate_tests_dir = tmp / repo / "tests"
        if candidate_tests_dir.exists():
            for item in candidate_tests_dir.iterdir():
                if item.name == "__init__.py":
                    continue
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item, ignore_errors=True)
        candidate_file = candidate_tests_dir / "test_demo_candidate.py"
        candidate_file.write_text(code, encoding="utf-8")
        # Run pytest from the package parent so `from <repo>.module import ...` works.
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-x",
             str(candidate_file)],
            cwd=str(tmp),
            capture_output=True,
            text=True,
            timeout=60,
        )
        ok = proc.returncode == 0
        tail = (proc.stdout + proc.stderr)[-600:]
        return ok, tail


# --- Demo construction ----------------------------------------------------


_TOP_LEVEL_IMPORTS_PER_REPO: dict[str, str] = {
    "bloom_filter_lite": (
        "from bloom_filter_lite.bloom import (\n"
        "    BloomFilter, BloomFilterError, optimal_parameters,\n"
        ")\n"
    ),
    "mini_calendar": (
        "from mini_calendar.parser import (\n"
        "    is_leap_year, days_in_month, parse_iso_date,\n"
        ")\n"
    ),
    "csv_normalizer": (
        "from csv_normalizer.normalizer import (\n"
        "    coerce_value, is_numeric, normalize_header, parse_row,\n"
        ")\n"
    ),
    "interval_tree": (
        "from interval_tree.tree import IntervalTree\n"
    ),
}


def _trim_existing_suite(text: str, max_chars: int = 1400) -> str:
    """Keep the leading docstring/imports + as many full test functions as
    fit in ``max_chars``. We split on blank-line boundaries between top-level
    `def test_*` blocks so we never truncate mid-function."""
    if len(text) <= max_chars:
        return text
    lines = text.splitlines(keepends=True)
    # Find prelude: everything up to the first `def test_`.
    prelude_end = 0
    for i, ln in enumerate(lines):
        if ln.startswith("def test_"):
            prelude_end = i
            break
    prelude = "".join(lines[:prelude_end])
    body_lines = lines[prelude_end:]
    # Group body into test-function blocks (each starts with `def test_`).
    blocks: list[str] = []
    current: list[str] = []
    for ln in body_lines:
        if ln.startswith("def test_") and current:
            blocks.append("".join(current))
            current = [ln]
        else:
            current.append(ln)
    if current:
        blocks.append("".join(current))
    out = prelude
    for block in blocks:
        if len(out) + len(block) > max_chars:
            break
        out += block
    return out.rstrip() + "\n"


def build_good_examples(repo: str) -> list[dict[str, str]]:
    """Two GOOD examples per repo, each well under the 2.4KB GOOD budget:
    1. ``mutation_killers_only`` — just the hand-written assertions targeting
       common mutations (compact, ~700-1100 chars).
    2. ``existing_suite_trimmed`` — first few tests from the corpus's own
       known-passing suite, capped at ~1400 chars."""
    base = _existing_test_path(repo).read_text(encoding="utf-8")
    imports = _TOP_LEVEL_IMPORTS_PER_REPO[repo]
    killers_only = imports + MUTATION_KILLER_SUFFIX[repo].lstrip("\n")
    trimmed = _trim_existing_suite(base, max_chars=1400)
    return [
        {"name": "mutation_killers_only", "code": killers_only},
        {"name": "existing_suite_trimmed", "code": trimmed},
    ]


def build_bad_examples(repo: str) -> list[dict[str, str]]:
    return list(BAD_EXAMPLES[repo])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-json", type=str, default=None,
                    help="Local path to baseline_zeroshot.json. If omitted, "
                         "tries HF Hub then ./final_results/.")
    ap.add_argument("--out", type=str,
                    default=str(DATA_DIR / "demonstrations.json"))
    ap.add_argument("--skip-validate", action="store_true",
                    help="Skip running pytest on each GOOD example.")
    args = ap.parse_args()

    baseline = load_baseline(Path(args.baseline_json) if args.baseline_json else None)
    labels = label_episodes(baseline)

    demos: dict[str, dict] = {}
    for lib, module in LIBRARIES.items():
        good = build_good_examples(lib)
        bad = build_bad_examples(lib)

        if not args.skip_validate:
            validated_good: list[dict[str, str]] = []
            for ex in good:
                ok, tail = _validate_good_passes(lib, ex["code"])
                status = "PASS" if ok else "FAIL"
                print(f"[mine] {lib} GOOD/{ex['name']}: {status}")
                if not ok:
                    print(f"  pytest tail: {tail.strip()[:300]}")
                if ok:
                    validated_good.append(ex)
            if not validated_good:
                print(f"[mine] WARNING: {lib} has no validated good examples; "
                      "keeping unvalidated set so the pipeline still produces output.")
                validated_good = good
            good = validated_good

        demos[lib] = {
            "module": module,
            "zero_shot_good_seeds": labels[lib]["good"],
            "zero_shot_bad_seeds": labels[lib]["bad"],
            "good": good,
            "bad": bad,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(demos, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"[mine] wrote {out_path}")
    for lib, entry in demos.items():
        print(f"  {lib}: good={len(entry['good'])} bad={len(entry['bad'])} "
              f"zs_good_seeds={entry['zero_shot_good_seeds']} "
              f"zs_bad_seeds={entry['zero_shot_bad_seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
