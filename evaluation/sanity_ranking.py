"""Layer 1/2/3/4 reward-sanity validation.

Submits a fixed set of hand-crafted policies against the env and prints
a per-policy reward + component breakdown table sorted high→low.

Layer 1 (default subcommand: ``layer1``)
    Differential reward sanity on mini_calendar.parser.

Layer 2 (subcommand: ``layer2``)
    Determinism check — runs single_strong three times with the same seed
    and asserts byte-identical components/final reward.

Layer 4 (subcommand: ``layer4``)
    Cross-module variance — runs a comprehensive policy on every module
    in the manifest and prints rewards side by side.

Each subcommand writes its raw results to ``evaluation/_results/``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mutant_hunter.corpus import MANIFEST_PATH  # noqa: E402
from mutant_hunter.models import Action  # noqa: E402
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Hand-crafted policies                                                       #
# --------------------------------------------------------------------------- #

# NOTE: the actual mini_calendar.parser function is `parse_iso_date` (raises
# ValueError on invalid input, returns a (year, month, day) tuple). We adapt
# the user-spec'd "parse_date" policies to that real signature.

POLICIES_MINI_CALENDAR: dict[str, str] = {
    "empty": "",
    "vacuous": (
        "def test_vacuous():\n"
        "    assert True\n"
    ),
    "regression": (
        "def test_will_fail_on_clean_code():\n"
        "    assert 1 == 2\n"
    ),
    "single_weak": (
        # The existing test suite never calls `from_ordinal`. Picking a date
        # that requires the leap-year-2000 branch (366 valid only when leap)
        # means mutants that flip is_leap_year(2000) to False cause this
        # function to raise — caught even by an `is not None` assertion.
        "from mini_calendar.parser import from_ordinal\n"
        "\n"
        "def test_from_ordinal_returns_value():\n"
        "    assert from_ordinal(2000, 366) is not None\n"
    ),
    "single_strong": (
        "from mini_calendar.parser import parse_iso_date\n"
        "\n"
        "def test_parse_three_dates():\n"
        "    assert parse_iso_date('2024-06-15') == (2024, 6, 15)\n"
        "    assert parse_iso_date('2000-02-29') == (2000, 2, 29)\n"
        "    assert parse_iso_date('1999-12-31') == (1999, 12, 31)\n"
    ),
    "comprehensive": (
        "import pytest\n"
        "from mini_calendar.parser import parse_iso_date\n"
        "\n"
        "def test_parse_iso_format_basic():\n"
        "    # ISO 8601 calendar form.\n"
        "    assert parse_iso_date('2024-06-15') == (2024, 6, 15)\n"
        "\n"
        "def test_parse_leap_year_feb29():\n"
        "    assert parse_iso_date('2000-02-29') == (2000, 2, 29)\n"
        "\n"
        "def test_parse_boundary_year_1900_non_leap():\n"
        "    # 1900 is divisible by 100 but not 400, so Feb 29 is invalid.\n"
        "    with pytest.raises(ValueError):\n"
        "        parse_iso_date('1900-02-29')\n"
        "    # ...but Feb 28 is fine.\n"
        "    assert parse_iso_date('1900-02-28') == (1900, 2, 28)\n"
        "\n"
        "def test_parse_invalid_day_raises():\n"
        "    with pytest.raises(ValueError):\n"
        "        parse_iso_date('2024-04-31')\n"
        "\n"
        "def test_parse_invalid_month_raises():\n"
        "    with pytest.raises(ValueError):\n"
        "        parse_iso_date('2024-13-01')\n"
        "\n"
        "def test_parse_off_by_one_day_jan31():\n"
        "    # 2024-01-31 is the last day of January; 2024-01-32 must fail.\n"
        "    assert parse_iso_date('2024-01-31') == (2024, 1, 31)\n"
        "    with pytest.raises(ValueError):\n"
        "        parse_iso_date('2024-01-32')\n"
    ),
}

# Layer 4 — a comprehensive-quality policy per repo, 4-6 tests each.
POLICIES_BY_REPO: dict[str, str] = {
    "mini_calendar": POLICIES_MINI_CALENDAR["comprehensive"],
    "csv_normalizer": (
        "import pytest\n"
        "from csv_normalizer.normalizer import (\n"
        "    normalize_header, parse_row, parse_csv, is_numeric, coerce_value, NormalizationError,\n"
        ")\n"
        "\n"
        "def test_normalize_header_basic_snake_case():\n"
        "    assert normalize_header('Hello World') == 'hello_world'\n"
        "    assert normalize_header('  --foo--  ') == 'foo'\n"
        "    assert normalize_header('') == 'col'\n"
        "\n"
        "def test_parse_row_with_quoted_field_and_escaped_quote():\n"
        "    assert parse_row('a,\"b,c\",d') == ['a', 'b,c', 'd']\n"
        "    assert parse_row('a,\"he said \"\"hi\"\"\",b') == ['a', 'he said \"hi\"', 'b']\n"
        "\n"
        "def test_parse_row_unterminated_quote_raises():\n"
        "    with pytest.raises(NormalizationError):\n"
        "        parse_row('a,\"b,c')\n"
        "\n"
        "def test_is_numeric_and_coerce_value():\n"
        "    assert is_numeric('42') is True\n"
        "    assert is_numeric('3.14') is True\n"
        "    assert is_numeric('NaN') is False\n"
        "    assert is_numeric('') is False\n"
        "    assert coerce_value('42') == 42\n"
        "    assert coerce_value('3.14') == 3.14\n"
        "    assert coerce_value('TRUE') is True\n"
        "    assert coerce_value('false') is False\n"
        "\n"
        "def test_parse_csv_basic_with_header_dedup():\n"
        "    text = 'Name,Age,Name\\nAlice,30,X\\nBob,25,Y\\n'\n"
        "    rows = parse_csv(text)\n"
        "    assert len(rows) == 2\n"
        "    assert rows[0]['name'] == 'Alice'\n"
        "    assert rows[0]['age'] == 30\n"
        "    assert rows[0]['name_2'] == 'X'\n"
        "\n"
        "def test_parse_csv_too_wide_row_raises():\n"
        "    text = 'a,b\\n1,2,3\\n'\n"
        "    with pytest.raises(NormalizationError):\n"
        "        parse_csv(text)\n"
    ),
    "interval_tree": (
        "import pytest\n"
        "from interval_tree.tree import IntervalTree, IntervalError\n"
        "\n"
        "def test_add_and_overlaps_basic():\n"
        "    t = IntervalTree()\n"
        "    t.add(1, 5, 'a')\n"
        "    t.add(10, 20, 'b')\n"
        "    assert t.overlaps(4, 6) is True\n"
        "    assert t.overlaps(5, 10) is False  # touching is not overlapping\n"
        "    assert t.overlaps(0, 1) is False\n"
        "\n"
        "def test_query_point_half_open_semantics():\n"
        "    t = IntervalTree()\n"
        "    t.add(1, 5, 'a')\n"
        "    assert t.query_point(1) == [(1, 5, 'a')]\n"
        "    assert t.query_point(5) == []  # exclusive upper bound\n"
        "    assert t.query_point(0) == []\n"
        "\n"
        "def test_query_range_returns_overlapping_only():\n"
        "    t = IntervalTree()\n"
        "    t.add(1, 5, 'a')\n"
        "    t.add(10, 20, 'b')\n"
        "    assert t.query_range(4, 11) == [(1, 5, 'a'), (10, 20, 'b')]\n"
        "    assert t.query_range(5, 10) == []\n"
        "\n"
        "def test_invalid_interval_raises():\n"
        "    t = IntervalTree()\n"
        "    with pytest.raises(IntervalError):\n"
        "        t.add(5, 1)\n"
        "\n"
        "def test_remove_returns_bool_and_total_length():\n"
        "    t = IntervalTree()\n"
        "    t.add(1, 5, 'a')\n"
        "    t.add(10, 20, 'b')\n"
        "    assert t.total_length() == 14\n"
        "    assert t.remove(1, 5, 'a') is True\n"
        "    assert t.remove(1, 5, 'a') is False\n"
        "    assert len(t) == 1\n"
        "\n"
        "def test_merge_overlapping_collapses():\n"
        "    t = IntervalTree()\n"
        "    t.add(1, 5)\n"
        "    t.add(4, 8)\n"
        "    t.add(10, 12)\n"
        "    removed = t.merge_overlapping()\n"
        "    assert removed == 1\n"
        "    spans = sorted((s, e) for s, e, _ in t)\n"
        "    assert spans == [(1, 8), (10, 12)]\n"
    ),
    "bloom_filter_lite": (
        "import pytest\n"
        "from bloom_filter_lite.bloom import BloomFilter, BloomFilterError, optimal_parameters\n"
        "\n"
        "def test_optimal_parameters_basic_shape():\n"
        "    m, k = optimal_parameters(100, 0.01)\n"
        "    assert m >= 1 and k >= 1\n"
        "    assert isinstance(m, int) and isinstance(k, int)\n"
        "\n"
        "def test_optimal_parameters_invalid_inputs_raise():\n"
        "    with pytest.raises(BloomFilterError):\n"
        "        optimal_parameters(0, 0.01)\n"
        "    with pytest.raises(BloomFilterError):\n"
        "        optimal_parameters(10, 0.0)\n"
        "    with pytest.raises(BloomFilterError):\n"
        "        optimal_parameters(10, 1.0)\n"
        "    with pytest.raises(BloomFilterError):\n"
        "        optimal_parameters(True, 0.01)  # bool rejected\n"
        "\n"
        "def test_add_and_contains_no_false_negative():\n"
        "    bf = BloomFilter(capacity=64, error_rate=0.01)\n"
        "    items = ['alpha', 'beta', 'gamma', b'\\x00\\x01', 42, -7, True, False]\n"
        "    for x in items:\n"
        "        bf.add(x)\n"
        "    for x in items:\n"
        "        assert x in bf\n"
        "    assert len(bf) == len(items)\n"
        "\n"
        "def test_clear_resets_count_and_bits():\n"
        "    bf = BloomFilter(capacity=16, error_rate=0.05)\n"
        "    bf.add('x')\n"
        "    bf.add('y')\n"
        "    assert len(bf) == 2\n"
        "    bf.clear()\n"
        "    assert len(bf) == 0\n"
        "    assert bf.saturation == 0.0\n"
        "\n"
        "def test_merge_requires_shape_match():\n"
        "    a = BloomFilter(capacity=64, error_rate=0.01)\n"
        "    b = BloomFilter(capacity=64, error_rate=0.01)\n"
        "    a.add('x')\n"
        "    b.add('y')\n"
        "    merged = a.merge(b)\n"
        "    assert 'x' in merged and 'y' in merged\n"
        "    other = BloomFilter(capacity=128, error_rate=0.01)\n"
        "    with pytest.raises(BloomFilterError):\n"
        "        a.merge(other)\n"
    ),
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def find_seed_for_repo(target_repo: str, search_max: int = 500) -> int:
    """Find the smallest integer seed such that env._pick_module() lands on target_repo."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    local_repos = [r for r in manifest["repos"] if r.get("source") == "local"]
    for s in range(search_max):
        rng = random.Random(s)
        r = rng.choice(local_repos)
        if r["name"] == target_repo:
            return s
    raise RuntimeError(f"no seed in [0, {search_max}) selects repo={target_repo}")


def submit_policy(env: MutantHunterEnvironment, seed: int, test_code: str) -> dict:
    """Reset, attempt to submit. Returns a flat row dict of metrics."""
    obs = env.reset(seed=seed)
    repo = obs.repo_name
    module = obs.module_path
    baseline_surviving = len(env._baseline.surviving_mutants)
    baseline_cov = env._baseline.coverage_baseline

    # Empty policy is rejected at the Action level.
    try:
        action = Action(kind="submit_tests", test_code=test_code)
    except Exception as exc:
        return {
            "repo": repo,
            "module": module,
            "baseline_surviving": baseline_surviving,
            "baseline_coverage": baseline_cov,
            "final_reward": 0.0,
            "components": {
                "mutation_kill": 0.0,
                "coverage_delta": 0.0,
                "format": 0.0,
                "parsimony": 0.0,
            },
            "no_regression_gate": 0.0,
            "killed_by_new_only": 0,
            "new_coverage": baseline_cov,
            "new_tests_pass_clean": False,
            "rejected_at_action": True,
            "action_error": f"{type(exc).__name__}: {exc}",
        }

    t0 = time.time()
    obs2 = env.step(action)
    elapsed = time.time() - t0
    md = obs2.metadata or {}
    return {
        "repo": repo,
        "module": module,
        "baseline_surviving": md.get("baseline_surviving", baseline_surviving),
        "baseline_coverage": baseline_cov,
        "final_reward": float(obs2.reward or 0.0),
        "components": dict(md.get("components") or {}),
        "no_regression_gate": md.get("no_regression_gate"),
        "killed_by_new_only": md.get("killed_by_new_only"),
        "new_coverage": md.get("new_coverage"),
        "new_tests_pass_clean": md.get("new_tests_pass_clean"),
        "elapsed_s": round(elapsed, 2),
        "status": md.get("status"),
    }


def _row_str(name: str, row: dict) -> str:
    c = row["components"]
    gate = row["no_regression_gate"]
    return (
        f"{name:<14}"
        f" reward={row['final_reward']:.4f}"
        f" kill={c.get('mutation_kill', 0.0):.4f}"
        f" cov_d={c.get('coverage_delta', 0.0):.4f}"
        f" fmt={c.get('format', 0.0):.2f}"
        f" pars={c.get('parsimony', 0.0):.2f}"
        f" gate={gate}"
        f" k/{row.get('baseline_surviving')}={row.get('killed_by_new_only')}"
        f" t={row.get('elapsed_s', '?')}s"
    )


# --------------------------------------------------------------------------- #
# Layer 1                                                                     #
# --------------------------------------------------------------------------- #


def run_layer1() -> int:
    seed = find_seed_for_repo("mini_calendar")
    print(f"[layer1] target=mini_calendar.parser  seed={seed}", flush=True)

    env = MutantHunterEnvironment()
    rows: dict[str, dict] = {}
    try:
        for name, code in POLICIES_MINI_CALENDAR.items():
            print(f"  running policy={name} ...", flush=True)
            row = submit_policy(env, seed=seed, test_code=code)
            rows[name] = row
            print("    " + _row_str(name, row), flush=True)
    finally:
        env.close()

    out_path = RESULTS_DIR / "layer1_sanity_ranking.json"
    out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    # Sorted print.
    print("\n[layer1] sorted by final_reward (high to low):")
    sorted_items = sorted(rows.items(), key=lambda kv: kv[1]["final_reward"], reverse=True)
    for name, row in sorted_items:
        print("  " + _row_str(name, row))

    # Required ordering: comprehensive > single_strong > single_weak > vacuous > empty ~= regression ~= 0
    r = {n: rows[n]["final_reward"] for n in rows}
    violations: list[str] = []
    if not (r["comprehensive"] > r["single_strong"]):
        violations.append(f"comprehensive ({r['comprehensive']:.4f}) NOT > single_strong ({r['single_strong']:.4f})")
    if not (r["single_strong"] > r["single_weak"]):
        violations.append(f"single_strong ({r['single_strong']:.4f}) NOT > single_weak ({r['single_weak']:.4f})")
    if not (r["single_weak"] > r["vacuous"]):
        violations.append(f"single_weak ({r['single_weak']:.4f}) NOT > vacuous ({r['single_weak']:.4f})")
    if not (r["vacuous"] > max(r["empty"], r["regression"]) - 1e-9):
        violations.append(f"vacuous ({r['vacuous']:.4f}) NOT > max(empty, regression)")
    if not (abs(r["empty"]) < 1e-6):
        violations.append(f"empty ({r['empty']:.4f}) NOT ~= 0")
    if not (abs(r["regression"]) < 1e-6):
        violations.append(f"regression ({r['regression']:.4f}) NOT ~= 0")

    print()
    if violations:
        print("[layer1] FAIL — ordering violations:")
        for v in violations:
            print(f"  - {v}")
        print("\nDo NOT proceed to Layer 2. Reward function is broken.")
        return 1
    print("[layer1] PASS — ordering: comprehensive > single_strong > single_weak > vacuous > empty ~= regression ~= 0")
    return 0


# --------------------------------------------------------------------------- #
# Layer 2                                                                     #
# --------------------------------------------------------------------------- #


def run_layer2() -> int:
    seed = find_seed_for_repo("mini_calendar")
    code = POLICIES_MINI_CALENDAR["single_strong"]
    print(f"[layer2] target=mini_calendar.parser  seed={seed}  policy=single_strong", flush=True)

    env = MutantHunterEnvironment()
    runs: list[dict] = []
    try:
        for i in range(3):
            print(f"  run {i + 1}/3 ...", flush=True)
            row = submit_policy(env, seed=seed, test_code=code)
            runs.append(row)
            print("    " + _row_str(f"run{i+1}", row), flush=True)
    finally:
        env.close()

    out_path = RESULTS_DIR / "layer2_determinism.json"
    out_path.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")

    # Compare components and final_reward exactly.
    fields = ["final_reward", "killed_by_new_only", "new_coverage", "no_regression_gate"]
    diffs: list[str] = []
    base = runs[0]
    for k in fields:
        vals = [r[k] for r in runs]
        if any(v != base[k] for v in vals):
            diffs.append(f"{k} varies: {vals}")
    base_c = base["components"]
    for ck in base_c:
        vals = [r["components"].get(ck) for r in runs]
        if any(v != base_c[ck] for v in vals):
            diffs.append(f"components.{ck} varies: {vals}")

    print()
    if diffs:
        print("[layer2] FAIL — non-determinism detected:")
        for d in diffs:
            print(f"  - {d}")
        return 1
    print("[layer2] PASS — all 3 runs produced byte-identical components/reward.")
    return 0


# --------------------------------------------------------------------------- #
# Layer 4                                                                     #
# --------------------------------------------------------------------------- #


def run_layer4() -> int:
    print("[layer4] cross-module variance — running comprehensive policy on each repo", flush=True)
    rows: dict[str, dict] = {}
    env = MutantHunterEnvironment()
    try:
        for repo, code in POLICIES_BY_REPO.items():
            seed = find_seed_for_repo(repo)
            print(f"  repo={repo} seed={seed} ...", flush=True)
            row = submit_policy(env, seed=seed, test_code=code)
            rows[repo] = row
            print("    " + _row_str(repo, row), flush=True)
    finally:
        env.close()

    out_path = RESULTS_DIR / "layer4_cross_module.json"
    out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print("\n[layer4] side-by-side rewards:")
    for repo, row in rows.items():
        print("  " + _row_str(repo, row))

    rewards = [row["final_reward"] for row in rows.values()]
    spread = max(rewards) - min(rewards)
    print(f"\n[layer4] reward spread = {spread:.4f}  (acceptable <= 0.7)")
    if spread > 0.7:
        outlier = max(rows.items(), key=lambda kv: abs(kv[1]["final_reward"] - sum(rewards) / len(rewards)))
        print(f"[layer4] FAIL — spread > 0.7  outlier={outlier[0]} ({outlier[1]['final_reward']:.4f})")
        return 1
    print("[layer4] PASS — spread within tolerance.")
    return 0


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", choices=["layer1", "layer2", "layer4", "all"], default="layer1", nargs="?")
    args = ap.parse_args()

    layers: dict[str, Callable[[], int]] = {
        "layer1": run_layer1,
        "layer2": run_layer2,
        "layer4": run_layer4,
    }
    if args.layer == "all":
        rc = 0
        for name in ("layer1", "layer2", "layer4"):
            print(f"\n{'='*72}\n  {name}\n{'='*72}", flush=True)
            rc |= layers[name]()
        return rc
    return layers[args.layer]()


if __name__ == "__main__":
    raise SystemExit(main())
