"""Standalone probe: does make_reward_fn produce non-zero rewards on
known-good completions?

Builds the env + prompt dataset + reward_fn exactly the way train_grpo.py
does, then for each prompt submits the corpus's *actual existing tests*
for that module as the "model completion". Those tests are known to
import and pass on the unmutated source, so the reward path should
return a non-zero reward (positive format, gate satisfied, plus whatever
mutation-kill the existing suite achieves).

If reward is still 0 across all probes, there is a third bug downstream
of the seed/fence fixes — DO NOT launch a training run.

Usage:
    python training/smoke_reward_fn.py --n 4 --seed-start 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mutant_hunter.corpus import repo_dir  # noqa: E402
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment  # noqa: E402
from training.train_grpo import make_prompt_dataset, make_reward_fn  # noqa: E402


def _existing_tests_concatenated(repo: str) -> str:
    """Concatenate every tests/test_*.py file in the repo into one string.

    This is a known-good completion: the existing suite is what the
    baseline mutation score was computed against, so the env should
    accept it as a valid pytest file (or at minimum, it should import
    and run cleanly)."""
    rd = repo_dir(repo)
    test_root = rd / "tests"
    if not test_root.exists():
        return ""
    parts: list[str] = []
    for p in sorted(test_root.rglob("test_*.py")):
        try:
            parts.append(p.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n\n# ---\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    env = MutantHunterEnvironment()
    train_dataset, prompt_to_seed = make_prompt_dataset(
        env, n=args.n, seed_start=args.seed_start
    )
    reward_fn = make_reward_fn(env, prompt_to_seed, debug_rollouts=args.n)

    prompts = [row["prompt"] for row in train_dataset]
    completions: list[str] = []
    for prompt, seed in prompt_to_seed.items():
        # Re-derive obs to know which repo this prompt was built from.
        obs = env.reset(seed=seed)
        existing = _existing_tests_concatenated(obs.repo_name)
        if not existing.strip():
            existing = "import pytest\n\ndef test_dummy():\n    assert True\n"
        completions.append(existing)
        print(
            f"[smoke] seed={seed} repo={obs.repo_name} module={obs.module_path} "
            f"existing_tests_chars={len(existing)}",
            flush=True,
        )

    print("[smoke] calling reward_fn ...", flush=True)
    rewards = reward_fn(prompts=prompts, completions=completions)

    print()
    print(f"[smoke] rewards = {rewards}")
    nonzero = [r for r in rewards if r > 0.0]
    print(f"[smoke] {len(nonzero)}/{len(rewards)} rollouts produced non-zero reward")
    if not nonzero:
        print("[smoke] FAIL — every reward is 0. There is a third bug. Do NOT train.")
        return 1
    print("[smoke] PASS — reward signal is alive. Safe to launch a short training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
