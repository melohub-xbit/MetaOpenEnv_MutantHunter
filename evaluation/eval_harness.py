"""Eval harness for the MutantHunter environment.

Runs a callable "policy" (`fn(observation) -> Action | str`) against the
in-process environment for N episodes, recording per-episode reward and
component breakdowns. Also runnable against a remote env server via the
``--server-url`` flag, in which case it uses the ``MutantHunterEnv``
WebSocket client.

The harness is policy-agnostic: pass any callable; for the canonical
"submit a single trivial test" policy, the default ``always_pass_policy``
is used.

Usage
-----

    python -m evaluation.eval_harness --episodes 5
    python -m evaluation.eval_harness --episodes 5 --server-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable, Optional

from mutant_hunter.client import MutantHunterEnv
from mutant_hunter.models import Action, Observation
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment


PolicyFn = Callable[[Observation], Action]
AsyncPolicyFn = Callable[[Observation], Awaitable[Action]]


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    repo: str
    module: str
    baseline_score: float
    reward: float
    components: dict
    new_coverage: float | None
    killed_by_new_only: int | None
    no_regression_gate: float | None
    turns: int


@dataclass
class EvalReport:
    episodes: list[EpisodeResult]

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return statistics.mean(e.reward for e in self.episodes)

    @property
    def median_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return statistics.median(e.reward for e in self.episodes)

    def to_json(self) -> str:
        return json.dumps(
            {
                "mean_reward": self.mean_reward,
                "median_reward": self.median_reward,
                "episodes": [asdict(e) for e in self.episodes],
            },
            indent=2,
        )


# --- Built-in policies -----------------------------------------------------


def always_pass_policy(_obs: Observation) -> Action:
    """Submit one trivial test that passes — minimum no-regression baseline."""
    return Action(kind="submit_tests", test_code="def test_a():\n    assert True\n")


# --- Local in-process runner ----------------------------------------------


def run_local(
    policy: PolicyFn,
    *,
    n_episodes: int,
    seed_start: int = 0,
) -> EvalReport:
    env = MutantHunterEnvironment()
    try:
        out: list[EpisodeResult] = []
        for i in range(n_episodes):
            seed = seed_start + i
            obs = env.reset(seed=seed)
            turns = 0
            done = False
            reward = 0.0
            last_obs = obs
            while not done:
                action = policy(obs)
                obs = env.step(action)
                last_obs = obs
                turns += 1
                done = bool(obs.done)
                if done:
                    reward = float(obs.reward or 0.0)
            md = last_obs.metadata or {}
            out.append(
                EpisodeResult(
                    episode=i,
                    seed=seed,
                    repo=last_obs.repo_name,
                    module=last_obs.module_path,
                    baseline_score=last_obs.baseline_mutation_score,
                    reward=reward,
                    components=md.get("components") or {},
                    new_coverage=md.get("new_coverage"),
                    killed_by_new_only=md.get("killed_by_new_only"),
                    no_regression_gate=md.get("no_regression_gate"),
                    turns=turns,
                )
            )
            print(
                f"  ep{i} seed={seed} repo={last_obs.repo_name} "
                f"module={last_obs.module_path} reward={reward:.3f} "
                f"killed={md.get('killed_by_new_only')}/{md.get('baseline_surviving')}",
                flush=True,
            )
        return EvalReport(episodes=out)
    finally:
        env.close()


# --- Remote (WebSocket) runner --------------------------------------------


async def run_remote(
    policy: PolicyFn | AsyncPolicyFn,
    *,
    n_episodes: int,
    base_url: str,
    seed_start: int = 0,
) -> EvalReport:
    out: list[EpisodeResult] = []
    async with MutantHunterEnv(base_url=base_url) as env:
        for i in range(n_episodes):
            seed = seed_start + i
            result = await env.reset(seed=seed)
            obs = result.observation
            turns = 0
            done = False
            reward = 0.0
            while not done:
                p = policy(obs)
                action = await p if asyncio.iscoroutine(p) else p
                result = await env.step(action)
                obs = result.observation
                turns += 1
                done = bool(result.done)
                if done:
                    reward = float(result.reward or 0.0)
            md = obs.metadata or {}
            out.append(
                EpisodeResult(
                    episode=i,
                    seed=seed,
                    repo=obs.repo_name,
                    module=obs.module_path,
                    baseline_score=obs.baseline_mutation_score,
                    reward=reward,
                    components=md.get("components") or {},
                    new_coverage=md.get("new_coverage"),
                    killed_by_new_only=md.get("killed_by_new_only"),
                    no_regression_gate=md.get("no_regression_gate"),
                    turns=turns,
                )
            )
            print(f"  ep{i} reward={reward:.3f}", flush=True)
    return EvalReport(episodes=out)


# --- Entrypoint ------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--server-url", type=str, default=None,
                    help="If set, drives the server over WebSocket instead of in-process.")
    ap.add_argument("--out", type=str, default=None,
                    help="Write the JSON report here.")
    args = ap.parse_args()

    if args.server_url:
        report = asyncio.run(
            run_remote(
                always_pass_policy,
                n_episodes=args.episodes,
                base_url=args.server_url,
                seed_start=args.seed_start,
            )
        )
    else:
        report = run_local(
            always_pass_policy,
            n_episodes=args.episodes,
            seed_start=args.seed_start,
        )

    js = report.to_json()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
    print(js)
    print(f"\nmean_reward={report.mean_reward:.4f}  median_reward={report.median_reward:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
