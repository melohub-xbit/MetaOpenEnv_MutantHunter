---
title: MutantHunter Train
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
---

# MutantHunter

An RL environment that teaches LLMs to write tests that actually catch bugs.

> Links (HF Space, write-up, demo) coming soon.

---

## The problem

How do you know if a test suite is any good?

Coverage tells you which lines ran. It doesn't tell you whether the tests
would *notice* if those lines were wrong. Most production test suites have
high coverage and low value: they call the code, then assert something so
weak that the assertion would still hold if the code were broken.

LLMs are now writing a lot of those tests. They get rewarded — by humans,
by linters, by coverage gates — for tests that *pass*, not for tests that
*detect bugs*. So they learn to write tests that pass. The capability gap
is narrow but load-bearing: we want a model that writes tests a real
adversary would have to work to fool.

## The trick: mutation testing as a reward signal

Mutation testing flips the question around. Instead of asking "did the
test run the code," it asks: **if I deliberately break the code in a
small way, will the tests notice?**

A toy example. Source:

```python
def add(a, b):
    return a + b
```

A developer writes one test:

```python
def test_add():
    assert add(2, 2) == 4
```

This passes. Coverage is 100%. Looks great. Now break the source:

- Change `+` to `-` → `add(2, 2)` returns `0`, test fails. **Mutant killed.** Good.
- Change `+` to `*` → `add(2, 2)` returns `4`, test passes. **Mutant survived.** Bad test.
- Change `+` to `**` → `add(2, 2)` returns `4`, test passes. **Mutant survived.** Bad test.

One out of three. Mutation score: 33%. The 100%-coverage test suite is
mostly useless. A stronger suite covers more inputs:

```python
def test_add():
    assert add(2, 2) == 4
    assert add(2, 3) == 5     # kills the * mutant
    assert add(0, 0) == 0     # kills the ** mutant (0**0 == 1 in Python)
    assert add(-1, 1) == 0
    assert add(100, -50) == 50
```

Mutation score climbs to 100%. Same coverage, much better test.

That's the whole reward signal. Better tests → more mutants killed →
higher reward. We hand that gradient to the model.

## How the environment works

Each episode, the env hands the LLM:

1. A small Python library (e.g. `mini_calendar`, `csv_normalizer`, `bloom_filter_lite`).
2. The library's *existing* (weak) test suite.
3. A list of mutants the existing tests **fail to catch** — the gaps to fill.

The model writes new pytest functions and submits them. The env then:

1. Runs the new tests against the **unmodified** source. They must all pass.
   If any fail, the model wrote broken tests — no-regression gate fires, reward = 0.
2. Runs the new tests against each surviving mutant. Every mutant that
   *now* fails has been killed by the model's tests.
3. Computes reward ≈ (mutants killed) / (mutants the existing tests missed),
   with side terms for parsimony, coverage delta, and format.

That scalar gets fed back into the model's weights via GRPO.

## Why this matters

Anyone shipping LLM-written test code today is shipping confidence theatre:
green CI, undetected regressions. A model trained against mutation score
has been forced — by gradient, not by prompt — to think about which
inputs would *separate* a correct implementation from a wrong one. That's
the actual job of a test.

The same setup generalises beyond pytest: any verifier that turns a
correctness question into a pass/fail signal can plug into the same
reward shape.

## What's in the repo

- `src/mutant_hunter/` — the OpenEnv server: environment, mutation engine,
  rubric, sandbox, validators, tools the agent calls.
- `src/mutant_hunter/corpus/` — target libraries, manifest, precomputed
  baselines (so `/reset` is fast).
- `evaluation/` — sanity, determinism, adversarial, and zero-shot probes.
- `training/` — GRPO training loop.
- `scripts/precompute_baselines.py` — regenerates the per-module baseline cache.

## Running it locally

```bash
pip install -e .
mutant-hunter-server  # uvicorn on :8000
```

Or in Docker:

```bash
docker build -t mutant-hunter:latest .
docker run -p 8000:8000 mutant-hunter:latest
```

Tests:

```bash
python -m pytest -q tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).
