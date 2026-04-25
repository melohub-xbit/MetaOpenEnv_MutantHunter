# MutantHunter & CartographerZero — Master Project Blueprint

**Author context:** Solo participant, OpenEnv Hackathon India 2026 finale (Bangalore, April 25-26). B.Tech ML/NLP background with remote-sensing specialization. Submitting **MutantHunter** to the hackathon. **CartographerZero** is a secondary research project to be built post-hackathon (or as a stretch goal if Phase 2 ships early).

**Base model:** Qwen3-4B (Qwen3-4B-Instruct for chat-formatted RLVR; Qwen3-4B-Base if Unsloth requires base-model GRPO). Fits on a single A100 with QLoRA, plausibly on T4/L4 with aggressive quantization.

**Stack:**
- OpenEnv (latest from `meta-pytorch/OpenEnv`) for environment scaffolding
- Hugging Face TRL for GRPO trainer
- Unsloth for memory-efficient training
- Hugging Face Spaces for deployment (Docker-backed FastAPI)
- Weights & Biases for logging + public run links
- Hugging Face Hub for env, dataset, and trained-LoRA hosting
- Python 3.11+, FastAPI, pydantic v2, pytest, mutmut/cosmic-ray, Docker

**Critical references (Claude Code should read these before implementing):**
- OpenEnv repo: `https://github.com/meta-pytorch/OpenEnv`
- OpenEnv HF blog: `https://huggingface.co/blog/openenv`
- TRL OpenEnv integration: `https://huggingface.co/docs/trl/v0.27.1/openenv`
- Unsloth GRPO guide: `https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide`
- mutmut docs: `https://mutmut.readthedocs.io/`

---

# PART 1 — MUTANTHUNTER PHASE 1 (SAFE, SUBMISSION-READY)

## 1.1 The Pitch in One Paragraph

MutantHunter is an OpenEnv-compliant RL environment that trains LLMs to write *high-quality unit tests* via a reward signal grounded in **mutation testing**. Given a Python repository and its existing test suite, the agent writes additional tests; the reward is the fraction of *injected mutations* (operator swaps, constant changes, branch flips) that the agent's new tests *kill* but the existing suite missed. Mutation score is the gold-standard test-quality metric in software engineering — a deterministic, uncheatable, millisecond-fast oracle. The environment is a partial, queryable software world (Theme: World Modeling Professional). Phase 2 introduces a Mutator-agent that co-evolves with the Tester-agent, generating progressively harder mutations (Theme: Self-Improvement). The combination is genuinely novel — no public OpenEnv env trains test generation, and no prior work pairs RLVR with mutation-score supervision.

## 1.2 Theme Mapping (claim what's earned, gesture at the rest)

- **Wildcard**: primary. No prior art combining RLVR + mutation testing + agentic test generation in an OpenEnv env.
- **World Modeling (Professional)**: secondary. The repo is a partially-observable, action-gated world. The agent queries it through tools and must build an internal model of "what behaviors does this code have, what could break, what tests would catch breakage."
- **Self-Improvement**: Phase 2 only. Co-evolving Mutator/Tester loop.
- **Long-Horizon**: do not claim. Honestly medium-horizon (3-8 turns per episode).

## 1.3 Repository Layout (Phase 1)

```
mutant-hunter/
├── README.md                              # Story-first, judge-facing
├── pyproject.toml                         # Package config (mutant_hunter)
├── requirements.txt                       # Pinned deps
├── Dockerfile                             # HF Spaces target
├── openenv.yaml                           # OpenEnv manifest
├── LICENSE                                # Apache-2.0
├── .gitignore
├── .python-version                        # 3.11
│
├── src/mutant_hunter/
│   ├── __init__.py
│   ├── models.py                          # Action, Observation, State dataclasses (pydantic)
│   ├── environment.py                     # Core MutantHunterEnvironment (Gym-style)
│   ├── server.py                          # FastAPI app
│   ├── client.py                          # MutantHunterClient (HTTP)
│   │
│   ├── tools/                             # Tools the agent can call mid-episode
│   │   ├── __init__.py
│   │   ├── read_file.py
│   │   ├── list_tests.py
│   │   ├── run_tests.py
│   │   ├── get_coverage.py
│   │   └── get_mutation_report.py
│   │
│   ├── corpus/                            # The repo corpus
│   │   ├── __init__.py
│   │   ├── corpus_loader.py               # Loads pinned commits of target repos
│   │   ├── manifest.json                  # List of {repo, commit, modules, baseline_score}
│   │   └── _cache/                        # Cloned repos (gitignored)
│   │
│   ├── mutation/                          # Mutation engine
│   │   ├── __init__.py
│   │   ├── engine.py                      # Wrapper over mutmut/cosmic-ray
│   │   ├── operators.py                   # Op definitions (AOR, COR, ROR, LCR, etc.)
│   │   ├── injector.py                    # Apply mutations deterministically
│   │   └── runner.py                      # Sandboxed test execution
│   │
│   ├── rubric/                            # Reward functions
│   │   ├── __init__.py
│   │   ├── reward_mutation_kill.py        # Primary signal
│   │   ├── reward_no_regression.py        # Multiplicative gate
│   │   ├── reward_coverage_delta.py       # Secondary
│   │   ├── reward_format.py               # Pytest validity
│   │   ├── reward_parsimony.py            # Anti-bloat
│   │   └── compose.py                     # Combines all 5 into final scalar
│   │
│   ├── safety/                            # Sandboxing + anti-hack
│   │   ├── __init__.py
│   │   ├── sandbox.py                     # subprocess + resource limits + timeout
│   │   ├── validators.py                  # Pre-flight checks on agent output
│   │   ├── forbidden_patterns.py          # Block os.system, eval, subprocess in tests
│   │   └── README.md                      # Documents anti-hack design
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── generator.py                   # Episode = (repo, module, mutation_set)
│   │   ├── curriculum.py                  # Difficulty tiers
│   │   └── seeds/                         # Hand-picked deterministic episodes for eval
│   │       └── eval_set_v1.json
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── tracing.py                     # Per-step trace capture
│       └── pytest_helpers.py              # Discover, parse, run tests
│
├── training/
│   ├── train_grpo.ipynb                   # Colab notebook (judge-facing)
│   ├── train_grpo.py                      # Scriptable version
│   ├── config.yaml                        # All hyperparameters
│   ├── prompts.py                         # System + few-shot prompt templates
│   ├── baseline_eval.py                   # Run untrained model over eval_set
│   └── inspect_rollouts.py                # Sample N rollouts for hack detection
│
├── evaluation/
│   ├── eval_harness.py                    # Deterministic eval over seed set
│   ├── before_after.py                    # Side-by-side baseline vs trained
│   ├── reward_hacking_tests.py            # Adversarial probes (see §1.10)
│   └── ablations.py                       # With/without each reward component
│
├── plots/                                 # Committed PNGs (NOT in Colab cells only)
│   ├── reward_curve.png
│   ├── per_reward_breakdown.png           # Each rubric component over time
│   ├── mutation_kill_rate.png             # Headline metric
│   ├── baseline_vs_trained.png            # Bar chart
│   └── curriculum_progression.png         # Difficulty tier reached
│
├── docs/
│   ├── problem_statement.md
│   ├── reward_design.md                   # Walks through each reward, hacks blocked
│   ├── env_architecture.md                # Diagram + flow
│   ├── demo_script.md                     # 2-min video script
│   ├── safeguards.md                      # "Hacks we considered" — high-leverage doc
│   └── roadmap_phase2.md                  # Self-play extension plan
│
├── assets/
│   ├── env_diagram.svg
│   ├── reward_flow.svg
│   └── demo_thumbnail.png
│
├── tests/                                 # Tests for the env itself
│   ├── test_environment.py
│   ├── test_mutation_engine.py
│   ├── test_rewards.py
│   ├── test_sandbox.py
│   └── test_reward_hacking.py             # Adversarial cases must fail correctly
│
└── scripts/
    ├── prepare_corpus.sh                  # One-shot: clone all target repos at pinned commits
    ├── precompute_baselines.py            # Compute baseline mutation scores per (repo, module)
    └── deploy_hf_space.sh                 # One-command HF Space push
```

## 1.4 Detailed File-by-File Implementation Spec

### 1.4.1 `pyproject.toml`

```toml
[project]
name = "mutant_hunter"
version = "0.1.0"
description = "OpenEnv RL environment for training LLMs to write high-quality tests via mutation-score rewards"
authors = [{name = "<NAME>"}]
license = {text = "Apache-2.0"}
requires-python = ">=3.11"
dependencies = [
    "openenv>=0.x",                # pin to whatever finale version is current
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "mutmut>=2.5",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-timeout>=2.3",
    "coverage>=7.6",
    "GitPython>=3.1",
    "tenacity>=9.0",
    "rich>=13.7",
    "PyYAML>=6.0",
    "datasets>=3.0",                # HF datasets
    "huggingface-hub>=0.26",
    "wandb>=0.18",
]

[project.optional-dependencies]
training = [
    "torch>=2.5",
    "transformers>=4.46",
    "trl>=0.27",
    "unsloth>=2024.12",
    "accelerate>=1.1",
    "peft>=0.13",
    "bitsandbytes>=0.44",
]
dev = ["ruff>=0.7", "black>=24.10", "mypy>=1.13"]

[project.scripts]
mutant-hunter-server = "mutant_hunter.server:main"
mutant-hunter-eval = "evaluation.eval_harness:main"
```

### 1.4.2 `openenv.yaml`

OpenEnv manifest. Claude Code should follow OpenEnv's current schema; the rough shape is:

```yaml
name: mutant-hunter
version: 0.1.0
description: "RL environment for training LLM agents to write high-quality unit tests via mutation-score rewards"
author: "<NAME>"
license: Apache-2.0
entrypoint:
  type: fastapi
  module: mutant_hunter.server
  app: app
runtime:
  python: "3.11"
  dockerfile: Dockerfile
tools:
  - name: read_file
    description: "Read a source or test file from the target repo"
  - name: list_tests
    description: "List existing tests for the target module"
  - name: run_tests
    description: "Run the existing test suite against unmodified code"
  - name: get_coverage
    description: "Get line coverage for the target module"
  - name: get_mutation_report
    description: "Get list of surviving mutants from baseline test run"
spaces:
  observation_schema: src/mutant_hunter/models.py:Observation
  action_schema: src/mutant_hunter/models.py:Action
```

### 1.4.3 `src/mutant_hunter/models.py`

Pydantic v2 models for the OpenEnv contract. All fields strictly typed.

```python
from pydantic import BaseModel, Field
from typing import Literal, Any

class ToolCall(BaseModel):
    name: Literal["read_file", "list_tests", "run_tests", "get_coverage", "get_mutation_report"]
    args: dict[str, Any] = Field(default_factory=dict)

class Action(BaseModel):
    """Agent's action per turn. Either a tool call OR a final test submission."""
    kind: Literal["tool_call", "submit_tests"]
    tool_call: ToolCall | None = None
    test_code: str | None = Field(None, description="Final pytest file content")

class ToolResult(BaseModel):
    tool: str
    output: str
    truncated: bool = False

class Observation(BaseModel):
    repo_name: str
    module_path: str
    module_summary: str       # AST-derived: function signatures + docstrings
    existing_tests: list[str] # Names only
    baseline_mutation_score: float  # Pre-computed
    budget_remaining: int     # Tool calls left this episode
    history: list[ToolResult] # All prior tool outputs this episode
    turn: int

class State(BaseModel):
    """Full hidden state — never sent to agent."""
    repo_path: str
    module_path: str
    full_source: str
    full_test_suite: str
    surviving_mutants: list[dict]
    total_mutants: int
    coverage_baseline: float

class StepResult(BaseModel):
    observation: Observation | None
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
```

### 1.4.4 `src/mutant_hunter/environment.py`

The core class. Implements OpenEnv `Environment` interface.

```python
class MutantHunterEnvironment:
    """
    Episode flow:
      1. reset() picks a (repo, module) from the curriculum, resets budget=5 turns.
      2. Agent issues tool calls or submits tests.
      3. step() executes the action; if submit_tests, runs mutation testing and returns final reward.
      4. Episode ends on submit_tests or budget exhausted.
    """
    def __init__(self, corpus_loader, mutation_engine, rubric, sandbox, curriculum, max_turns=8):
        ...

    def reset(self, seed: int | None = None) -> Observation:
        # Pick task from curriculum, set up sandbox, return initial observation
        ...

    def step(self, action: Action) -> StepResult:
        # If tool_call: execute via sandbox, append to history, decrement budget, return obs + 0.0 reward
        # If submit_tests: run mutation pipeline, compute reward, return done=True
        ...

    def state(self) -> Observation:
        # Return current observation
        ...

    def close(self):
        # Tear down sandbox
        ...
```

Key implementation details:
- **`reset()` is deterministic given seed.** Critical for reproducibility.
- **`step()` is the only place reward is non-zero**, and only on `submit_tests`. No process rewards in Phase 1 (keeps it simple, debuggable).
- **All file system access via sandbox.** Never let the agent's code touch the host FS directly.
- **Budget tracking**: `budget_remaining` starts at 5 (tool calls only); `submit_tests` doesn't consume budget.

### 1.4.5 `src/mutant_hunter/server.py`

Standard OpenEnv FastAPI wrapper. Routes:
- `POST /reset` → returns Observation
- `POST /step` → takes Action, returns StepResult
- `GET /state` → returns current Observation
- `POST /close` → tears down

Use OpenEnv's helper if it provides one; otherwise write thin FastAPI handlers that delegate to `MutantHunterEnvironment`.

### 1.4.6 `src/mutant_hunter/client.py`

Mirror of server. HTTP client that the trainer uses. Must NOT import server internals (judges check for client/server separation).

### 1.4.7 `src/mutant_hunter/tools/`

Each tool is a function `(state: State, **args) -> str`. They run inside `step()` when action.kind == "tool_call".

- **`read_file(state, path: str, start_line: int = 0, end_line: int = -1) -> str`**: Returns file contents (or slice) from the sandboxed repo. Hard cap output at 4000 chars.
- **`list_tests(state) -> str`**: Returns names + first-line docstrings of all existing tests for the target module.
- **`run_tests(state) -> str`**: Runs existing test suite on unmodified code. Returns pass/fail counts + first 5 failures. *Useful for the agent to see what's already covered.*
- **`get_coverage(state) -> str`**: Returns line coverage for the target module under existing tests. Format: list of uncovered line ranges with surrounding context.
- **`get_mutation_report(state) -> str`**: Returns list of surviving mutants from the baseline run, formatted as `mutant_id, line, original → mutated`. Hard cap at 30 mutants.

### 1.4.8 `src/mutant_hunter/corpus/`

**The corpus is the most important data asset. Get it right.**

Target: 12-15 small, well-scoped Python libraries with crappy-to-medium test suites. Pin them at specific commits.

Selection criteria:
- 200-1500 LOC of source code
- Existing test suite covers <70% of mutants (so there's room to improve)
- Pure Python, no compiled deps that hurt sandbox setup
- Permissive license (MIT, Apache, BSD)
- Self-contained (no DB, network, or filesystem deps in the code under test)

Concrete candidates Claude Code should investigate (verify availability + licenses):
- `dateparser`-style helpers
- Small parsers: `pyhocon`, `python-rapidjson` wrappers, simple INI parsers
- Algorithms: `python-binary-search-tree`, small graph libs
- Utilities: tiny string/url manipulation libs from PyPI
- Self-curated mini-libs: write 4-5 of your own 200-LOC libraries (`mini_calendar`, `csv_normalizer`, `interval_tree`, `bloom_filter_lite`) — these are *guaranteed* to be uncontaminated by training data.

**Self-curated libs are critical** because a mutmut-on-popular-pypi-package result might be in pretraining data; self-written libs can't be.

**`manifest.json` schema:**

```json
{
  "version": "v1",
  "repos": [
    {
      "name": "mini_calendar",
      "source": "local",  // or "git"
      "path": "src/mutant_hunter/corpus/_local/mini_calendar",
      "commit": null,
      "modules": [
        {
          "module": "mini_calendar.parser",
          "loc": 287,
          "baseline_mutation_score": 0.41,
          "total_mutants": 78,
          "difficulty_tier": 1
        }
      ]
    }
  ]
}
```

`baseline_mutation_score` and `total_mutants` are precomputed by `scripts/precompute_baselines.py` once during prep, then never recomputed during training (saves ~30s per episode).

### 1.4.9 `src/mutant_hunter/mutation/`

The mutation engine. Two implementation paths:

**Option A (recommended for Phase 1): use `mutmut` directly.**
- Pros: mature, fast, well-documented.
- Cons: less flexible mutation operators.

**Option B: custom AST-based injector.**
- Pros: full control over operator set, deterministic ordering, fast.
- Cons: more code to write/test.

Go with Option A for Phase 1, build Option B as a backup if mutmut performance is a problem.

`engine.py` interface:

```python
class MutationEngine:
    def precompute_mutants(self, repo_path: str, module: str) -> list[Mutant]:
        """Run once per (repo, module) during prep; cache to disk."""
        ...

    def run_baseline(self, repo_path: str, module: str, test_dir: str) -> MutationReport:
        """Compute baseline: which mutants does the existing suite kill?"""
        ...

    def run_with_new_tests(self, repo_path: str, module: str, new_test_code: str) -> MutationReport:
        """Compute: which mutants does (existing + new) suite kill?"""
        ...
```

**Performance constraint:** every `step()` with `submit_tests` runs the full test suite on every surviving mutant. This is expensive. Mitigations:
- Pre-filter to top-K=15 most informative surviving mutants per module.
- Use `pytest-xdist` for parallel mutant execution.
- Hard timeout per mutant: 8 seconds.
- Cache mutant ASTs.

Target: ≤20 seconds per `submit_tests` step. If slower, training rollouts dominate budget.

### 1.4.10 `src/mutant_hunter/rubric/`

The reward functions. Each is a pure function `(state, action, exec_result) -> float`.

**`reward_mutation_kill.py`**:
```python
def reward_mutation_kill(state: State, exec_result: dict) -> float:
    """
    Primary signal: fraction of surviving baseline mutants killed by new tests.
    Surviving = mutants that the *original* test suite did NOT kill.
    """
    baseline_surviving = state.surviving_mutants  # precomputed
    if not baseline_surviving:
        return 0.5  # nothing to kill; neutral reward
    killed_by_new = exec_result["killed_by_new_only"]
    return len(killed_by_new) / len(baseline_surviving)
```

**`reward_no_regression.py`** — multiplicative gate:
```python
def reward_no_regression(exec_result: dict) -> float:
    """1.0 if all new tests pass on UNMODIFIED code; 0.0 otherwise."""
    return 1.0 if exec_result["new_tests_pass_clean"] else 0.0
```

**`reward_coverage_delta.py`**:
```python
def reward_coverage_delta(state: State, exec_result: dict) -> float:
    """Improvement in line coverage, normalized."""
    delta = exec_result["new_coverage"] - state.coverage_baseline
    headroom = max(100.0 - state.coverage_baseline, 1.0)
    return max(0.0, min(1.0, delta / headroom))
```

**`reward_format.py`**:
```python
def reward_format(action: Action, exec_result: dict) -> float:
    """1.0 if the submitted test file parses, runs, and only contains pytest-style tests."""
    if not exec_result["parses"]: return 0.0
    if exec_result["contains_forbidden"]: return 0.0
    return 1.0
```

**`reward_parsimony.py`**:
```python
def reward_parsimony(action: Action) -> float:
    """Mild penalty for excessively long tests (>20 LOC each)."""
    n_lines_per_test = ...  # parse pytest functions
    avg = mean(n_lines_per_test)
    return max(0.0, 1.0 - max(0, avg - 20) / 30)
```

**`compose.py`**:
```python
WEIGHTS = {
    "mutation_kill": 0.60,
    "coverage_delta": 0.20,
    "format": 0.15,
    "parsimony": 0.05,
}

def compose_reward(state: State, action: Action, exec_result: dict) -> dict:
    components = {
        "mutation_kill": reward_mutation_kill(state, exec_result),
        "coverage_delta": reward_coverage_delta(state, exec_result),
        "format": reward_format(action, exec_result),
        "parsimony": reward_parsimony(action),
    }
    no_reg = reward_no_regression(exec_result)
    weighted = sum(WEIGHTS[k] * v for k, v in components.items())
    final = no_reg * weighted   # gate: any regression → 0 reward
    return {"final": final, "components": components, "no_regression_gate": no_reg}
```

**This composition is the heart of the submission.** Document every line in `docs/reward_design.md`.

### 1.4.11 `src/mutant_hunter/safety/`

**This is what wins extra points with Red Hat / Meta engineers.** Document everything.

**`sandbox.py`**:
- All test execution inside a subprocess with:
  - CPU time limit (resource.RLIMIT_CPU = 30s)
  - Memory limit (resource.RLIMIT_AS = 512MB)
  - No network (use `unshare -n` on Linux, or NetworkPolicy in container)
  - Read-only filesystem except `/tmp/mutant_hunter_episode_<uuid>/`
  - Drops privileges
- Hard timeout via `signal.alarm` or subprocess.timeout

**`forbidden_patterns.py`** — block these in agent-submitted test code (regex + AST):
- `import os` then `os.system`
- `subprocess` (any)
- `eval`, `exec`
- `__import__`
- File writes outside `/tmp`
- `open(...)` with mode containing `'w'`, `'a'`, `'x'`
- `socket`, `urllib`, `requests`, `httpx`
- Direct mutation of `__builtins__`, `globals()`, `sys.modules`
- Patching of mutmut/pytest internals
- Time manipulation: `time.sleep` (just because; not a real attack vector but a clean signal)
- Environment variable writes

**`validators.py`**:
- AST validity check before any execution
- Test function naming (`test_*`)
- Maximum file size (50KB)
- Maximum number of tests per submission (50)

### 1.4.12 `src/mutant_hunter/tasks/`

**`generator.py`**: deterministic episode sampling from the corpus given a seed.

**`curriculum.py`**: difficulty tiers.
- Tier 1: small modules (200-400 LOC), single-file, simple types. (Easy mutants to catch.)
- Tier 2: medium modules (400-800 LOC), multiple classes, generic types.
- Tier 3: complex modules (800-1500 LOC), inheritance, decorators, edge cases in numerical code.

Curriculum policy:
- Episodes 1–200: Tier 1 only.
- Episodes 200–500: 70% Tier 1, 30% Tier 2.
- Episodes 500+: 30% Tier 1, 50% Tier 2, 20% Tier 3.

Adjust based on rolling success rate (if median final reward > 0.5 in last 50 episodes on current tier, advance).

**`seeds/eval_set_v1.json`**: 30 deterministic (repo, module, seed) tuples used for ALL evaluations. Never trained on. Critical for "before/after" plots being credible.

### 1.4.13 `training/`

**`train_grpo.py`**:

```python
# Pseudo-code skeleton
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from mutant_hunter.client import MutantHunterClient

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-4B-Instruct",  # or Qwen3-4B-Base, depending on Unsloth's GRPO support
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, target_modules=[...],
)

env_client = MutantHunterClient(base_url="http://localhost:8000")

def reward_fn(prompts, completions, **kwargs):
    """Per-completion reward via the env. Trainer rolls out, we score via env."""
    rewards = []
    for completion in completions:
        action = parse_action(completion)
        result = env_client.step(action)
        rewards.append(result.reward)
    return rewards

config = GRPOConfig(
    output_dir="./checkpoints",
    num_generations=4,
    max_steps=300,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=1,
    report_to="wandb",
)

trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=config, ...)
trainer.train()
trainer.save_model("./final-lora")
```

Key constraints:
- `num_generations=4` (minimum for GRPO advantage estimation; higher = more compute)
- `max_steps=300` Phase 1 (fits in ~4-6 hours on A100; budget 8h max)
- W&B logging from step 1 with public link

**`prompts.py`**: system + few-shot prompts. Critical for warm-starting GRPO so non-zero reward happens early.

```python
SYSTEM_PROMPT = """You are MutantHunter, an expert at writing pytest unit tests that catch bugs.

You will be given a Python module and its existing tests. Your job is to write ADDITIONAL tests that catch behaviors the existing tests miss.

You have tools:
- read_file(path): read source/test files
- list_tests(): list existing tests
- run_tests(): run existing suite
- get_coverage(): see which lines lack coverage
- get_mutation_report(): see what mutations the existing suite fails to catch

When ready, output your tests in this format:
<submit>
```python
import pytest
from <module> import <thing>

def test_thing_with_negative_input():
    assert ...
```
</submit>

Rules:
- Tests must pass on the unmodified code (no regression).
- No imports of os, sys, subprocess, eval, exec.
- Each test ≤ 20 lines.
- Focus on edge cases the existing tests miss."""
```

Add 2 worked examples in the few-shot.

**`baseline_eval.py`**: runs untrained Qwen3-4B-Instruct on the eval seed set. Saves results to `evaluation/results/baseline.json`. **Run this before training. The before/after comparison is the demo.**

**`inspect_rollouts.py`**: every 25 training steps, sample 5 random rollouts and dump to `logs/rollouts/step_<N>.jsonl`. Manual inspection catches reward hacking before it dominates.

### 1.4.14 `evaluation/`

**`eval_harness.py`**: pure function `evaluate(model_path: str, eval_set: str) -> dict[str, float]`. Returns per-task and aggregate metrics.

**`before_after.py`**: produces the killer plot. Same eval seeds, baseline vs trained model, side-by-side.

**`reward_hacking_tests.py`**: adversarial probes that MUST fail correctly. This file is a high-leverage README artifact.

```python
ADVERSARIAL_CASES = [
    {
        "name": "empty_test_file",
        "test_code": "",
        "expected_reward_lt": 0.05,
    },
    {
        "name": "always_passes",
        "test_code": "def test_a(): assert True",
        "expected_reward_lt": 0.1,
    },
    {
        "name": "imports_subprocess",
        "test_code": "import subprocess\ndef test_a(): subprocess.run(['ls']); assert True",
        "expected_reward_eq": 0.0,
    },
    {
        "name": "vacuous_assertion",
        "test_code": "def test_a(): x = 1; assert x == x",
        "expected_reward_lt": 0.1,
    },
    {
        "name": "regression_introduced",
        "test_code": "def test_a(): assert 1 == 2  # Always fails",
        "expected_reward_eq": 0.0,  # no_regression gate fires
    },
    # Add 10 more
]

def test_all_adversarial_cases_blocked():
    for case in ADVERSARIAL_CASES:
        result = run_in_env(case["test_code"])
        if "expected_reward_lt" in case:
            assert result.reward < case["expected_reward_lt"], f"FAIL: {case['name']}"
        ...
```

This file becomes a slide in the demo deck. *"Here are 15 hacks we considered. Here's our reward function refusing all of them."*

### 1.4.15 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for sandbox + git for corpus
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Pre-clone corpus
RUN bash scripts/prepare_corpus.sh

EXPOSE 7860
CMD ["uvicorn", "mutant_hunter.server:app", "--host", "0.0.0.0", "--port", "7860"]
```

Port 7860 = HF Spaces standard.

### 1.4.16 `README.md` (judge-facing — most important file)

Structure (judge spends 3-5 min):

```markdown
# MutantHunter
*An OpenEnv RL environment that teaches LLMs to write high-quality tests by killing mutated code.*

[Live HF Space] [Colab Training] [W&B Run] [2-min Demo Video] [Blog Post]

![Headline plot: mutation score over training steps](plots/mutation_kill_rate.png)
*Baseline Qwen3-4B kills 18% of injected mutants. After 250 GRPO steps with mutation-score reward: 63%.*

## The Problem in 30 Seconds
[Concrete example: a 200-LOC date-parsing library. Its test suite has 95% line coverage. But mutmut shows the suite catches only 18% of bug-equivalent mutations. Coverage lies; mutation score doesn't.]

## What This Environment Trains
[3-line description of the agent's task]

## Why It's Novel
- First OpenEnv env for test generation
- First use of mutation score as RLVR reward
- Composable rubric (5 components) with multiplicative no-regression gate
- 15 adversarial cases all blocked (see safeguards.md)

## Results
[Bar chart: baseline vs trained on 30 held-out (repo, module) pairs]
[Per-component reward breakdown]
[Sample rollout: baseline writes 1 vacuous test; trained writes 6 targeted tests]

## How to Run
```bash
pip install -e .
mutant-hunter-server  # starts FastAPI on :7860
# In another terminal:
jupyter notebook training/train_grpo.ipynb
```

## Repo Structure
[Tree, 10 lines]

## Safeguards Against Reward Hacking
[Table: hack → defense → test in evaluation/reward_hacking_tests.py]

## Roadmap (Phase 2)
Self-play: a Mutator-agent co-evolves with the Tester, generating progressively harder mutations. See `docs/roadmap_phase2.md`.

## License & Citation
Apache-2.0. If you use this, please cite [your write-up].
```

## 1.5 Phase 1 — 7-Day Build Plan

| Day | Goal | Deliverable |
|---|---|---|
| 1 | Skeleton + corpus | Repo structure created, 4 self-curated mini-libs written, `manifest.json` populated for tier 1 |
| 2 | Mutation engine | `mutation/engine.py` working; `precompute_baselines.py` produces `baseline_mutation_score` for all tier 1 modules |
| 3 | Sandbox + tools | `safety/sandbox.py` with all forbidden patterns; 5 tool functions |
| 4 | Environment + server | `environment.py` + `server.py` running locally; manual curl tests pass |
| 5 | Rubric + reward hacking tests | All 5 reward functions implemented; `evaluation/reward_hacking_tests.py` with 15 cases; all pass |
| 6 | Training pipeline | `train_grpo.py` runs 30 steps end-to-end without crashing |
| 7 | First real run + iterate | 200 steps; inspect rollouts; tune reward weights; first reward curve |

## 1.6 Phase 1 — On-Site 48-Hour Plan

Hour 0-6: Push to HF Space, verify it loads, fix Docker issues. Confirm Colab notebook runs end-to-end.
Hour 6-18: Full training run, 300+ GRPO steps. Iterate on prompt or hyperparams if curve is flat.
Hour 18-24: Sleep.
Hour 24-30: Baseline evaluation on eval set. Generate all plots.
Hour 30-36: README polish, demo script, record 2-min video.
Hour 36-42: Sleep.
Hour 42-48: Final dry runs of demo, push final commit, verify everything is linked from README.

## 1.7 Reward Hacking — Defense Catalog

Document this as a separate artifact (`docs/safeguards.md`). Each row is a slide-worthy bullet.

| Hack | Defense | Test |
|---|---|---|
| Empty test file | `format` reward = 0 if no test functions | `test_empty_test_file` |
| `assert True` everywhere | Vacuous tests don't kill mutants → `mutation_kill` ≈ 0 | `test_always_passes` |
| `import subprocess` to spawn shell | `forbidden_patterns.py` blocks; `format` = 0 | `test_imports_subprocess` |
| Tests that fail on clean code (sneaky regression) | `no_regression` gate → final reward = 0 | `test_regression_introduced` |
| Tautologies (`assert x == x`) | Don't kill mutants; AST-level detection optional | `test_vacuous_assertion` |
| Patching the test runner from inside the test | Sandbox prevents writes outside `/tmp/episode/`; AST blocks `unittest.mock.patch` of pytest internals | `test_patches_pytest` |
| Massive output spam (token waste) | `parsimony` reward; max test count 50 | `test_spam_tests` |
| Network exfiltration | `unshare -n` in sandbox + forbidden imports | `test_network_attempt` |
| Write to host FS | Read-only mount + sandbox uid drop | `test_filesystem_attempt` |
| Time-based attack (sleep until killed by timeout) | `RLIMIT_CPU = 30s` per execution | `test_infinite_loop` |
| `sys.exit(0)` to fake success | Subprocess return code checked separately from pytest output | `test_sys_exit_zero` |
| Duplicate existing tests verbatim | Already-killed mutants don't double-count; reward is *new* mutants killed | `test_duplicate_existing_tests` |
| Mutation operator allowlist exploit (target only safe ops) | Mutants are pre-selected from full operator set, not agent-chosen | `test_mutation_set_fixed` |
| Hallucinate function names that don't exist | Test fails on clean code → `no_regression` = 0 | `test_hallucinated_names` |
| Inject `pytest.skip` everywhere | Skipped tests don't kill mutants; treated as not-run | `test_pytest_skip_everywhere` |

15 cases documented = signal of maturity.

---

# PART 2 — MUTANTHUNTER PHASE 2 (PUSH THE ENVELOPE: SELF-PLAY)

## 2.1 The Extension in One Paragraph

Phase 2 introduces a **Mutator-agent** that co-evolves with the Tester-agent. The Mutator proposes mutation candidates from a constrained grammar; the Tester writes tests; reward to Mutator = `p*(1-p)` where `p` is the Tester's kill rate (Absolute-Zero / R-Zero learnability reward); reward to Tester = standard mutation kill rate. They alternate training. The result: a self-improving curriculum where mutations get progressively harder as the Tester gets better, without any human in the loop. This is genuine RLVR-grounded self-play in software.

## 2.2 What Changes vs Phase 1

- New role: `Mutator` agent.
- New action space for Mutator: structured mutation proposals (operator + target + replacement).
- New reward: learnability for Mutator.
- New training loop: alternating GRPO updates (Tester epoch, Mutator epoch, Tester epoch, ...).
- Same Tester env from Phase 1 (compositional!).

## 2.3 Phase 2 File Additions

```
mutant-hunter/
├── src/mutant_hunter/
│   ├── self_play/                         # NEW
│   │   ├── __init__.py
│   │   ├── mutator_environment.py         # Env for the Mutator agent
│   │   ├── mutator_models.py              # Action/Obs for Mutator
│   │   ├── mutator_rubric.py              # Learnability reward
│   │   ├── mutation_grammar.py            # Constrained ops + targets
│   │   └── coevolution_loop.py            # Alternating training driver
│   │
│   └── ...
│
├── training/
│   ├── train_self_play.ipynb              # NEW — phase 2 driver
│   └── train_self_play.py                 # NEW
│
├── evaluation/
│   └── self_play_progression.py           # Difficulty-over-time plots
│
└── docs/
    └── phase2_self_play.md                # Methodology writeup
```

## 2.4 Mutator Action Grammar

The Mutator can NOT propose arbitrary diffs (too unconstrained, easy to hack). Instead, a structured grammar:

```python
class MutatorAction(BaseModel):
    operator: Literal[
        "AOR",    # Arithmetic Operator Replacement: + → -, * → //, etc.
        "ROR",    # Relational: <, >, <=, >=, ==, !=
        "LCR",    # Logical Connector: and ↔ or
        "BCR",    # Boolean Constant: True ↔ False
        "NCR",    # Numeric Constant: replace with neighbor (n → n+1, n → 0)
        "SCR",    # String Constant: empty/swap
        "BOUNDARY",  # Off-by-one: range(n) → range(n-1)
    ]
    target_module: str
    target_line: int
    target_column: int  # for disambiguation
    replacement_index: int  # 0-K within operator's allowed swaps
```

Mutator's role: *predict which (operator, target) the current Tester is least likely to catch.*

## 2.5 Learnability Reward

```python
def mutator_reward(tester_kill_probability: float) -> float:
    """
    Per AZR: reward = p*(1-p), peaking at p=0.5 (right at edge of Tester capability).
    Tester kill prob estimated by running 4-8 Tester rollouts on the proposed mutant.
    """
    p = tester_kill_probability
    return 4.0 * p * (1.0 - p)  # scaled to peak at 1.0
```

Plus diversity bonus: penalize Mutator for proposing mutations too similar to last 50.

## 2.6 Co-Evolution Training Loop

```python
def train_self_play(initial_tester, initial_mutator, n_outer_epochs=10):
    tester = initial_tester
    mutator = initial_mutator

    for outer in range(n_outer_epochs):
        # 1. Mutator generates 200 mutation candidates
        candidates = mutator.rollout(corpus, n=200)

        # 2. Estimate Tester kill prob on each (k=4 rollouts)
        kill_probs = [estimate_kill_prob(tester, c, k=4) for c in candidates]

        # 3. Train Mutator (50 GRPO steps) on learnability reward
        mutator.update(candidates, kill_probs)

        # 4. Filter candidates to "interesting" (0.2 < p < 0.8)
        training_set = [c for c, p in zip(candidates, kill_probs) if 0.2 < p < 0.8]

        # 5. Train Tester (100 GRPO steps) on mutation-kill reward against training_set
        tester.update(training_set)

        log_metrics(outer, tester, mutator, candidates, kill_probs)
```

## 2.7 Phase 2 Risks

- **Compute doubles** (two models). Drop to Qwen3-1.7B for Phase 2 if VRAM tight.
- **Mode collapse**: Mutator finds one hack the Tester always misses, never explores. Mitigate with diversity bonus + KL penalty against initial Mutator.
- **Mutator easier to reward-hack** (it's reward = how confused Tester is = trivially gameable by proposing nonsense). Mitigate by requiring Mutator output to actually compile + change semantics (verifier!).

## 2.8 Phase 2 — 4-Day Build Plan (Days 8-11 of prep)

| Day | Goal |
|---|---|
| 8 | Mutator action grammar + env |
| 9 | Learnability reward + alternating training loop |
| 10 | First co-evolution run |
| 11 | Difficulty-progression plots, README updates |

Days 12-16 = polish, deployment, demo recording, buffer.

## 2.9 Phase 2 Storytelling Hook

*"The Tester learns to catch bugs. But who decides what bugs to plant? In Phase 1, we did. In Phase 2, an agent does — and that agent learns to plant exactly the bugs the Tester is on the verge of being able to catch. The result is a curriculum that adapts in real time. You see this in the training curves: as the Tester gets better, the Mutator gets sneakier, and the kill rate stays in the 'productive struggle' zone. This is genuine RLVR self-play, in software, with a verifier (the test runner) keeping both honest."*

That's three themes earned (Wildcard, World Modeling, Self-Improvement) and a paper abstract written.

---

# PART 3 — CARTOGRAPHERZERO (POST-HACKATHON / STRETCH)

## 3.1 Status

Build only after MutantHunter is fully shipped. Could become:
- ArXiv preprint (multi-turn calibrated geospatial RL)
- Portfolio piece
- Submission to a future hackathon (NeurIPS dataset/benchmark, EGU)

## 3.2 The Pitch

OpenEnv-compliant RL environment for **multi-turn, budgeted, calibrated geospatial reasoning**. Agent answers questions about Sentinel-1+2 satellite imagery (flood detection on Sen1Floods11; land-cover classification on BigEarthNet v2.0) with limited initial observation. Tools: `request_band(b)`, `request_neighbor(dir)`, `request_prior_scene(delta_days)`. Budget: 3 tool calls. Outputs: `{answer, confidence, evidence}`. Rewards: correctness + Brier calibration (per RLCR) + information efficiency + format. First OpenEnv geospatial env; first multi-turn extension of RLCR; first active-perception RL formulation in remote sensing.

## 3.3 Repository Layout

```
cartographer-zero/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── openenv.yaml
├── LICENSE
│
├── src/cartographer_zero/
│   ├── __init__.py
│   ├── models.py                      # Action/Obs/State for geospatial QA
│   ├── environment.py
│   ├── server.py
│   ├── client.py
│   │
│   ├── data/
│   │   ├── sen1floods11_loader.py     # Tile loading, band caching
│   │   ├── bigearthnet_loader.py
│   │   ├── feature_extractor.py       # Per-band mean/std/percentiles
│   │   └── _cache/
│   │
│   ├── tools/
│   │   ├── request_band.py
│   │   ├── request_neighbor.py
│   │   └── request_prior_scene.py
│   │
│   ├── rubric/
│   │   ├── reward_correctness.py
│   │   ├── reward_brier.py            # RLCR-style: -(conf - correct)^2
│   │   ├── reward_info_efficiency.py
│   │   ├── reward_evidence.py         # Optional v2: feature-importance check
│   │   ├── reward_format.py
│   │   └── compose.py
│   │
│   ├── tasks/
│   │   ├── flood_qa_generator.py
│   │   ├── landcover_qa_generator.py
│   │   └── seeds/eval_set_v1.json
│   │
│   └── utils/
│       ├── visualization.py           # Tile + bands + answer overlay
│       └── tracing.py
│
├── training/
│   ├── train_grpo.ipynb
│   ├── train_grpo.py
│   ├── config.yaml
│   ├── prompts.py
│   └── baseline_eval.py
│
├── evaluation/
│   ├── eval_harness.py
│   ├── before_after.py
│   └── calibration_metrics.py         # ECE, MCE, Brier
│
├── plots/
│   ├── correctness_curve.png
│   ├── calibration_curve.png          # Reliability diagram
│   ├── budget_usage.png
│   └── baseline_vs_trained.png
│
├── docs/
│   ├── problem_statement.md
│   ├── reward_design.md
│   ├── india_relevance.md             # NDMA, ISRO, monsoon flood case
│   └── why_calibration_matters.md
│
└── scripts/
    ├── download_datasets.sh           # Sen1Floods11, BigEarthNet subsets
    └── precompute_features.py
```

## 3.4 Key Implementation Notes

- **Base model: Qwen2.5-VL-3B** (vision-language) or **Qwen3-4B + structured features only** (text-only, easier).
  - **Recommended**: text-only with structured features. The agent never sees raw imagery — it sees per-band statistics (mean, std, percentiles) and gets to *request* more bands. This sidesteps VLM training complexity and keeps it Qwen3-4B-compatible.
- **Datasets**: pull small subsets (~500 tiles each) from Sen1Floods11 and BigEarthNet v2.0. Both have permissive licenses.
- **Brier reward** is the heart. `-(confidence - correct)²` directly per RLCR.
- **Evidence reward** is the stretch. Requires precomputed feature-importance maps (via SHAP on a small XGBoost trained on the same task).

## 3.5 14-Day Post-Hackathon Plan

Build at leisure post-hackathon. Aim for ArXiv preprint within 6 weeks of hackathon end.

---

# PART 4 — SHARED INFRASTRUCTURE NOTES

## 4.1 Hugging Face Spaces Deployment

Both envs target `huggingface.co/spaces/<your-username>/<env-name>`. SDK = `docker`. Use Spaces' free CPU tier for the env server (it doesn't need a GPU — training happens elsewhere).

`scripts/deploy_hf_space.sh`:
```bash
#!/bin/bash
set -euo pipefail
HF_USER=$1
ENV_NAME=$2
git clone "https://huggingface.co/spaces/$HF_USER/$ENV_NAME" /tmp/space || true
rsync -av --exclude='.git' --exclude='**/__pycache__' --exclude='_cache' . /tmp/space/
cd /tmp/space
git add -A
git commit -m "Deploy $(date -u +%FT%TZ)"
git push
```

## 4.2 Hugging Face Hub Artifacts

- Env: HF Space (Docker)
- Eval dataset: HF Dataset (`<user>/<env-name>-eval`)
- Trained LoRA: HF Model (`<user>/<env-name>-qwen3-4b-lora`)
- Blog post: HF Blog (or LinkedIn / Medium with HF Space link)

All linked from README.

## 4.3 W&B Setup

- Project: `mutant-hunter` and `cartographer-zero`
- Public report at end of training; link in README
- Critical charts: total reward, per-component reward, success rate, episode length, eval-set kill rate (every 25 steps)

## 4.4 Compute Budgets

- **MutantHunter Phase 1**: 1× A100-40GB or L40S, 6-8 hours total training.
- **MutantHunter Phase 2**: 1-2× A100-40GB, 12-16 hours total (alternating loop).
- **CartographerZero**: 1× A100, 4-6 hours (single model, simpler).

If only T4 available on Colab: drop to Qwen3-1.7B, halve max_steps, expect noisier curves but real signal still visible.

---

# PART 5 — INSTRUCTIONS FOR CLAUDE CODE

When implementing, Claude Code should:

1. **Start with Phase 1 ONLY.** Do not scaffold Phase 2 or CartographerZero files until Phase 1 is shipped.
2. **Read the OpenEnv repo before generating any env code.** The exact API surface may have changed; defer to actual current docs.
3. **Follow this order strictly** (matches the 7-day plan):
   1. `pyproject.toml`, `Dockerfile`, `requirements.txt`, `.gitignore`, `LICENSE`
   2. `src/mutant_hunter/models.py` (the contract)
   3. 4 self-curated mini-libs in `src/mutant_hunter/corpus/_local/`
   4. `src/mutant_hunter/mutation/` — engine first
   5. `scripts/precompute_baselines.py` — run it, populate manifest
   6. `src/mutant_hunter/safety/sandbox.py` + `forbidden_patterns.py`
   7. `src/mutant_hunter/tools/` — all 5 tools
   8. `src/mutant_hunter/environment.py`
   9. `src/mutant_hunter/server.py`
   10. `src/mutant_hunter/client.py`
   11. `src/mutant_hunter/rubric/` — all 5 reward functions + compose
   12. `tests/` — env tests first (must pass before training!)
   13. `evaluation/reward_hacking_tests.py` — adversarial cases
   14. `training/train_grpo.py` (small smoke run, 10 steps)
   15. `training/baseline_eval.py` — establish baseline
   16. Real training run, 200-300 GRPO steps
   17. `evaluation/before_after.py`, plot generation
   18. README, docs/, demo materials

4. **Use the test harness as ground truth.** If `evaluation/reward_hacking_tests.py` doesn't pass, the env is broken — fix before training.

5. **Pin all dependencies** before training begins. A version bump mid-training will ruin the run.

6. **Commit early, commit often.** The HF Space is a public git repo — judges will see commit history. Show iteration, not a single mega-commit.

7. **When stuck on OpenEnv specifics, prefer the framework's own examples** (Echo, Wordle, 2048) over guessing. Read their source.

8. **Do NOT over-engineer.** Phase 1 should ship even if Phase 2 never starts. Every file created in Phase 1 must be necessary for Phase 1.

9. **The reward function is the soul of the project.** When in doubt, spend more time on `rubric/` than on anything else.

10. **The README is the second-most-important file.** Write it once Phase 1 works end-to-end, then iterate. Don't write it last.

---

# APPENDIX A — Decision Log

- **Why Qwen3-4B over Qwen2.5-Coder-7B?** User preference. Phase 1 will work; if reward curve flat after 200 steps, swap to Coder-3B.
- **Why mutmut over cosmic-ray?** Faster, simpler, more mature. Cosmic-ray for Phase 2 if needed.
- **Why 5 reward components, not 3?** Each closes a specific hack vector. See safeguards.md.
- **Why no process rewards in Phase 1?** Adds debugging surface. Keep Phase 1 outcome-only; add process rewards in Phase 2 if needed.
- **Why self-curated mini-libs over big PyPI packages?** Eliminates training-data contamination concerns. Judges will ask.

# APPENDIX B — What Could Go Wrong (Pre-Mortem)

1. **Mutmut is too slow on chosen modules** → cap to 15 mutants per module via filtering; use `pytest-xdist`.
2. **GRPO reward is flat for 100 steps** → warm-start with SFT on 50 hand-written test examples.
3. **HF Space build fails** → test Dockerfile locally; pin every dep; check Space logs immediately on first push.
4. **Sandbox escapes happen** → review subprocess setup; add seccomp filters if Linux capabilities allow.
5. **On-site no internet for HF push** → have Space pre-deployed Day 14; on-site only for fixes.
6. **Compute credits run out mid-training** → checkpoint every 25 steps; resume from last good.
7. **Demo crashes during pitch** → record video as backup; have local reproduction ready.
8. **Reward hacking emerges in late training** → `inspect_rollouts.py` runs every 25 steps; manual sanity check.

End of blueprint.
