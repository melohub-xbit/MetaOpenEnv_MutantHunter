# Teaching LLMs to Write Tests That Actually Catch Bugs

**Demo video**: <https://youtu.be/WW2YbD9o2eA>

## The 100% coverage paradox

Open any large Python codebase with a coverage gate. You will find files at 100%, modules at 95%, packages reporting "fully tested." Now break a function — flip a `+` to a `-`, an `>=` to a `>`, a `True` to a `False` — and run the suite. Many of those tests will still pass. Coverage measured *which lines executed during the test*. It said nothing about whether the tests would have *noticed* if those lines were wrong. A test that calls `add(2, 2)` and then asserts the result is "an integer" runs the line, hits the gate, and protects nothing. The same gap is now being widened by every LLM that writes tests for free in CI: optimized for green checkmarks, not for the property a test is supposed to provide.

## The problem

The capability gap is narrow but load-bearing. We do not need an LLM that writes more tests; we need an LLM that writes tests an adversary would have to work to fool. The training signal it has been given so far points the wrong way.

When humans write tests, the implicit reward is "the test passes and the PR merges." When LLMs write tests, the explicit reward is "lint clean, coverage up, CI green." Both reward functions are satisfied by the *vacuous* test — the `assert isinstance(result, int)`, the `try / except: pass`, the `assert add(2, 2) is not None`. Such tests pass on the correct code and they pass on broken code, and that is exactly the failure mode you do not want to ship into a regression suite that future humans will trust. Worse, when an LLM gets stuck, "make the test less specific" is the easiest move available to it. Every bad incentive in the data points the same way.

What is missing is an objective signal — a thing the model cannot fake by writing softer assertions — that tells it whether a given test would actually have caught a bug. That signal is what mutation testing provides.

## Mutation testing as an RL reward

Mutation testing inverts the question. Instead of asking *did the test run the code*, it asks: *if I deliberately break the code in a small way, will the tests notice?*

Consider:

```python
def add(a, b):
    return a + b

def test_add():
    assert add(2, 2) == 4
```

Coverage is 100%. Now flip the operator:

- `+` → `-`: `add(2, 2)` returns `0`, test fails. **Mutant killed.**
- `+` → `*`: `add(2, 2)` returns `4`, test passes. **Mutant survived.**
- `+` → `**`: `add(2, 2)` returns `4`, test passes. **Mutant survived.**

One out of three. Mutation score: 33%. Adding `assert add(2, 3) == 5` and `assert add(0, 0) == 0` kills the other two. Same coverage, materially stronger test.

That fraction — `mutants_killed / mutants_introduced` — is a verifiable, dense, non-negotiable scalar. It is not a preference signal; it is a property of the code under known perturbation. It cannot be reward-hacked by writing more confident-sounding prose. It is exactly the shape of reward modern RL'd-LLM training pipelines were built to consume. So we build the environment that produces it, and we hand the gradient to GRPO.

## The environment

`MutantHunter` is an [OpenEnv](https://github.com/openreasoning/openenv) FastAPI server. Each episode hands the agent a small Python library (`mini_calendar`, `csv_normalizer`, `bloom_filter_lite`, `interval_tree`), the library's existing weak test suite, and a list of mutants the existing tests fail to catch. The agent calls tools (`read_file`, `list_tests`, `run_tests`, `get_coverage`, `get_mutation_report`) to inspect the module, then submits a pytest file. The env then:

1. Runs the new tests against the unmodified source. Every one must pass; if any fail, the agent has written broken tests, and the **no-regression gate** fires — reward goes to zero regardless of any mutant kills. This is the central anti-hack mechanism.
2. Runs the new tests against each surviving mutant. Each mutant whose suite now fails is killed by the agent's tests.
3. Computes reward from four components: mutation kill rate (the dominant term), coverage delta on lines the existing suite missed, format validity, and a parsimony penalty that prevents flooding the suite with redundant tests. The components are combined into a scalar in [0, 1], gated to zero on regression.

Mutants are precomputed per `(repo, module)` and cached, so `/reset` is fast and deterministic given a seed. The corpus is small on purpose — four libraries, hand-curated, covering enough operator coverage that the mutation engine produces meaningful surviving sets without the agent being able to memorize a single repo's structure.

## Reward-hacking defenses

If the reward function is a number an LLM is trying to maximize, every shortcut to that number is a candidate exploit. We enumerated 15 adversarial test patterns — empty submissions, `assert True` flooders, vacuous `isinstance` chains, `try/except: pass` swallowers, subprocess escapes, file-system writes, mocked imports of the module under test, regression-introducing tests masquerading as new ones, redundant duplicate-of-existing tests, and 6 more variations. Each was implemented in `evaluation/reward_hacking_tests.py` and run against the env.

All 15 score below threshold. The unifying mechanism is the no-regression gate: any test that fails on the unmutated source — including tests that *introduce* a regression by relying on side effects — collapses the reward to zero before the mutation-kill term is computed. The parsimony penalty handles flood attacks. Coverage delta refuses credit for tests that exercise no new lines. Format validation rejects unparseable submissions. The defenses compose; we do not need a separate detector for each hack class because the gate covers most of them by construction.

## Implementation details

**Stack.** Python 3.11, [OpenEnv](https://github.com/openreasoning/openenv) FastAPI server, HuggingFace `transformers` + `peft` (LoRA) + `trl` (GRPO) + `bitsandbytes` (4-bit). Training runs on HuggingFace Jobs (single GPU) on top of `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`. Logging via Weights & Biases. Eval and plots live under `evaluation/`; training under `training/`; the env server under `src/mutant_hunter/server/`.

**Mutation engine.** Pure-Python AST-based, written in-house (`src/mutant_hunter/mutation/engine.py`). Five operator classes — NCR (numeric constant), BCR (boolean constant), ROR (relational op), LCR (logical connector), AOR (arithmetic op). We avoid `mutmut`/`cosmic-ray` because they do not run cleanly on Windows in our environment, and an in-house engine lets us guarantee deterministic mutant ids per `(repo, module, seed)`. Per-module mutant sets and the corresponding "baseline-surviving" lists are precomputed via `scripts/precompute_baselines.py` and committed under `src/mutant_hunter/corpus/_baselines/` so `/reset` is O(1).

**Reward composition.** From `src/mutant_hunter/rubric/compose.py`:

```
final = no_regression_gate * (
    0.60 * mutation_kill   +
    0.20 * coverage_delta  +
    0.15 * format          +
    0.05 * parsimony
)
```

Mutation kill is the dominant term; the gate is multiplicative so a single failing test on the clean source collapses the entire reward. `format` and `parsimony` exist mainly to keep the reward landscape from being a step function in the early-training regime where `mutation_kill` is almost always zero.

**Sandboxing.** Submitted test code runs in a subprocess with RLIMITs (CPU time, address space, file size, open files) and `unshare -n` for network isolation. `open(...)`, `subprocess`, `os.system`, `eval`, `exec`, and `__import__` are also rejected pre-execution by `validate_test_code_structure` to short-circuit a wasted ~30 s env step. Defense in depth: the static check is fast and the dynamic sandbox catches whatever the static check misses.

**Models.** Cold-start was tested with `Qwen/Qwen2.5-Coder-1.5B-Instruct` (decisive failure — every rollout failed the gate). Phase 1 zero-shot baseline and GRPO target are `Qwen/Qwen2.5-Coder-7B-Instruct`. We do not use Unsloth in the final config (`--no-unsloth`) — the prebuilt torch+CUDA stack on the HF Jobs base image is incompatible with Unsloth's force-reinstall path.

**Training config (Phase 3, `scripts/run_hf_job_7b.sh`).**

| Knob | Value |
|---|---|
| `steps` | 80 |
| `rollouts_per_step` | 3 |
| `learning_rate` | 5e-6 |
| `seed` | 42 |
| LoRA `r` / `alpha` / target modules | 16 / 32 / `q_proj k_proj v_proj o_proj` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| `max_completion_length` | 2048 |
| Rollout sampling | `temperature=0.3`, `top_p=0.9`, `top_k=50` |
| `max_prompt_length` (when supported) | 2048 |

The sampling settings are deliberately tighter than the TRL default (`temperature ≈ 1.0`). The default produced too much malformed/truncated pytest, pinning rewards at zero even after the seed-routing fix described below; we drop temperature to 0.3 and grow the completion budget to 2048 tokens so the policy has both the determinism *and* the space to finish a coherent test file.

**Reward routing (the bug that ate Phase 1).** TRL's `GRPOTrainer` calls `reward_funcs(prompts=[...], completions=[...])`. Our reward function had to figure out *which episode* each prompt belonged to so it could `env.reset(seed=...)` to the same state the prompt was constructed from. The first iteration used a hash of the prompt string as a fallback seed, which silently routed every completion to an unrelated episode. The fix (`training/train_grpo.py::make_reward_fn`) is a `prompt_to_seed: dict[str, int]` populated when the prompt dataset is built and consulted on every reward call. We also strip markdown fences (` ```python ... ``` `) from the completion before constructing `Action(kind="submit_tests", test_code=...)`, since the env parses raw Python.

**Pre-flight smoke test.** Before launching a multi-hour GRPO job, we run `training/smoke_grpo_inference.py`: same model, same prompt, same sampling config, 5 sampled completions, validated by `validate_test_code_structure`. If `< 4/5` pass, GRPO will not produce useful gradient signal at scale — abort and re-tune. This is wired into `scripts/run_hf_job_7b.sh` as Phase 2.5.

**Inference-time self-correction (eval-only, not training).** `evaluation/zero_shot_distribution.py` supports `--max-retries N` (capped at 3): on a structural-validity failure or a reward-zero env step, the next attempt receives a feedback suffix describing why the previous attempt failed (e.g. "tests broke the unmodified source" → no-regression gate explanation). The env is re-`reset()` with the same seed before each retry so attempts are evaluated against an identical episode. Retry stats are dumped per-episode for analysis.

**Reproducibility.** Three scripts in `scripts/` cover the relevant grids: `run_hf_job_7b.sh` (full Phase 1 → 6 with the 7B target), `run_hf_job_demo.sh` (5-min demo with a smaller model), `run_hf_job_retry.sh` (zero-shot eval grid over `--max-retries`). All three use the same env, same corpus, same precomputed baselines, and push results to `jester1177/mutant-hunter-results` on HF Hub. The training notebook lives at `training/train_grpo.ipynb`.

## Findings

We evaluated three policies against 15 deterministic eval episodes spanning all four libraries:

| Policy | Mean Reward | p(reward > 0.3) | p(gate == 0) |
|---|---|---|---|
| Heuristic (mutation_aware) | 0.000 | 0.00 | 1.00 |
| Qwen2.5-Coder-1.5B zero-shot | 0.000 | 0.00 | 1.00 |
| Qwen2.5-Coder-7B zero-shot | 0.172 | 0.27 | 0.47 |
| Qwen2.5-Coder-7B + 80 GRPO steps | 0.172 | 0.27 | 0.47 |

The cold-start at 1.5B was decisive: every rollout failed the no-regression gate. The 7B zero-shot baseline cleared 0.17 mean reward with 27% of episodes scoring above 0.3 — a real signal, not noise, and consistent with the env being calibrated correctly.

Then GRPO did nothing. Eighty steps, flat curve, identical post-training distribution. We dug in. Two bugs in the reward-routing path were found and fixed: a seed-mismatch where the prompt's episode and the env-step's episode were unrelated, and a missing markdown-fence strip that left the env trying to parse `\`\`\`python` as Python. A smoke test with hand-fed valid pytest now produced reward 0.20 across four seeds with the correct episode routing — the reward function works.

The actual training run still produced reward zero. The remaining cause is upstream of any wiring fix: under GRPO's stochastic rollout sampling, the 7B model emits malformed pytest in essentially every rollout. Outputs ranged from length-1 truncations to full-length 1024-token runs without valid Python structure. Lowering temperature from 1.0 to 0.7 did not change the outcome. Greedy zero-shot generation works, sampled rollouts do not, and the gate honestly reports both as zero reward.

This is the finding worth documenting. Small-to-mid coder models, even ones strong enough to clear the env zero-shot, do not maintain valid pytest format under stochastic sampling. The standard remedy in the literature — warm-start SFT on the target output format, then RL — is the obvious next step, and we did not have time to run it. But the underlying claim is now backed by data on this env: pure RL from scratch on a code-generation task with a hard format gate will burn tokens on rollouts that the gate correctly rejects, and the gradient will be zero. RL'ing a code generator without an SFT stage is, on this evidence, not how this should be done.

Before falling back to Phase 2, we ran one more cheap experiment to test whether the Phase 1 ceiling was a *prompt* problem: in-context demonstrations. We mined two GOOD pytest examples per library (the corpus's own test file plus a hand-written mutation-killer suite, both validated by pytest on clean source) and three BAD examples (empty test, vacuous `assert True`, broken import). These were spliced into each prompt after the source code and before the task instruction, capped at 4000 chars, and the same 15-episode zero-shot eval was rerun on Qwen2.5-Coder-7B. The result was a *regression*: mean reward fell from 0.172 to 0.097 and the format-zero fraction rose from 0.00 to 0.27 — i.e. the BAD examples taught the model to imitate the broken-import / vacuous-assert patterns instead of avoiding them. The takeaway is consistent with the rest of the Phase 1 picture: a small coder cannot, by prompt alone, cleanly distinguish "do this" from "do not do this" in the same context window. Format-correctness for this model has to be built in at training time, not negotiated in-context — which is why Phase 2 replaces free-form generation with a constrained Mutator action space.

## Why GRPO did not move the rewards

It is worth being precise about *why* GRPO produced a flat curve, because the natural inclination is to blame the algorithm and try a different one. That inclination is wrong here.

GRPO is one of several on-policy RL algorithms that could have been used: PPO, RLOO, REINFORCE-with-baseline, ReMax, and (in a preference-pair framing) DPO/KTO/IPO. They share the same fundamental update structure: gradient ∝ ∇log π(a|s) · advantage(a, s). For GRPO specifically, the advantage is *group-relative*: `A_i = R_i − mean(R_group)`. If every rollout in a group scores zero, `mean(R) = 0`, every `A_i = 0`, and the gradient is zero. PPO and RLOO collapse the same way under the same condition — they only differ in how they estimate the advantage, not in whether they have one to estimate.

So the question is not "why did GRPO fail" but "why was the per-group reward variance zero?" The bottleneck is upstream of the algorithm — it is the **exploration prior**. RL methods assume the policy already produces non-zero reward in some fraction of rollouts. Three causes, stacked, broke that assumption here:

1. **Format-gate brittleness under stochastic sampling.** The reward composition is multiplicative on `no_regression_gate`: a single submitted test that fails on the unmodified source collapses the entire reward to zero, regardless of how good the rest of the suite is. Greedy decoding on Qwen2.5-Coder-7B produced reward 0.17 mean — the model, taken at its mode, is on the valid-pytest manifold often enough. But sampled rollouts at temperature 0.3, 0.7, and 1.0 all drifted off that manifold: hallucinated function signatures, missing imports, indentation errors, references to fixtures that do not exist, length-1 truncations. One broken test in a file is enough to nuke the episode, so essentially every sampled rollout returned zero.

2. **Reward sparsity even when the gate passes.** `mutation_kill = killed_by_new / surviving_mutants` is often `0/N` or `1/N` for the early policy. `coverage_delta` is near-zero on short test files. With group size 3 (chosen for memory reasons on a single A100), even when the gate occasionally passes, the resulting gradient estimate is tiny and high-variance. GRPO needs *intra-group* reward variance to produce a useful update, and three rollouts is too few to estimate it reliably.

3. **No SFT warm-start.** This is the standard recipe in every published RL-on-code pipeline (InstructGPT, DeepSeek-Coder-RL, CodeRL, Codex-style RL): SFT on demonstrations of valid output format first, *then* RL. SFT teaches the policy what shape of output is even allowed; RL nudges it within that valid space. Without it, RL starts in a region of output-space where most rollouts are invalid → gate → zero → gradient zero → nothing learned.

The model itself is partially implicated, but not in the way "the LLM isn't capable" suggests. Greedy zero-shot worked. The model *can* produce valid pytest. What the 7B does not have is a stable enough format prior to keep producing valid pytest under sampling — i.e. the *distribution* over its outputs is not concentrated on the valid manifold, even though its argmax is. A 32B or 70B coder model would help (the format prior gets sharper with scale) but would not fix the underlying mechanism. The fix has to come from one of:

- **SFT warm-start** — even ~30-100 examples of valid pytest in the right format would shift the policy onto the manifold before GRPO begins. Highest-impact, low-cost.
- **Soft reward shaping** — replace the multiplicative gate with `final = gate · weighted + ε · format` so rollouts that parse but miss the gate still produce non-zero gradient. Trades a small amount of anti-hack tightness for non-zero exploration signal. We sketched but did not run this.
- **Constrained decoding** — `outlines`, `lm-format-enforcer`, or vLLM grammar-guided decoding would force the rollout output through a Python AST grammar. Eliminates the format-gate-zero failure mode entirely; combinable with any RL algorithm. Requires plumbing into TRL's rollout path, which we did not have time to wire.
- **A constrained action space** — replace free-form Python with a categorical action where format failure is impossible by construction. This is exactly what Phase 2's Mutator policy does on the mutation side, and the same pattern is the cleanest long-term fix on the test-generation side too.

Switching to PPO, RLOO, or any other on-policy algorithm without addressing the exploration prior would have produced the same flat curve. Switching to DPO would have required preference pairs we did not have. The lesson is the structural one stated above: **for code-generation with a hard format gate, the choice of RL algorithm is the last knob to turn, not the first.**

## Phase 2 roadmap

The Phase 2 design (see [`docs/phase2_self_play.md`](docs/phase2_self_play.md)) addresses the format-gate problem head-on by adding a second agent that *does not generate natural language*. A **Mutator** policy emits a tuple `(operator, target_line, replacement_index)` over a small categorical action space — there is no free-form Python to malform. The Tester (current MutantHunter agent) writes pytest as before. The Mutator is rewarded by *learnability* — `4·p·(1 − p)` where `p` is the empirical kill rate of its mutation under the current Tester — so it gravitates toward mutations on the frontier of what the Tester can almost-but-not-quite catch. The two policies update in alternating GRPO rounds, and the curriculum becomes endogenous: as the Tester improves, the Mutator follows the productive-struggle frontier. The formulation borrows from *Absolute Zero* (AZR) and *R-Zero* (DeepSeek). Phase 2 also keeps the existing no-regression gate, parsimony, and coverage components on the Tester side, so all 15 reward-hack defenses transfer for free.

## Limitations

The corpus is four libraries. It was sized to make `/reset` fast and to keep the mutation engine fully deterministic, not to be a benchmark. Generalization to large real-world repos with cross-module mutants is unverified. Phase 1's training pipeline failed for the reasons documented above — we publish the env, the eval, and the findings, but not a trained checkpoint. The mutation engine is single-mutant per episode; mutation-coupling effects are out of scope. We have not yet measured how much of the Tester's signal generalizes across libraries versus how much is per-library memorization.

## Closing

- **Repo**: <https://github.com/melohub-xbit/MetaOpenEnv_MutantHunter>
- **HF Space (env)**: hosts the MutantHunter OpenEnv server — see Space metadata in `README.md`
- **HF Hub (dataset)**: corpus + per-module precomputed baselines
- **HF Hub (model)**: zero-shot eval logs for Qwen2.5-Coder-7B; no trained checkpoint published
- **W&B run**: 80-step GRPO run with full reward decomposition
- **Demo video**: <https://youtu.be/WW2YbD9o2eA>

Mutation testing is an RL reward we already had; we just had not been wiring it up. The env, defenses, and zero-shot evidence in this Phase 1 are reusable for anyone working on verified-reward RL for code. The training story is unfinished, and that part is the next chapter.
