# Phase 2: Co-Evolutionary Self-Play for Test Generation

Phase 1 established that mutation-score-as-reward is a learnable, hard-to-hack signal — heuristic baselines score 0.0, zero-shot 7B clears 0.17 mean reward, and the no-regression gate plus the 15-case adversarial suite block known reward hacks. The training pipeline itself, however, never produced gradient: every GRPO rollout from a 1.5B–7B coder model was malformed pytest, which the env correctly rejected. The lesson: putting a free-form natural-language generator inside the RL loop is the wrong inductive bias for this signal. Phase 2 fixes the inductive bias by adding a second agent that does *not* generate natural language.

## Methodology

We train two policies in alternating GRPO updates:

- **Tester** — the existing MutantHunter agent. Given a module, writes pytest. Rewarded by mutation kill rate against the *current* mutant population (computed by the env, exactly as in Phase 1).
- **Mutator** — a new agent that proposes a single mutation from a constrained action grammar. Rewarded by *learnability*: the reward peaks when the Tester gets the mutation right roughly half the time.

A round looks like: freeze the Mutator, run a batch of GRPO updates on the Tester against a sampled mutant population. Then freeze the Tester, run a batch of GRPO updates on the Mutator. Repeat. The two agents co-evolve: the Mutator learns to find mutations that are hard but not impossible for the current Tester; the Tester learns to close those gaps.

## Mutator action grammar

Instead of emitting source patches as free text, the Mutator emits a tuple drawn from a finite action space:

```
(operator, target_line, replacement_index)
```

- `operator ∈ {AOR, ROR, LCR, BCR, NCR, SCR, BOUNDARY}` — arithmetic / relational / logical-connector / boolean-constant / numeric-constant / string-constant / boundary-shift.
- `target_line` — an integer index into the module's mutable AST positions (precomputed at episode start).
- `replacement_index` — index into the operator-specific replacement table (e.g., for `AOR` the table is `{+, -, *, /, //, %, **}`).

The env decodes the tuple into a concrete patch deterministically. There is no free-form Python to be malformed: the Mutator's policy head is a categorical over a small, fully enumerated space. This is the part of the design that directly addresses Phase 1's blocker.

## Learnability reward

Given a sampled mutation, run the Tester's current test suite against it across `K` rollouts. Let `p` be the fraction of rollouts in which the Tester's tests *kill* the mutant. The Mutator's reward is:

```
R_mutator = 4 * p * (1 - p)
```

This is a parabola peaking at `p = 0.5`. Mutations the Tester always kills (`p = 1`) and mutations it never kills (`p = 0`) both score zero. The Mutator is therefore pushed toward the *productive struggle zone* — mutations on the frontier of the Tester's current capability. As the Tester improves, the productive frontier moves, and the Mutator follows.

The Tester reward is unchanged from Phase 1 (mutation kill rate plus parsimony, coverage delta, format, gated by no-regression).

## Training loop

```mermaid
flowchart LR
    subgraph Round[One co-evolution round]
        M[Mutator policy] -->|sample mutations| Pool[Mutant pool]
        Pool -->|episode resets| T[Tester policy]
        T -->|generate tests| Env[MutantHunter env]
        Env -->|kill rate per mutant| MR[Mutator reward<br/>4·p·(1−p)]
        Env -->|mutation score| TR[Tester reward]
        MR -->|GRPO update<br/>(Mutator turn)| M
        TR -->|GRPO update<br/>(Tester turn)| T
    end
```

## Why this addresses Phase 1's findings

1. **No malformed rollouts.** The Mutator emits structured tuples; the Tester's pytest output remains the only free-form generation, and Phase 2 will warm-start it via SFT on the few-shot completions before GRPO.
2. **Automatic curriculum.** Phase 1's flat reward curve was partly a corpus problem: many baseline mutants are trivially killed by even shallow tests, so signal was noisy. Learnability reward pushes the Mutator to surface the hard-but-killable mutants the Tester actually learns from.
3. **Defenses transfer.** The Mutator's action space cannot encode any of the 15 reward-hacks (no subprocess, no `assert True`, no test deletion) — those all live in the Tester's output space, where the existing no-regression gate already blocks them.

## References

- *Absolute Zero: Reinforced Self-play Reasoning with Zero Data* (AZR) — [arXiv:2403.02543](https://arxiv.org/abs/2403.02543). Source of the self-play formulation we adapt.
- *R-Zero* (DeepSeek) — co-evolution of solver and proposer agents in mathematical reasoning; the learnability-reward shape `4·p·(1−p)` is directly borrowed.
- Phase 1 results and findings: [`README.md`](../README.md#results).
