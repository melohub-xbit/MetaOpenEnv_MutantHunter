"""GRPO trainer wired to the MutantHunter env.

Pipeline
--------

For each rollout step:

1. Sample a (repo, module) episode from the env via reset(seed).
2. Build a prompt containing the agent system prompt, module summary,
   existing tests names, and a "submit a pytest file" instruction.
3. Have the policy (Hugging Face transformers + optional Unsloth /
   bitsandbytes 4-bit + PEFT LoRA) generate a candidate ``test_code``.
4. Call env.step(submit_tests) and read the scalar reward from the
   resulting Observation.
5. GRPOTrainer (TRL) uses (prompt, candidate, reward) tuples to update
   the policy.

The script is fully end-to-end: import-clean, instantiable, and runs N
optimization steps. It does NOT include speculative tricks like reward
rescaling or curriculum scheduling — those are Phase 2 territory.

Dependency note
---------------

The training extras are NOT in the base install. Install via:

    pip install -e ".[training]"

For 4-bit / Unsloth path:

    pip install -e ".[training,unsloth]"

Unsloth currently runs best on Linux + CUDA. On macOS / Windows fall back
to the plain transformers + PEFT path by passing ``--no-unsloth``.

Smoke run (10 optimization steps, no actual learning expected, just
verifies the loop executes end-to-end):

    python training/train_grpo.py --steps 10 --rollouts-per-step 2 \
        --base-model sshleifer/tiny-gpt2 --no-unsloth

Real run example (Qwen3 4B, 4-bit, LoRA):

    python training/train_grpo.py --steps 1000 --rollouts-per-step 4 \
        --base-model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Repo root on sys.path so `from training.prompts import ...` works when this
# file is run as a script (python training/train_grpo.py) and not just as a
# module (python -m training.train_grpo).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mutant_hunter.corpus import dotted_to_workspace_relpath, repo_dir  # noqa: E402
from mutant_hunter.models import Action, Observation  # noqa: E402
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment  # noqa: E402
from training.prompts import SYSTEM_PROMPT, render_few_shot  # noqa: E402


@dataclass
class TrainingConfig:
    base_model: str
    steps: int
    rollouts_per_step: int
    learning_rate: float
    use_unsloth: bool
    use_4bit: bool
    use_lora: bool
    lora_rank: int
    max_new_tokens: int
    seed: int
    output_dir: Path
    wandb_project: str | None


# --- Completion post-processing --------------------------------------------


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Pull out the first ```python|py block, else the first plain ``` block,
    else the raw text. Mirrors evaluation/zero_shot_distribution.py so the
    GRPO reward path sees the same test_code the zero-shot eval path sees."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip("\n")
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if (
            ln.startswith("import ")
            or ln.startswith("from ")
            or ln.startswith("def test_")
            or ln.startswith("class Test")
        ):
            return "\n".join(lines[i:])
    return text


# --- Prompt construction ----------------------------------------------------


def _read_module_source(obs: Observation) -> str | None:
    """Best-effort read of the target module's full source from the local
    corpus. Returns None if the file cannot be located or read; callers
    fall back to ``obs.module_summary`` in that case."""
    if not obs.repo_name or not obs.module_path:
        return None
    try:
        path = repo_dir(obs.repo_name) / dotted_to_workspace_relpath(obs.module_path)
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, ValueError):
        return None


def build_prompt(obs: Observation) -> str:
    full_source = _read_module_source(obs)
    if full_source is not None:
        source_section = ["## Module source", "```python", full_source.rstrip(), "```"]
    else:
        source_section = ["## Module summary", obs.module_summary or "(no summary)"]

    parts = [
        SYSTEM_PROMPT,
        "",
        "## Few-shot examples",
        render_few_shot(),
        "",
        "## Live task",
        f"Module: {obs.module_path}",
        f"Repo:   {obs.repo_name}",
        f"Baseline mutation score: {obs.baseline_mutation_score:.3f}",
        "",
        *source_section,
        "",
        "## Existing tests",
    ]
    parts.extend(f"  - {n}" for n in (obs.existing_tests or []))
    parts.append("")
    parts.append("## Tool history this episode")
    if obs.history:
        for h in obs.history:
            parts.append(f"### {h.tool}")
            parts.append(h.output[:1000])
    else:
        parts.append("(none)")
    parts.append("")
    parts.append("## Output the pytest file content now:")
    return "\n".join(parts)


# --- Model loading ----------------------------------------------------------


def _try_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def load_policy(cfg: TrainingConfig):
    """Return (model, tokenizer). Prefer Unsloth+4bit when available."""
    transformers = _try_import("transformers")
    if transformers is None:
        raise RuntimeError(
            "transformers is required. Install training extras: "
            "`pip install -e \".[training]\"`"
        )

    if cfg.use_unsloth:
        unsloth = _try_import("unsloth")
        if unsloth is None:
            raise RuntimeError(
                "Unsloth requested but not installed. Either "
                "`pip install -e \".[training,unsloth]\"` or pass --no-unsloth."
            )
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=4096,
            load_in_4bit=cfg.use_4bit,
        )
        if cfg.use_lora:
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha=cfg.lora_rank * 2,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        return model, tokenizer

    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {}
    if cfg.use_4bit:
        bnb = _try_import("bitsandbytes")
        if bnb is None:
            raise RuntimeError("--4bit requested but bitsandbytes not installed")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
        )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)

    if cfg.use_lora:
        peft = _try_import("peft")
        if peft is None:
            raise RuntimeError("--lora requested but peft not installed")
        from peft import LoraConfig, get_peft_model, TaskType
        cfg_lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            target_modules=["c_attn"] if "gpt2" in cfg.base_model.lower() else None,
        )
        model = get_peft_model(model, cfg_lora)
    return model, tokenizer


# --- Reward function (per-prompt callable used by GRPOTrainer) -------------


def make_reward_fn(
    env: MutantHunterEnvironment,
    prompt_to_seed: dict[str, int],
    debug_rollouts: int = 4,
):
    """Return a callable suitable for TRL's GRPOTrainer ``reward_funcs``.

    GRPOTrainer calls reward_funcs(prompts=[...], completions=[...]) and
    expects a list of floats. Each completion is a candidate test file;
    we score it by stepping the env once.

    ``prompt_to_seed`` maps the prompt string back to the seed that built
    it via ``make_prompt_dataset``, so env.reset() targets the SAME episode
    the model was prompted on. Falling back to a hash-derived seed would
    evaluate completions against an unrelated episode (the original bug
    that pinned all rewards to 0).
    """
    debug_remaining = {"n": debug_rollouts}

    def reward_fn(prompts: list[str], completions: list[str], **_: Any) -> list[float]:
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            mapped_seed = prompt_to_seed.get(prompts[i])
            if mapped_seed is None:
                seed = (hash(prompts[i]) + i) & 0xFFFFFFFF
                seed_source = "fallback-hash"
            else:
                seed = mapped_seed
                seed_source = "dataset"
            env.reset(seed=seed)

            test_code = _strip_markdown_fences(completion)
            had_fence = test_code != completion

            try:
                action = Action(kind="submit_tests", test_code=test_code)
            except Exception as exc:
                if debug_remaining["n"] > 0:
                    print(
                        f"[reward_fn] DEBUG seed={seed} ({seed_source}) "
                        f"REJECTED at Action(): {type(exc).__name__}: {exc} "
                        f"raw_len={len(completion)} stripped_len={len(test_code)} "
                        f"had_fence={had_fence}",
                        flush=True,
                    )
                    debug_remaining["n"] -= 1
                rewards.append(0.0)
                continue

            obs = env.step(action)
            reward = float(obs.reward or 0.0)
            if debug_remaining["n"] > 0:
                md = getattr(obs, "metadata", None) or {}
                print(
                    f"[reward_fn] DEBUG seed={seed} ({seed_source}) reward={reward:.4f} "
                    f"raw_len={len(completion)} stripped_len={len(test_code)} "
                    f"had_fence={had_fence} "
                    f"components={md.get('components')} "
                    f"gate={md.get('no_regression_gate')} "
                    f"status={md.get('status')}",
                    flush=True,
                )
                debug_remaining["n"] -= 1
            rewards.append(reward)
        return rewards

    return reward_fn


# --- Dataset --------------------------------------------------------------


def make_prompt_dataset(env: MutantHunterEnvironment, n: int, seed_start: int):
    """Build a prompt dataset and a prompt_str -> seed map.

    The map is the canonical way to recover, in the reward function, which
    episode each prompt was constructed from. Without it, the reward
    function has no idea which (repo, module) the completion was meant for.
    """
    datasets = _try_import("datasets")
    if datasets is None:
        raise RuntimeError("datasets is required for GRPOTrainer prompt iteration")
    rows: list[dict[str, str]] = []
    prompt_to_seed: dict[str, int] = {}
    for i in range(n):
        seed = seed_start + i
        obs = env.reset(seed=seed)
        prompt = build_prompt(obs)
        rows.append({"prompt": prompt})
        prompt_to_seed[prompt] = seed
    return datasets.Dataset.from_list(rows), prompt_to_seed


# --- Main loop ------------------------------------------------------------


def run(cfg: TrainingConfig) -> int:
    random.seed(cfg.seed)

    report_to: list[str] = []
    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        report_to = ["wandb"]

    trl = _try_import("trl")
    if trl is None:
        raise RuntimeError(
            "trl is required. Install training extras: `pip install -e \".[training]\"`"
        )

    GRPOConfig = trl.GRPOConfig
    GRPOTrainer = trl.GRPOTrainer

    env = MutantHunterEnvironment()
    model, tokenizer = load_policy(cfg)

    # Build a prompt dataset large enough to feed `steps * rollouts_per_step`
    # rollouts. The dataset cycles internally per TRL semantics.
    n_prompts = max(8, cfg.steps * max(1, cfg.rollouts_per_step // 2))
    train_dataset, prompt_to_seed = make_prompt_dataset(
        env, n=n_prompts, seed_start=cfg.seed
    )

    grpo_kwargs: dict[str, Any] = dict(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=max(1, cfg.rollouts_per_step),
        num_generations=max(2, cfg.rollouts_per_step),
        learning_rate=cfg.learning_rate,
        max_steps=cfg.steps,
        max_completion_length=cfg.max_new_tokens,
        # Rollout sampling: keep generations close to what the zero-shot eval
        # used. The TRL default (~1.0) produced too much malformed/truncated
        # pytest, pinning rewards at 0 even after the seed-routing fix.
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        bf16=False,
        fp16=False,
        logging_steps=1,
        save_steps=max(1, cfg.steps),
        save_strategy="steps",
        report_to=report_to,
        seed=cfg.seed,
    )
    # Older TRL versions accept max_prompt_length; newer ones drop it.
    import inspect as _inspect
    _accepted = set(_inspect.signature(GRPOConfig.__init__).parameters.keys())
    if "max_prompt_length" in _accepted:
        grpo_kwargs["max_prompt_length"] = 2048
    args = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_dataset,
        reward_funcs=[make_reward_fn(env, prompt_to_seed)],
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir / "final"))
    print(f"Training complete. Saved to {cfg.output_dir / 'final'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default="sshleifer/tiny-gpt2",
                    help="HF model id or local path.")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rollouts-per-step", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default="./training/_runs/grpo")
    ap.add_argument("--wandb-project", type=str, default=None,
                    help="Weights & Biases project name. If set, reports to wandb.")
    ap.add_argument("--no-unsloth", action="store_true",
                    help="Skip the Unsloth path (use plain transformers + peft).")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Skip 4-bit quantization (default ON when CUDA + bitsandbytes available).")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg = TrainingConfig(
        base_model=args.base_model,
        steps=args.steps,
        rollouts_per_step=args.rollouts_per_step,
        learning_rate=args.learning_rate,
        use_unsloth=not args.no_unsloth,
        use_4bit=not args.no_4bit,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        output_dir=out,
        wandb_project=args.wandb_project,
    )

    # Persist the final config so eval_harness / write-up can reference it.
    (out / "config.json").write_text(
        json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, indent=2),
        encoding="utf-8",
    )

    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
