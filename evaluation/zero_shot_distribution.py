"""Layer 6 — zero-shot reward distribution under an untrained LLM policy.

Generates one candidate test file per episode for ``--episodes`` episodes,
sampled across all 4 libraries via the env's seed-driven repo selection,
and reports the reward distribution + failure-mode fractions.

Saves raw per-episode results to
``evaluation/_results/zero_shot_distribution.json``.

Pass criteria:
    * mean reward in [0.10, 0.40]
    * fraction of episodes with reward > 0.3 ≥ 0.15
    * fraction with format == 0 < 0.30
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mutant_hunter.models import Action  # noqa: E402
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment  # noqa: E402
from training.train_grpo import build_prompt  # noqa: E402

# Anchor used to splice the demonstration block into the existing prompt.
# `build_prompt` ends with this exact line; we insert the demo block right
# before it so the user's task instruction stays the LAST thing the policy
# sees.
_TASK_ANCHOR = "## Output the pytest file content now:"
_DEMO_BUDGET_CHARS = 4000


def _format_examples(examples: list[dict], header: str, budget: int) -> tuple[str, int]:
    lines = [header]
    used = len(header) + 1
    for i, ex in enumerate(examples, 1):
        block = f"### Example {i} — {ex.get('name', '')}\n```python\n{ex['code'].rstrip()}\n```"
        if used + len(block) + 2 > budget:
            break
        lines.append(block)
        used += len(block) + 2
    return "\n".join(lines), used


def render_demonstrations(demos_for_lib: dict, budget: int = _DEMO_BUDGET_CHARS) -> str:
    """Render GOOD/BAD demos into a single string, capped at ``budget`` chars.

    Splits the budget 60/40 between good/bad so the policy always sees at
    least one of each, even with verbose examples."""
    good = list(demos_for_lib.get("good") or [])
    bad = list(demos_for_lib.get("bad") or [])
    good_budget = int(budget * 0.6)
    bad_budget = budget - good_budget
    parts: list[str] = ["## In-context demonstrations"]
    if good:
        good_str, _ = _format_examples(
            good, "Here are examples of GOOD tests that catch bugs:", good_budget
        )
        parts.append(good_str)
    if bad:
        bad_str, _ = _format_examples(
            bad, "Here are examples of BAD tests to AVOID:", bad_budget
        )
        parts.append(bad_str)
    parts.append("Now write your tests, following the GOOD pattern (mirror exact "
                 "behavior of the source — never guess return values or exception types).")
    return "\n\n".join(parts)


def build_prompt_with_demos(obs, demos: dict | None) -> str:
    base = build_prompt(obs)
    if not demos:
        return base
    repo = obs.repo_name
    if repo not in demos:
        return base
    demo_block = render_demonstrations(demos[repo])
    if _TASK_ANCHOR in base:
        return base.replace(_TASK_ANCHOR, demo_block + "\n\n" + _TASK_ANCHOR)
    # Fallback: append at end if anchor missing (build_prompt was changed).
    return base + "\n\n" + demo_block

RESULTS_DIR = ROOT / "evaluation" / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Inference-time self-correction ---------------------------------------

_FORBIDDEN_PATTERNS = (
    "subprocess",
    "os.system",
    "eval(",
    "exec(",
    "__import__",
    "socket.",
    "open(",  # File access at module-import time is a no-regression risk;
              # the env's sandbox blocks it but a forbidden-pattern hint
              # short-circuits a wasted env step.
)


def validate_test_code_structure(text: str) -> str | None:
    """Return None if ``text`` is a plausible pytest file, else a human-
    readable error message describing why it isn't.

    This runs BEFORE the env step so a structurally invalid completion can be
    fed back to the model without paying the env-eval cost (~30s+)."""
    if not text or not text.strip():
        return "empty output"
    if "def test_" not in text:
        return "no test_* function found"
    try:
        ast.parse(text)
    except SyntaxError as e:
        return f"syntax error: {e.msg}"
    for f in _FORBIDDEN_PATTERNS:
        if f in text:
            return f"forbidden pattern: {f}"
    return None


def extract_failure_reason(components: dict, gate: float | None) -> str:
    """Map env reward components to a single hint to feed back to the model."""
    if gate == 0 or gate == 0.0:
        return ("Your tests broke the unmodified source code "
                "(no_regression_gate failed). Make sure each test passes on "
                "the original implementation as shown in the prompt.")
    if components.get("format", 1.0) == 0:
        return "Output had format issues. Use valid pytest syntax with `assert` statements."
    if components.get("mutation_kill", 0.0) == 0:
        return "Tests didn't kill any mutations. Add edge-case tests targeting boundaries."
    return "Tests scored 0 — re-read the source carefully and try again."


def build_retry_suffix(error_history: list[tuple[str, str]]) -> str:
    """Construct a feedback addendum that nudges the model toward a correction
    on its next attempt. Only the most recent failure is shown — earlier
    failures pile up tokens without adding signal."""
    last_kind, last_msg = error_history[-1]
    parts = [
        "",
        "## Your previous attempt failed",
        f"Reason ({last_kind}): {last_msg}",
        "",
        "Please write a complete pytest test file with valid Python syntax "
        "and at least one function named `def test_*`.",
        "Output ONLY Python code in a ```python ... ``` block; no prose, no "
        "explanation.",
        "",
    ]
    return "\n".join(parts)


def _generate_completion(model, tokenizer, prompt: str, device: str,
                         max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        templated = prompt
    inputs = tokenizer(templated, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    import torch  # local import — torch is imported at top of main()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def run_episode_with_retries(
    *,
    env,
    base_prompt: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    max_retries: int,
) -> dict:
    """Generate, validate, submit, retry. Returns a dict with keys:
    test_code, full_completion, reward, components, no_regression_gate,
    retries_used, error_history, status, gen_elapsed_s, env_eval_elapsed_s.
    The env has already been reset for the current episode by the caller."""
    error_history: list[tuple[str, str]] = []
    last_full_completion = ""
    last_test_code = ""
    last_components: dict = {}
    last_gate: float | None = None
    last_md: dict = {}
    total_gen = 0.0
    total_eval = 0.0

    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if error_history:
            prompt = base_prompt + build_retry_suffix(error_history)

        t0 = time.time()
        try:
            full_completion = _generate_completion(
                model, tokenizer, prompt, device, max_new_tokens
            )
        except Exception as exc:
            error_history.append(("generation_error", f"{type(exc).__name__}: {exc}"))
            total_gen += time.time() - t0
            continue
        total_gen += time.time() - t0
        last_full_completion = full_completion
        test_code = _strip_markdown_fences(full_completion)
        last_test_code = test_code

        validation_error = validate_test_code_structure(test_code)
        if validation_error:
            error_history.append(("structural", validation_error))
            if attempt == max_retries:
                last_components = {"mutation_kill": 0.0, "coverage_delta": 0.0,
                                   "format": 0.0, "parsimony": 0.0}
                last_gate = 0.0
            continue

        try:
            action = Action(kind="submit_tests", test_code=test_code)
        except Exception as exc:
            error_history.append(("pydantic", f"{type(exc).__name__}: {exc}"))
            if attempt == max_retries:
                last_components = {"mutation_kill": 0.0, "coverage_delta": 0.0,
                                   "format": 0.0, "parsimony": 0.0}
                last_gate = 0.0
            continue

        t1 = time.time()
        obs2 = env.step(action)
        total_eval += time.time() - t1
        md = obs2.metadata or {}
        last_md = md
        components = dict(md.get("components") or {})
        gate = md.get("no_regression_gate")
        reward = float(obs2.reward or 0.0)
        last_components = components
        last_gate = gate

        if reward > 0:
            return {
                "test_code": test_code,
                "full_completion": full_completion,
                "reward": reward,
                "components": components,
                "no_regression_gate": gate,
                "metadata": md,
                "retries_used": attempt,
                "error_history": error_history,
                "status": "success_after_retries" if attempt > 0 else "success",
                "gen_elapsed_s": round(total_gen, 2),
                "env_eval_elapsed_s": round(total_eval, 2),
            }

        # reward == 0: extract why and try again (or stop if we're out).
        reason = extract_failure_reason(components, gate)
        error_history.append(("env_rejected", reason))
        if attempt == max_retries:
            return {
                "test_code": test_code,
                "full_completion": full_completion,
                "reward": 0.0,
                "components": components,
                "no_regression_gate": gate,
                "metadata": md,
                "retries_used": attempt,
                "error_history": error_history,
                "status": "failed_all_retries",
                "gen_elapsed_s": round(total_gen, 2),
                "env_eval_elapsed_s": round(total_eval, 2),
            }
        # else: loop back with feedback

    # All attempts exhausted without ever submitting successfully.
    return {
        "test_code": last_test_code,
        "full_completion": last_full_completion,
        "reward": 0.0,
        "components": last_components or {"mutation_kill": 0.0, "coverage_delta": 0.0,
                                          "format": 0.0, "parsimony": 0.0},
        "no_regression_gate": last_gate if last_gate is not None else 0.0,
        "metadata": last_md,
        "retries_used": max_retries,
        "error_history": error_history,
        "status": "exhausted_retries",
        "gen_elapsed_s": round(total_gen, 2),
        "env_eval_elapsed_s": round(total_eval, 2),
    }


def _strip_markdown_fences(text: str) -> str:
    """Pull out the first ```python|py block, else the first plain ``` block, else the raw text."""
    fence_re = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
    m = fence_re.search(text)
    if m:
        return m.group(1).strip("\n")
    # Fall back: if the model emitted an unfenced "Output the pytest file" body,
    # try to keep everything from the first def/import/class line onward.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HF model id. Falls back to a smaller model if downloading the default fails.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' | 'cuda' | 'cpu' — auto picks cuda when available else cpu",
    )
    ap.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="If set, load a PEFT LoRA adapter from this path on top of --model.",
    )
    ap.add_argument(
        "--use-demonstrations",
        type=str,
        default=None,
        help="Path to demonstrations.json (from training/mine_demonstrations.py). "
             "When set, GOOD/BAD examples are spliced into each prompt after the "
             "source code and before the task instruction, capped at "
             f"{_DEMO_BUDGET_CHARS} chars.",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Inference-time self-correction: number of additional attempts "
             "(beyond the first) the model gets, with a feedback suffix "
             "describing why the previous attempt failed. 0 reproduces the "
             "single-shot baseline. Capped at 3.",
    )
    ap.add_argument(
        "--retry-output",
        type=str,
        default=None,
        help="If set, dump per-episode retry stats (status, retries_used, "
             "error_history) to this JSON path for analysis.",
    )
    args = ap.parse_args()
    if args.max_retries < 0 or args.max_retries > 3:
        print(f"[layer6] FAIL — --max-retries must be in [0, 3], got {args.max_retries}")
        return 1

    demos: dict | None = None
    if args.use_demonstrations:
        demo_path = Path(args.use_demonstrations)
        if not demo_path.exists():
            print(f"[layer6] FAIL — --use-demonstrations path not found: {demo_path}")
            return 1
        demos = json.loads(demo_path.read_text(encoding="utf-8"))
        print(f"[layer6] loaded demonstrations for {list(demos.keys())} "
              f"from {demo_path}", flush=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"[layer6] FAIL — transformers/torch not installed: {exc}")
        return 1

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[layer6] device={device} model={args.model}", flush=True)
    if device == "cpu":
        print(
            "[layer6] WARN — CPU inference of a 1B+ model with 1024 new tokens is "
            "extremely slow (~minutes per episode).",
            flush=True,
        )

    print("[layer6] loading model ...", flush=True)
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        if args.lora_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                print(f"[layer6] FAIL — --lora-path set but peft not installed: {exc}")
                return 1
            print(f"[layer6] attaching LoRA adapter from {args.lora_path} ...", flush=True)
            model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.to(device)
        model.eval()
    except Exception as exc:
        print(f"[layer6] FAIL — could not load model {args.model}: {exc}")
        traceback.print_exc()
        return 1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[layer6] model loaded in {time.time() - t0:.1f}s", flush=True)

    env = MutantHunterEnvironment()
    rows: list[dict] = []
    try:
        for i in range(args.episodes):
            seed = args.seed_start + i
            obs = env.reset(seed=seed)
            prompt = build_prompt_with_demos(obs, demos)

            result = run_episode_with_retries(
                env=env,
                base_prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=args.max_new_tokens,
                max_retries=args.max_retries,
            )

            md = result.get("metadata") or {}
            row: dict = {
                "episode": i,
                "seed": seed,
                "repo": obs.repo_name,
                "module": obs.module_path,
                "gen_elapsed_s": result["gen_elapsed_s"],
                "env_eval_elapsed_s": result["env_eval_elapsed_s"],
                "raw_completion_chars": len(result["full_completion"]),
                "test_code_chars": len(result["test_code"]),
                "final_reward": result["reward"],
                "components": result["components"],
                "no_regression_gate": result["no_regression_gate"],
                "killed_by_new_only": md.get("killed_by_new_only"),
                "new_coverage": md.get("new_coverage"),
                "new_tests_pass_clean": md.get("new_tests_pass_clean"),
                "status": result["status"],
                "retries_used": result["retries_used"],
                "error_history": result["error_history"],
            }
            rows.append(row)
            print(
                f"  ep{i} seed={seed} repo={obs.repo_name} reward={row['final_reward']:.4f} "
                f"fmt={(row['components'] or {}).get('format', 0.0):.2f} "
                f"gate={row['no_regression_gate']} "
                f"retries={row['retries_used']} "
                f"status={row['status']} "
                f"gen={row['gen_elapsed_s']}s eval={row['env_eval_elapsed_s']}s",
                flush=True,
            )
    finally:
        env.close()

    rewards = [r["final_reward"] for r in rows]
    n = len(rewards)
    mean_r = statistics.mean(rewards) if rewards else 0.0
    std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    min_r = min(rewards) if rewards else 0.0
    max_r = max(rewards) if rewards else 0.0
    p_above_03 = sum(1 for r in rewards if r > 0.3) / n if n else 0.0
    fmt_zero = sum(1 for r in rows if r["components"].get("format", 0.0) == 0.0) / n if n else 0.0
    gate_zero = sum(1 for r in rows if r.get("no_regression_gate") == 0.0) / n if n else 0.0

    n_with_retries = sum(1 for r in rows if int(r.get("retries_used") or 0) > 0)
    n_success_after_retry = sum(
        1 for r in rows if r.get("status") == "success_after_retries"
    )
    n_exhausted = sum(
        1 for r in rows if r.get("status") in ("failed_all_retries", "exhausted_retries")
    )

    summary = {
        "n_episodes": n,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "min_reward": min_r,
        "max_reward": max_r,
        "fraction_reward_gt_0.3": p_above_03,
        "fraction_format_zero": fmt_zero,
        "fraction_regression_gate_zero": gate_zero,
        "max_retries": args.max_retries,
        "n_episodes_with_retries": n_with_retries,
        "n_episodes_success_after_retry": n_success_after_retry,
        "n_episodes_failed_all_retries": n_exhausted,
    }

    out_path = RESULTS_DIR / "zero_shot_distribution.json"
    out_path.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n", encoding="utf-8")

    if args.retry_output:
        retry_dump = {
            "summary": summary,
            "per_episode": [
                {
                    "episode": r["episode"],
                    "seed": r["seed"],
                    "repo": r["repo"],
                    "status": r["status"],
                    "retries_used": r["retries_used"],
                    "final_reward": r["final_reward"],
                    "error_history": r["error_history"],
                }
                for r in rows
            ],
        }
        Path(args.retry_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.retry_output).write_text(
            json.dumps(retry_dump, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[layer6] retry stats written to {args.retry_output}")

    print()
    print(f"[layer6] n={n} mean={mean_r:.4f} std={std_r:.4f} min={min_r:.4f} max={max_r:.4f}")
    print(f"[layer6] p(reward > 0.3) = {p_above_03:.3f}")
    print(f"[layer6] p(format == 0)  = {fmt_zero:.3f}")
    print(f"[layer6] p(gate   == 0)  = {gate_zero:.3f}")
    if args.max_retries > 0:
        print(f"[layer6] retries: max={args.max_retries} "
              f"used={n_with_retries} "
              f"recovered_after_retry={n_success_after_retry} "
              f"failed_all={n_exhausted}")

    issues: list[str] = []
    if not (0.10 <= mean_r <= 0.40):
        issues.append(f"mean reward {mean_r:.4f} outside [0.10, 0.40]")
    if p_above_03 < 0.15:
        issues.append(f"p(reward>0.3) = {p_above_03:.3f} < 0.15")
    if fmt_zero >= 0.30:
        issues.append(f"p(format=0) = {fmt_zero:.3f} ≥ 0.30")

    if issues:
        print("[layer6] FAIL — issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("[layer6] PASS — distribution within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
