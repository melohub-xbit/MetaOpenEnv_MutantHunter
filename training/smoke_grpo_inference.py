"""Pre-flight smoke test for GRPO rollouts.

Runs the SAME (model, prompt, sampling-config) combo GRPOTrainer would use
during rollouts, generates a small batch of completions, and reports how
many pass ``validate_test_code_structure``. The goal is to catch
malformed-pytest / truncation issues BEFORE paying for a full GRPO run.

Usage:

    # Default: Qwen 2.5-Coder 7B, 5 completions, CPU-friendly fallback
    python training/smoke_grpo_inference.py

    # On a GPU box:
    python training/smoke_grpo_inference.py --device cuda

Pass criterion (printed only — no exit code gate):
    >= 4 / 5 completions pass validate_test_code_structure

If you see 1/5 pass, the GRPO sampling config will not give you positive
reward signal at scale — fix the prompt, the model, or the sampling knobs
before launching Phase 3.
"""

from __future__ import annotations

import argparse
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

from evaluation.zero_shot_distribution import validate_test_code_structure  # noqa: E402
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment  # noqa: E402
from training.train_grpo import _strip_markdown_fences, build_prompt  # noqa: E402


def _generate_one(model, tokenizer, prompt: str, device: str,
                  max_new_tokens: int, temperature: float, top_p: float,
                  seed: int) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        templated = prompt
    inputs = tokenizer(
        templated, return_tensors="pt", truncation=True, max_length=4096
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-flight smoke test: generate N completions with the "
                    "GRPO rollout sampling config and report validation pass rate."
    )
    ap.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HF model id. Same as Phase 3 base model by default.",
    )
    ap.add_argument(
        "--n-completions",
        type=int,
        default=5,
        help="Number of completions to sample for the same prompt.",
    )
    ap.add_argument(
        "--episode-seed",
        type=int,
        default=42,
        help="Seed passed to env.reset() to pick the (repo, module) episode "
             "the prompt is built from.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' | 'cuda' | 'cpu'",
    )
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"[smoke] FAIL — transformers/torch not installed: {exc}")
        return 1

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[smoke] device={device} model={args.model}", flush=True)
    if device == "cpu":
        print(
            "[smoke] WARN — CPU inference of a 7B model with "
            f"{args.max_new_tokens} new tokens will take many minutes per "
            "completion. Consider --device cuda or a smaller --model.",
            flush=True,
        )

    print("[smoke] building env + prompt ...", flush=True)
    env = MutantHunterEnvironment()
    obs = env.reset(seed=args.episode_seed)
    prompt = build_prompt(obs)
    print(
        f"[smoke] episode seed={args.episode_seed} repo={obs.repo_name} "
        f"module={obs.module_path} prompt_chars={len(prompt)}",
        flush=True,
    )

    print("[smoke] loading model ...", flush=True)
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model = model.to(device)
        model.eval()
    except Exception as exc:
        print(f"[smoke] FAIL — could not load model {args.model}: {exc}")
        traceback.print_exc()
        return 1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[smoke] model loaded in {time.time() - t0:.1f}s", flush=True)

    print(
        f"[smoke] generating {args.n_completions} completions @ "
        f"temp={args.temperature} top_p={args.top_p} "
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
    )

    n_valid = 0
    sample_valid: tuple[int, str] | None = None
    sample_invalid: tuple[int, str, str] | None = None
    per_attempt: list[dict] = []

    for i in range(args.n_completions):
        t1 = time.time()
        try:
            full = _generate_one(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.episode_seed * 1000 + i,
            )
        except Exception as exc:
            print(f"  [{i}] generation_error: {type(exc).__name__}: {exc}",
                  flush=True)
            per_attempt.append({"i": i, "valid": False,
                                "reason": f"generation_error: {exc}"})
            continue
        gen_s = time.time() - t1

        test_code = _strip_markdown_fences(full)
        err = validate_test_code_structure(test_code)
        valid = err is None
        if valid:
            n_valid += 1
            if sample_valid is None:
                sample_valid = (i, test_code)
        else:
            if sample_invalid is None:
                sample_invalid = (i, test_code, err or "unknown")
        per_attempt.append({
            "i": i,
            "valid": valid,
            "reason": err,
            "raw_chars": len(full),
            "stripped_chars": len(test_code),
            "gen_s": round(gen_s, 1),
        })
        print(
            f"  [{i}] valid={valid} reason={err!r} raw={len(full)}c "
            f"stripped={len(test_code)}c gen={gen_s:.1f}s",
            flush=True,
        )

    print()
    print(f"[smoke] RESULT: {n_valid}/{args.n_completions} completions "
          f"pass validate_test_code_structure")
    print()

    if sample_valid is not None:
        i, code = sample_valid
        snippet = code[:1000]
        ellipsis = "" if len(code) <= 1000 else "\n... (truncated)"
        print(f"[smoke] sample VALID completion (i={i}):")
        print("---8<---")
        print(snippet + ellipsis)
        print("---8<---")
    else:
        print("[smoke] no valid completion to show")

    print()
    if sample_invalid is not None:
        i, code, reason = sample_invalid
        snippet = code[:1000]
        ellipsis = "" if len(code) <= 1000 else "\n... (truncated)"
        print(f"[smoke] sample INVALID completion (i={i}, reason={reason!r}):")
        print("---8<---")
        print(snippet + ellipsis)
        print("---8<---")
    else:
        print("[smoke] no invalid completion to show (all passed!)")

    print()
    pass_rate = n_valid / args.n_completions if args.n_completions else 0.0
    if pass_rate >= 0.8:
        print(f"[smoke] PASS — {pass_rate:.0%} valid. GRPO config looks good.")
        return 0
    print(
        f"[smoke] WARN — only {pass_rate:.0%} valid. GRPO will likely give "
        "near-zero reward signal at this rate. Re-tune sampling or prompt "
        "before launching Phase 3."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
