"""Local CPU probe to test the hypothesis that the sparse module_summary
is what's killing Layer 6's no_regression_gate.

Runs ep0 (bloom_filter_lite) twice against the env:
    A) prompt with the current sparse summary (`class BloomFilter: ...`)
    B) prompt with the actual module source as the summary

Reports reward + components for each. If A gives gate=0 and B gives gate=1
(or any non-zero reward), hypothesis is confirmed.

Uses Qwen2.5-Coder-0.5B-Instruct (cached locally), greedy decoding,
max_new_tokens=512. The 0.5B is strictly weaker than the 1.5B that's been
failing on the Space, so a positive result here is a strong signal.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from mutant_hunter.models import Action  # noqa: E402
from mutant_hunter.server.mutant_hunter_environment import MutantHunterEnvironment  # noqa: E402
from training.prompts import SYSTEM_PROMPT, render_few_shot  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MAX_NEW_TOKENS = 512


def _strip_markdown_fences(text: str) -> str:
    fence_re = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
    m = fence_re.search(text)
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


def build_sparse_prompt(obs) -> str:
    parts = [
        SYSTEM_PROMPT, "",
        "## Few-shot examples", render_few_shot(), "",
        "## Live task",
        f"Module: {obs.module_path}",
        f"Repo:   {obs.repo_name}",
        f"Baseline mutation score: {obs.baseline_mutation_score:.3f}",
        "",
        "## Module summary",
        obs.module_summary or "(no summary)",
        "",
        "## Existing tests",
    ]
    parts.extend(f"  - {n}" for n in (obs.existing_tests or []))
    parts.append("")
    parts.append("## Output the pytest file content now:")
    return "\n".join(parts)


def build_zeroshot_rich_prompt(obs, source_text: str) -> str:
    """Rich source, NO few-shots. Tests whether few-shots are poisoning output."""
    parts = [
        SYSTEM_PROMPT, "",
        "## Live task",
        f"Module: {obs.module_path}",
        f"Repo:   {obs.repo_name}",
        f"Baseline mutation score: {obs.baseline_mutation_score:.3f}",
        "",
        "## Module source (full)",
        "```python",
        source_text.rstrip(),
        "```",
        "",
        "## Existing tests",
    ]
    parts.extend(f"  - {n}" for n in (obs.existing_tests or []))
    parts.append("")
    parts.append("## Output the pytest file content now:")
    return "\n".join(parts)


def build_rich_prompt(obs, source_text: str) -> str:
    parts = [
        SYSTEM_PROMPT, "",
        "## Few-shot examples", render_few_shot(), "",
        "## Live task",
        f"Module: {obs.module_path}",
        f"Repo:   {obs.repo_name}",
        f"Baseline mutation score: {obs.baseline_mutation_score:.3f}",
        "",
        "## Module source (full)",
        "```python",
        source_text.rstrip(),
        "```",
        "",
        "## Existing tests",
    ]
    parts.extend(f"  - {n}" for n in (obs.existing_tests or []))
    parts.append("")
    parts.append("## Output the pytest file content now:")
    return "\n".join(parts)


def generate(model, tokenizer, prompt: str, label: str) -> str:
    import torch
    messages = [{"role": "user", "content": prompt}]
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(templated, return_tensors="pt", truncation=True, max_length=4096)
    print(f"  [{label}] prompt tokens: {inputs['input_ids'].shape[1]}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  [{label}] gen={time.time()-t0:.1f}s, raw_chars={len(full)}", flush=True)
    return _strip_markdown_fences(full)


def submit_and_score(env: MutantHunterEnvironment, seed: int, test_code: str, label: str) -> dict:
    env.reset(seed=seed)
    try:
        action = Action(kind="submit_tests", test_code=test_code)
    except Exception as exc:
        return {"label": label, "rejected": True, "err": str(exc), "reward": 0.0}
    obs2 = env.step(action)
    md = obs2.metadata or {}
    return {
        "label": label,
        "reward": float(obs2.reward or 0.0),
        "components": dict(md.get("components") or {}),
        "no_regression_gate": md.get("no_regression_gate"),
        "killed_by_new_only": md.get("killed_by_new_only"),
        "new_tests_pass_clean": md.get("new_tests_pass_clean"),
        "status": md.get("status"),
    }


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    env = MutantHunterEnvironment()
    obs = env.reset(seed=0)
    print(f"=== ep0 / {obs.repo_name} / {obs.module_path} ===", flush=True)

    source_path = ROOT / "src" / "mutant_hunter" / "corpus" / "_local" / "bloom_filter_lite" / "bloom.py"
    source_text = source_path.read_text(encoding="utf-8")

    p_sparse = build_sparse_prompt(obs)
    p_rich = build_rich_prompt(obs, source_text)
    p_zero = build_zeroshot_rich_prompt(obs, source_text)
    print(f"sparse: {len(p_sparse)} chars, rich: {len(p_rich)} chars, zeroshot-rich: {len(p_zero)} chars", flush=True)

    print(f"loading {MODEL_ID} on cpu (this takes ~30-60s) ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"\n--- generating SPARSE (current env summary + new few-shots) ---", flush=True)
    code_sparse = generate(model, tokenizer, p_sparse, "sparse")
    print(f"\n--- generating RICH (full source + new few-shots) ---", flush=True)
    code_rich = generate(model, tokenizer, p_rich, "rich")

    out_dir = ROOT / "evaluation" / "_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_sparse.py").write_text(code_sparse, encoding="utf-8")
    (out_dir / "probe_rich.py").write_text(code_rich, encoding="utf-8")
    print(f"\nwrote generated tests to {out_dir}/probe_{{sparse,rich}}.py", flush=True)

    print(f"\n--- scoring SPARSE in env ---", flush=True)
    s_sparse = submit_and_score(env, 0, code_sparse, "sparse")
    print(f"\n--- scoring RICH in env ---", flush=True)
    s_rich = submit_and_score(env, 0, code_rich, "rich")
    s_zero = None

    env.close()

    def show(s: dict) -> None:
        print(f"\n[{s['label']}] reward={s.get('reward'):.4f}")
        print(f"  components={s.get('components')}")
        print(f"  no_regression_gate={s.get('no_regression_gate')}")
        print(f"  killed_by_new_only={s.get('killed_by_new_only')}")
        print(f"  new_tests_pass_clean={s.get('new_tests_pass_clean')}")
        print(f"  status={s.get('status')}")

    show(s_sparse)
    show(s_rich)

    print("\n=== verdict ===")
    g_sparse = s_sparse.get("no_regression_gate", 0.0) or 0.0
    g_rich = s_rich.get("no_regression_gate", 0.0) or 0.0
    if g_sparse > 0.0 or g_rich > 0.0:
        print("FIX WORKS: at least one prompt cleared the gate on 0.5B with new prompts.py.")
    else:
        print("STILL BROKEN: 0.5B couldn't pass gate even with new prompts. May need env-side summary fix or larger model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
