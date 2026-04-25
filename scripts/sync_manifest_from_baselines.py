"""Backfill `corpus/manifest.json` summary fields from `corpus/_baselines/*.json`.

Use when a parallel-write race against `manifest.json` has left a module
entry with null counts even though its baseline JSON is on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mutant_hunter.corpus.baselines import (  # noqa: E402
    BASELINE_ROOT,
    MANIFEST_PATH,
    baseline_path,
)


def _count_loc(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    changed = 0

    for repo in manifest["repos"]:
        repo_name = repo["name"]
        for mod in repo["modules"]:
            dotted = mod["module"]
            bp = baseline_path(repo_name, dotted)
            if not bp.exists():
                print(f"  skip {repo_name}:{dotted}  (no baseline file)")
                continue
            data = json.loads(bp.read_text(encoding="utf-8"))
            mod["total_mutants"] = data["total_mutants"]
            mod["surviving_mutants"] = len(data["surviving_mutants"])
            mod["baseline_mutation_score"] = round(float(data["baseline_mutation_score"]), 6)
            mod["coverage_baseline"] = round(float(data["coverage_baseline"]), 4)
            # loc is recomputed from the actual source file.
            relpath = Path(data["module_relpath"])
            src = ROOT / "src" / "mutant_hunter" / "corpus" / "_local" / repo_name / relpath
            if src.exists():
                mod["loc"] = _count_loc(src)
            print(
                f"  sync {repo_name}:{dotted}  loc={mod['loc']} "
                f"total={mod['total_mutants']} survived={mod['surviving_mutants']} "
                f"cov={mod['coverage_baseline']} score={mod['baseline_mutation_score']}"
            )
            changed += 1

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}  ({changed} modules synced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
