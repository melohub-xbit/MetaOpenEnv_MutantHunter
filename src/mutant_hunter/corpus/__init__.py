"""The corpus package: target repos + per-module precomputed baselines.

Layout
------

    corpus/
      __init__.py
      baselines.py     — baseline cache loader/saver + helpers
      manifest.json    — list of {repo, modules, source, ...}
      _local/          — self-curated mini-libs (mini_calendar, ...)
      _baselines/      — one JSON per (repo, module) with mutants + cov
      _cache/          — git-cloned upstream repos (gitignored, optional)
"""

from __future__ import annotations

from .baselines import (
    BASELINE_ROOT,
    Baseline,
    CACHE_ROOT,
    CORPUS_ROOT,
    LOCAL_LIBS_ROOT,
    MANIFEST_PATH,
    baseline_path,
    dotted_to_workspace_relpath,
    list_existing_tests,
    load_baseline,
    module_to_relpath,
    repo_dir,
    save_baseline,
    summarize_module,
)

__all__ = [
    "BASELINE_ROOT",
    "Baseline",
    "CACHE_ROOT",
    "CORPUS_ROOT",
    "LOCAL_LIBS_ROOT",
    "MANIFEST_PATH",
    "baseline_path",
    "dotted_to_workspace_relpath",
    "list_existing_tests",
    "load_baseline",
    "module_to_relpath",
    "repo_dir",
    "save_baseline",
    "summarize_module",
]
