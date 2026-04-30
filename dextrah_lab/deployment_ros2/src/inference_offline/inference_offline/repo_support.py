"""Helpers for resolving the local repository root at runtime."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path(explicit_repo_root: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_repo_root:
        candidates.append(Path(explicit_repo_root).expanduser().resolve())

    for parent in Path(__file__).resolve().parents:
        candidates.append(parent)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "setup.py").exists() and (candidate / "dextrah_lab" / "tasks").exists():
            repo_root = str(candidate)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return candidate

    raise ImportError(
        "Could not locate the tg2_dexman_isaac repository root. "
        "Pass --repo_root explicitly if running outside this checkout."
    )
