"""Helpers for finding the repository root at runtime."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "setup.py").exists() and (candidate / "dexsafedagger_lab" / "utils" / "kinematics.py").exists():
            repo_root = str(candidate)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return candidate
    raise ImportError(
        "Could not locate the repository root containing dexsafedagger_lab/utils. "
        "Run this package from the tg2_dexman_isaac checkout."
    )
