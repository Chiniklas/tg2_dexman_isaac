"""Scaffold for VLM-driven intervention decisions.

DexSafeDaggerUltra is an ablation idea, not a completed method yet. The goal is
to let a vision-language model decide when the teacher should intervene instead
of using a fixed action-disagreement threshold.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class VLMInterventionConfig:
    """Configuration shell for the future VLM intervention policy."""

    enabled: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_template: str = "Should the teacher intervene now? Return a risk score in [0, 1]."
    decision_threshold: Optional[float] = None
    temporal_window: int = 1
    scaffold_only: bool = True


class VLMInterventionPlanner:
    """Future VLM gate used by the dexsafedaggerUltra ablation.

    Intended data flow:
    1. Extract current stereo/RGB frames and compact robot/object state.
    2. Build a prompt that asks whether teacher takeover is needed now.
    3. Query a VLM for either a binary intervention decision or risk score.
    4. Apply temporal smoothing/caching before returning an unsafe mask.

    This class deliberately raises until a real VLM backend and batching policy
    are wired in. That makes the ablation visible without producing misleading
    results from a placeholder heuristic.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, *, rank: int = 0):
        cfg = cfg or {}
        self.config = VLMInterventionConfig(
            enabled=bool(cfg.get("enabled", False)),
            provider=cfg.get("provider", None),
            model=cfg.get("model", None),
            prompt_template=str(
                cfg.get(
                    "prompt_template",
                    "Should the teacher intervene now? Return a risk score in [0, 1].",
                )
            ),
            decision_threshold=cfg.get("decision_threshold", None),
            temporal_window=int(cfg.get("temporal_window", 1)),
            scaffold_only=bool(cfg.get("scaffold_only", True)),
        )
        self.enabled = self.config.enabled
        self.rank = rank

    def should_intervene(self, *, obs, student_action, teacher_action=None, info=None):
        """Return a boolean unsafe mask once the VLM backend is implemented."""
        raise NotImplementedError(
            "dexsafedaggerUltra is currently a scaffold. Implement VLM frame "
            "extraction, prompting, backend inference, and temporal smoothing "
            "inside VLMInterventionPlanner.should_intervene before running it."
        )
