from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexsafedagger_lab.distillation.safety.success_value_critic import SuccessValueCritic
from dexsafedagger_lab.distillation.safety.vlm_threshold_advisor import VLMThresholdAdvisor


def _critic(**overrides) -> SuccessValueCritic:
    cfg = {
        "enabled": True,
        "obs_key": "predictor_transition",
        "hidden_sizes": [8],
        "buffer_size": 32,
        "batch_size": 4,
        "min_samples": 0,
        "update_interval": 100,
        "success_threshold": 0.4,
        "success_key": "lift_success",
        "horizon_steps": 2,
        "device": "cpu",
    }
    cfg.update(overrides)
    return SuccessValueCritic(cfg, device="cpu", default_obs_key="predictor_transition")


def test_success_labels_are_stored_and_back_labeled() -> None:
    critic = _critic()
    obs = {"predictor_transition": torch.tensor([[0.0, 1.0], [1.0, 0.0]])}
    next_obs = {"predictor_transition": torch.tensor([[0.1, 1.0], [1.0, 0.1]])}
    actions = torch.zeros((2, 1))

    critic.add_step(
        obs=obs,
        action=actions,
        next_obs=next_obs,
        done=torch.tensor([False, False]),
        info={"lift_success": torch.tensor([True, False])},
    )

    labels = critic._success_buf[: critic._buf_count]
    assert labels.tolist() == [1.0, 0.0]


def test_intervention_is_triggered_by_low_success() -> None:
    critic = _critic(success_threshold=0.4)
    critic._initialized = True
    critic._has_pretrained_model = True
    critic.predict_success = lambda obs, action: torch.tensor([0.2, 0.7])

    intervene = critic.should_intervene(torch.zeros((2, 1)), torch.zeros((2, 1)))

    assert intervene.tolist() == [True, False]


def test_vlm_advisor_applies_success_threshold() -> None:
    advisor = VLMThresholdAdvisor(
        {
            "enabled": True,
            "mode": "active",
            "smoothing": 1.0,
            "success_min_scale": 0.5,
            "success_max_scale": 1.5,
        },
        base_l2_threshold=8.0,
        base_success_threshold=0.4,
    )

    applied = advisor._apply_recommendation(
        {
            "l2_threshold": 9.0,
            "success_threshold": 0.5,
            "confidence": 0.8,
            "reason": "More low-success states were observed.",
        }
    )

    assert applied["success_threshold"] == 0.5
    assert advisor.current_success_threshold == 0.5
