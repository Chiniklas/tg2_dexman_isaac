from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexsafedagger_lab.distillation.safety.success_value_critic import SuccessValueCritic
from dexsafedagger_lab.distillation.safety.vlm_threshold_advisor import VLMThresholdAdvisor
from dexsafedagger_lab.distillation.core.distill_warm_start import DistillWarmStart


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


def test_warm_start_calibrates_threshold_from_safe_successful_rollouts_only() -> None:
    class FakePredictor:
        enabled = True
        _initialized = True
        success_threshold = 0.4

        @staticmethod
        def predict_success(obs, action):
            del action
            return obs[:, 0]

    advisor = SimpleNamespace(
        base_success_threshold=0.4,
        state=SimpleNamespace(success_threshold=0.4),
    )
    agent = SimpleNamespace(
        rank=1,
        writer=None,
        failure_predictor=FakePredictor(),
        vlm_threshold_advisor=advisor,
        success_critic_base_threshold=0.4,
        warm_start_calibrate_success_threshold=True,
        warm_start_success_threshold_quantile=0.5,
        warm_start_success_threshold_floor=1.0e-6,
        warm_start_success_threshold_chunk_size=8,
    )
    samples = [
        {
            "episode_id": torch.tensor([0, 0]),
            "lift_success": torch.tensor([False, False]),
            "out_of_reach": torch.tensor([False, False]),
            "teacher_actions": torch.zeros((2, 1)),
            "obs": {"predictor_transition": torch.tensor([[0.1], [0.9]])},
        },
        {
            "episode_id": torch.tensor([0, 0]),
            "lift_success": torch.tensor([True, True]),
            "out_of_reach": torch.tensor([False, True]),
            "teacher_actions": torch.zeros((2, 1)),
            "obs": {"predictor_transition": torch.tensor([[0.3], [0.8]])},
        },
    ]

    metrics = DistillWarmStart(agent)._calibrate_success_threshold(samples)

    assert metrics is not None
    assert metrics["episode_count"] == 1
    assert metrics["sample_count"] == 2
    assert abs(metrics["threshold"] - 0.2) < 1.0e-6
    assert abs(agent.failure_predictor.success_threshold - 0.2) < 1.0e-6
    assert abs(agent.success_critic_base_threshold - 0.2) < 1.0e-6
    assert abs(advisor.base_success_threshold - 0.2) < 1.0e-6
    assert abs(advisor.state.success_threshold - 0.2) < 1.0e-6
