"""Legacy predictor path: Bellman critic with success-state labels.

This module keeps the legacy `FailurePredictor` API used by the pipeline, but
implements a Bellman-style safety critic (same training core as
`FailurePredictorCritic`) with success-centric supervision:

- Target label is success signal (not failure signal).
- Optional horizon back-labeling marks the recent `horizon_steps` as success
  whenever in-episode success is observed.
- Intervention is triggered when predicted success is below a threshold.

The class name remains `FailurePredictor` for backward compatibility with
`failure_predictor_type=legacy`.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    from dextrah_lab.distillation_new.failure_predictor import FailurePredictorCritic
except Exception:
    from failure_predictor import FailurePredictorCritic


class SuccessPredictorLegacy(FailurePredictorCritic):
    """Bellman critic that predicts success probability for intervention gating."""

    def __init__(self, config=None, device: str = "cpu", default_obs_key: Optional[str] = None, rank: int = 0):
        cfg = dict(config or {})
        success_threshold = float(cfg.get("success_threshold", cfg.get("failure_threshold", 0.5)))
        cfg["failure_threshold"] = success_threshold
        super().__init__(cfg, device=device, default_obs_key=default_obs_key, rank=rank)
        self.success_threshold = success_threshold
        configured_success_key = str(cfg.get("success_key", "lift_success"))
        if configured_success_key != "lift_success":
            raise ValueError(
                f"SuccessPredictorLegacy only supports success_key='lift_success', got '{configured_success_key}'."
            )
        self.success_key = "lift_success"
        if self.rank == 0 and self.enabled:
            print(
                "[SuccessPredictorLegacy] enabled: "
                f"success_threshold={self.success_threshold}, "
                f"horizon_steps={self.horizon_steps}, "
                f"success_key={self.success_key}",
                flush=True,
            )

    def add_step(self, obs, action, next_obs=None, reward=None, done=None, info=None):
        """Add one batched env step using success-centric labels."""
        if not self.enabled:
            return None

        obs_feats = self._flatten_features(self._extract_features(obs))
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        act = act.reshape(obs_feats.shape[0], -1)

        if next_obs is None:
            next_feats = obs_feats
            self._used_next_obs_fallback += int(obs_feats.shape[0])
        else:
            next_feats = self._flatten_features(self._extract_features(next_obs))
            if next_feats.shape[0] != obs_feats.shape[0]:
                raise ValueError(
                    f"next_obs batch size ({next_feats.shape[0]}) does not match obs batch size ({obs_feats.shape[0]})."
                )

        num_envs = obs_feats.shape[0]
        self._ensure_initialized(obs_feats.shape[1], act.shape[1], num_envs)
        done_mask, success_mask = self._compute_done_and_success_masks(
            obs=obs, reward=reward, done=done, info=info, num_envs=num_envs
        )
        obs_cpu = obs_feats.detach().to("cpu")
        act_cpu = act.detach().to("cpu")
        next_cpu = next_feats.detach().to("cpu")
        for env_id in range(num_envs):
            prev_ref = self._open_refs[env_id]
            if prev_ref is not None:
                self._set_next_action(prev_ref, act_cpu[env_id])
                self._open_refs[env_id] = None

            success_now = 1.0 if bool(success_mask[env_id].item()) else 0.0
            done_now = 1.0 if bool(done_mask[env_id].item()) else 0.0
            curr_ref = self._store_transition(
                obs_cpu[env_id], act_cpu[env_id], next_cpu[env_id], success_now, done_now
            )
            self._recent_refs[env_id].append(curr_ref)
            if bool(success_mask[env_id].item()):
                self._mark_recent_failure_labels(env_id)
            if done_now > 0.5:
                self._set_next_action(curr_ref, torch.zeros_like(act_cpu[env_id]))
                self._recent_refs[env_id].clear()
            else:
                self._open_refs[env_id] = curr_ref

        self._steps += 1
        if self._steps % self.update_interval == 0:
            return self.train_step()
        return None

    def should_intervene(self, obs, action):
        """Intervene when predicted success is below threshold."""
        if not self.enabled:
            return None
        if self._buf_count < self.min_samples and not self._has_pretrained_model:
            return None
        success_prob = self.predict_risk(obs, action)
        if success_prob is None:
            return None
        return success_prob < self.success_threshold

    def compute_failure_label(self, obs, reward, done, info):
        """Compatibility helper returning failure-at-terminal signal."""
        num_envs = self._num_envs_from_obs(obs)
        done_mask, success_mask = self._compute_done_and_success_masks(
            obs=obs, reward=reward, done=done, info=info, num_envs=num_envs
        )
        return (done_mask & (~success_mask)).to(dtype=torch.float32)

    def _compute_done_and_success_masks(self, obs, reward, done, info, num_envs: int):
        """Infer done mask and per-step success mask.

        Requires a per-env vector `info["lift_success"]`.
        Missing or malformed labels fail fast.
        """
        done_mask = self._to_bool_tensor(done, num_envs)
        if isinstance(info, dict):
            for key in ("done", "terminated"):
                val = self._to_bool_tensor_if_match(info.get(key, None), num_envs)
                if val is not None:
                    done_mask = done_mask | val
            for key in ("timed_out", "time_out"):
                val = self._to_bool_tensor_if_match(info.get(key, None), num_envs)
                if val is not None:
                    done_mask = done_mask | val
        else:
            raise TypeError(
                "SuccessPredictorLegacy requires info to be a dict containing per-env 'lift_success'."
            )

        if self.success_key not in info:
            raise KeyError(
                "SuccessPredictorLegacy requires info['lift_success'] for success labeling, but it was missing."
            )
        success_mask = self._to_bool_tensor_if_match(info.get(self.success_key, None), num_envs)
        if success_mask is None:
            raise ValueError(
                f"info['lift_success'] must be a per-env vector of length {num_envs}."
            )

        return done_mask, success_mask

    def _to_bool_tensor_if_match(self, value, num_envs: int):
        """Convert to bool tensor only when it matches vectorized env shape.

        This intentionally ignores scalar aggregated metrics (common in info),
        so they do not get broadcast to every environment.
        """
        if value is None:
            return None
        t = torch.as_tensor(value, device=self.device).reshape(-1)
        if t.numel() != num_envs:
            return None
        return t.to(dtype=torch.bool)


class FailurePredictor(SuccessPredictorLegacy):
    """Backward-compatible export used by `failure_predictor_type=legacy`."""
