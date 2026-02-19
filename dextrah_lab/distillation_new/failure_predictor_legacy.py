"""Failure predictor for online risk estimation.

This module is intentionally self-contained and NOT wired into the training loop yet.
It provides a simple MLP classifier that predicts failure probability given (state, action).

Labeling semantics:
- Target is "failure within N future steps".
- Labels are assigned online using per-env pending queues.
- On failure termination, the previous N buffered states are labeled positive.
- Non-failure episode endings flush pending states as negatives.
"""

from collections import deque
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FailurePredictor:
    def __init__(
        self,
        config: Optional[Dict] = None,
        device: str = "cpu",
        default_obs_key: Optional[str] = None,
        rank: int = 0,
    ):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.device = torch.device(device)
        self.rank = rank

        # Feature config
        self.obs_key = cfg.get("obs_key", None)
        self.default_obs_key = default_obs_key
        self.action_key = cfg.get("action_key", None)  # if actions are stored in obs dict

        # Model config
        self.hidden_sizes = cfg.get("hidden_sizes", [256, 128])
        self.lr = float(cfg.get("lr", 1e-3))
        self.dropout = float(cfg.get("dropout", 0.0))

        # Buffer config
        self.buffer_size = int(cfg.get("buffer_size", 100_000))
        self.batch_size = int(cfg.get("batch_size", 1024))
        self.min_samples = int(cfg.get("min_samples", 10_000))
        self.update_interval = int(cfg.get("update_interval", 1_000))
        self.train_steps = int(cfg.get("train_steps", 1))
        if self.train_steps <= 0:
            self.train_steps = 1

        # Failure labeling config
        self.horizon_steps = int(cfg.get("horizon_steps", 1))
        self.failure_threshold = float(cfg.get("failure_threshold", 0.5))
        # If True, includes terminal failure step itself in positive backfill window.
        self.include_current_step = bool(cfg.get("include_current_step", False))
        self.pos_weight = cfg.get("pos_weight", None)

        # Runtime state
        self._initialized = False
        self._steps = 0
        self._buf_idx = 0
        self._buf_count = 0
        self._token_counter = 0

        self._input_dim = None
        self._buffer_x = None
        self._buffer_y = None
        self._buffer_labeled = None
        self._buffer_token = None
        self._model = None
        self._optim = None
        self._num_envs = None
        self._pending_indices = None

        if self.rank == 0 and self.enabled:
            print(
                "FailurePredictor enabled: "
                f"buffer_size={self.buffer_size}, min_samples={self.min_samples}, "
                f"update_interval={self.update_interval}, train_steps={self.train_steps}, "
                f"horizon_steps={self.horizon_steps}, "
                f"include_current_step={self.include_current_step}"
            )

    # --------- Public API ---------
    def add_step(self, obs, action, reward=None, done=None, info=None):
        """Add a batched env step and update horizon-to-failure labels."""
        if not self.enabled:
            return None
        x = self._build_input(obs, action)
        num_envs = x.shape[0]
        if self._num_envs is None:
            self._num_envs = num_envs
            self._pending_indices = [deque() for _ in range(num_envs)]
        elif self._num_envs != num_envs:
            raise ValueError(
                f"FailurePredictor num_envs mismatch: expected {self._num_envs}, got {num_envs}."
            )

        done_mask, failure_done_mask = self._compute_done_and_failure_masks(
            obs=obs, reward=reward, done=done, info=info, num_envs=num_envs
        )
        refs = self._update_buffer(x)

        # Per-env online relabeling:
        # - If episode fails, backfill previous N steps as positive.
        # - If episode ends without failure, flush pending as negative.
        # - Otherwise, once a sample is older than N steps without failure, finalize negative.
        for env_id, ref in enumerate(refs):
            pending = self._pending_indices[env_id]
            pending.append(ref)

            if bool(done_mask[env_id].item()):
                if bool(failure_done_mask[env_id].item()):
                    if self.horizon_steps > 0:
                        if self.include_current_step:
                            pos_refs = list(pending)[-self.horizon_steps :]
                        else:
                            pos_refs = list(pending)[-(self.horizon_steps + 1) : -1]
                    else:
                        pos_refs = []
                    for pref in pos_refs:
                        self._finalize_ref(pref, 1.0)
                else:
                    for pref in pending:
                        self._finalize_ref(pref, 0.0)
                pending.clear()
            else:
                while len(pending) > self.horizon_steps:
                    oldest = pending.popleft()
                    self._finalize_ref(oldest, 0.0)

        self._steps += 1
        if self._steps % self.update_interval == 0:
            return self.train_step()
        return None

    def train_step(self):
        """Train one or more minibatch steps (if enough data)."""
        if not self.enabled or not self._initialized:
            return None
        labeled_indices = self._get_labeled_indices()
        if labeled_indices.numel() < self.min_samples:
            return None
        losses = []
        for _ in range(self.train_steps):
            sample_sel = torch.randint(0, labeled_indices.numel(), (self.batch_size,))
            idx = labeled_indices[sample_sel]
            x = self._buffer_x[idx].to(self.device)
            y = self._buffer_y[idx].to(self.device)
            logits = self._model(x).squeeze(-1)
            if self.pos_weight is not None:
                pos_weight = torch.tensor(float(self.pos_weight), device=self.device)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y)
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
            losses.append(float(loss.detach().item()))
        return float(sum(losses) / len(losses))

    def predict_risk(self, obs, action) -> torch.Tensor:
        """Return failure probability per env."""
        if not self.enabled or not self._initialized:
            return None
        x = self._build_input(obs, action).to(self.device)
        with torch.no_grad():
            logits = self._model(x).squeeze(-1)
            return torch.sigmoid(logits)

    def should_intervene(self, obs, action) -> Optional[torch.Tensor]:
        """Return boolean mask for intervention based on predicted risk."""
        if not self.enabled:
            return None
        if self._get_labeled_indices().numel() < self.min_samples:
            return None
        probs = self.predict_risk(obs, action)
        if probs is None:
            return None
        return probs > self.failure_threshold

    # --------- Labeling helpers ---------
    def compute_failure_label(self, obs, reward, done, info) -> torch.Tensor:
        """Backward-compatible helper: returns per-env failure flag."""
        num_envs = self._num_envs_from_obs(obs)
        _, failure_done = self._compute_done_and_failure_masks(
            obs=obs, reward=reward, done=done, info=info, num_envs=num_envs
        )
        return failure_done.to(dtype=torch.float32)

    def _compute_done_and_failure_masks(
        self,
        obs,
        reward,
        done,
        info,
        num_envs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer episode termination and failure-termination masks."""
        done_mask = self._to_bool_tensor(done, num_envs)
        failure_mask = None

        if isinstance(info, dict):
            # Prefer explicit failure signals if present.
            for key in ("failure", "failed", "task_failed", "unsafe", "out_of_reach"):
                if key in info:
                    failure_mask = self._to_bool_tensor(info[key], num_envs)
                    break

            # Optional done-like flags in info.
            if "done" in info:
                done_mask = done_mask | self._to_bool_tensor(info["done"], num_envs)
            if "terminated" in info:
                done_mask = done_mask | self._to_bool_tensor(info["terminated"], num_envs)
            if "timed_out" in info:
                done_mask = done_mask | self._to_bool_tensor(info["timed_out"], num_envs)
            if "time_out" in info:
                done_mask = done_mask | self._to_bool_tensor(info["time_out"], num_envs)

            if failure_mask is None and "in_success_region" in info:
                # If success region exists, treat not-success at termination as failure.
                success = self._to_bool_tensor(info["in_success_region"], num_envs)
                failure_mask = ~success

        if failure_mask is None:
            # Fallback: if only done is available, assume done indicates failure.
            failure_mask = done_mask.clone()

        failure_done = done_mask & failure_mask
        return done_mask, failure_done

    # --------- Internal helpers ---------
    def _build_input(self, obs, action):
        feats = self._extract_features(obs)
        feats = feats.reshape(feats.shape[0], -1)
        if action is None:
            raise ValueError("action is required for FailurePredictor input.")
        act = action.detach().to(dtype=torch.float32)
        act = act.reshape(act.shape[0], -1)
        x = torch.cat([feats, act], dim=-1)
        if not self._initialized:
            self._init_model_and_buffer(x.shape[1])
        return x

    def _extract_features(self, obs):
        if obs is None:
            raise ValueError("obs is required for failure predictor.")
        key = self.obs_key or self.default_obs_key
        if key is None:
            raise ValueError("obs_key is not set and no default_obs_key provided.")
        if key not in obs:
            raise KeyError(f"obs_key '{key}' not found in obs.")
        feats = obs[key]
        return feats.detach().to(dtype=torch.float32)

    def _init_model_and_buffer(self, input_dim: int):
        self._input_dim = int(input_dim)
        self._buffer_x = torch.empty(
            (self.buffer_size, self._input_dim), dtype=torch.float32, device="cpu"
        )
        self._buffer_y = torch.zeros((self.buffer_size,), dtype=torch.float32, device="cpu")
        self._buffer_labeled = torch.zeros((self.buffer_size,), dtype=torch.bool, device="cpu")
        self._buffer_token = torch.full((self.buffer_size,), -1, dtype=torch.long, device="cpu")
        self._buf_idx = 0
        self._buf_count = 0
        self._token_counter = 0

        layers = []
        in_dim = self._input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self._model = nn.Sequential(*layers).to(self.device)
        self._optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._initialized = True

    def _update_buffer(self, x):
        x_cpu = x.detach().to("cpu")
        num = x_cpu.shape[0]
        if num >= self.buffer_size:
            x_cpu = x_cpu[-self.buffer_size :]
            num = x_cpu.shape[0]
        tokens = torch.arange(self._token_counter, self._token_counter + num, dtype=torch.long)
        self._token_counter += num
        end = self._buf_idx + num
        if end <= self.buffer_size:
            self._buffer_x[self._buf_idx:end] = x_cpu
            self._buffer_y[self._buf_idx:end] = 0.0
            self._buffer_labeled[self._buf_idx:end] = False
            self._buffer_token[self._buf_idx:end] = tokens
            indices = list(range(self._buf_idx, end))
        else:
            first = self.buffer_size - self._buf_idx
            self._buffer_x[self._buf_idx:] = x_cpu[:first]
            self._buffer_y[self._buf_idx:] = 0.0
            self._buffer_labeled[self._buf_idx:] = False
            self._buffer_token[self._buf_idx:] = tokens[:first]
            self._buffer_x[: end % self.buffer_size] = x_cpu[first:]
            self._buffer_y[: end % self.buffer_size] = 0.0
            self._buffer_labeled[: end % self.buffer_size] = False
            self._buffer_token[: end % self.buffer_size] = tokens[first:]
            indices = list(range(self._buf_idx, self.buffer_size)) + list(range(0, end % self.buffer_size))
        self._buf_idx = end % self.buffer_size
        self._buf_count = min(self.buffer_size, self._buf_count + num)
        return list(zip(indices, tokens.tolist()))

    def _finalize_ref(self, ref, label: float):
        """Assign a final label to a buffered sample if it hasn't been overwritten."""
        idx, token = ref
        if idx < 0 or idx >= self.buffer_size:
            return
        if int(self._buffer_token[idx].item()) != int(token):
            return
        self._buffer_y[idx] = max(float(label), float(self._buffer_y[idx].item()))
        self._buffer_labeled[idx] = True

    def _to_bool_tensor(self, value, num_envs: int) -> torch.Tensor:
        if value is None:
            return torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        t = torch.as_tensor(value, device=self.device)
        if t.ndim == 0:
            t = t.repeat(num_envs)
        t = t.reshape(-1)
        if t.numel() == 1 and num_envs != 1:
            t = t.repeat(num_envs)
        if t.numel() != num_envs:
            raise ValueError(
                f"Expected tensor with {num_envs} elements, got shape {tuple(t.shape)}."
            )
        return t.to(dtype=torch.bool)

    def _num_envs_from_obs(self, obs):
        key = self.obs_key or self.default_obs_key
        if key is None or key not in obs:
            return 1
        return obs[key].shape[0]

    def _get_labeled_indices(self):
        if self._buffer_labeled is None:
            return torch.empty((0,), dtype=torch.long, device="cpu")
        valid_mask = self._buffer_labeled.clone()
        if self._buf_count < self.buffer_size:
            valid_mask[self._buf_count :] = False
        return torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)


