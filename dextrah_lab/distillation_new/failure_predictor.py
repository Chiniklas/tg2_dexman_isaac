"""Failure predictor for online risk estimation.

This module is intentionally self-contained and NOT wired into the training loop yet.
It provides a simple MLP classifier that predicts failure probability given (state, action).

Notes:
- Feature extraction uses raw observation tensors (placeholder). Replace with encoder features if needed.
- Failure label computation is task-specific (placeholder). See compute_failure_label().
- Supports a simple horizon-based labeling heuristic using a deque of recent buffer indices.
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

        # Failure labeling config
        self.horizon_steps = int(cfg.get("horizon_steps", 1))
        self.failure_threshold = float(cfg.get("failure_threshold", 0.5))

        # Runtime state
        self._initialized = False
        self._steps = 0
        self._buf_idx = 0
        self._buf_count = 0
        self._recent_indices = deque(maxlen=max(self.horizon_steps, 1))

        self._input_dim = None
        self._buffer_x = None
        self._buffer_y = None
        self._model = None
        self._optim = None

        if self.rank == 0 and self.enabled:
            print(
                "FailurePredictor enabled: "
                f"buffer_size={self.buffer_size}, min_samples={self.min_samples}, "
                f"update_interval={self.update_interval}, horizon_steps={self.horizon_steps}"
            )

    # --------- Public API ---------
    def add_step(self, obs, action, reward=None, done=None, info=None):
        """Add a step to the buffer and optionally update labels for recent horizon."""
        if not self.enabled:
            return
        x = self._build_input(obs, action)
        label = self.compute_failure_label(obs, reward, done, info)
        self._update_buffer(x, label)
        self._steps += 1
        if self._steps % self.update_interval == 0:
            self.train_step()

    def train_step(self):
        """Train one step on a random batch (if enough data)."""
        if not self.enabled or not self._initialized:
            return None
        if self._buf_count < self.min_samples:
            return None
        idx = torch.randint(0, self._buf_count, (self.batch_size,))
        x = self._buffer_x[idx].to(self.device)
        y = self._buffer_y[idx].to(self.device)
        logits = self._model(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        return loss.detach().item()

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
        probs = self.predict_risk(obs, action)
        if probs is None:
            return None
        return probs > self.failure_threshold

    # --------- Placeholder Labeling ---------
    def compute_failure_label(self, obs, reward, done, info) -> torch.Tensor:
        """Compute failure labels.

        Placeholder logic:
        - If info has explicit failure flags, use them.
        - Else, use done as failure.
        Customize this for your task (slip, drop, loss of object, etc.).
        """
        if isinstance(info, dict):
            for key in ("failure", "failed", "task_failed"):
                if key in info:
                    return torch.as_tensor(info[key], dtype=torch.float32, device=self.device)
            if "unsafe" in info:
                return torch.as_tensor(info["unsafe"], dtype=torch.float32, device=self.device)
            if "in_success_region" in info:
                # If success flag exists, treat failure as not in success region
                return 1.0 - torch.as_tensor(info["in_success_region"], dtype=torch.float32, device=self.device)
        if done is not None:
            return torch.as_tensor(done, dtype=torch.float32, device=self.device)
        # Placeholder: no signal available
        return torch.zeros(self._num_envs_from_obs(obs), dtype=torch.float32, device=self.device)

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
        self._buf_idx = 0
        self._buf_count = 0

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

    def _update_buffer(self, x, label):
        x_cpu = x.detach().to("cpu")
        label_cpu = label.detach().to("cpu").reshape(-1)
        num = x_cpu.shape[0]
        if num >= self.buffer_size:
            x_cpu = x_cpu[-self.buffer_size :]
            label_cpu = label_cpu[-self.buffer_size :]
            num = x_cpu.shape[0]
        end = self._buf_idx + num
        if end <= self.buffer_size:
            self._buffer_x[self._buf_idx:end] = x_cpu
            self._buffer_y[self._buf_idx:end] = label_cpu
            indices = list(range(self._buf_idx, end))
        else:
            first = self.buffer_size - self._buf_idx
            self._buffer_x[self._buf_idx:] = x_cpu[:first]
            self._buffer_y[self._buf_idx:] = label_cpu[:first]
            self._buffer_x[: end % self.buffer_size] = x_cpu[first:]
            self._buffer_y[: end % self.buffer_size] = label_cpu[first:]
            indices = list(range(self._buf_idx, self.buffer_size)) + list(range(0, end % self.buffer_size))
        self._buf_idx = end % self.buffer_size
        self._buf_count = min(self.buffer_size, self._buf_count + num)

        # Horizon-based labeling: if any current label is failure, mark recent states as failure.
        for idx, lbl in zip(indices, label_cpu.tolist()):
            self._recent_indices.append(idx)
            if lbl >= 0.5 and self.horizon_steps > 1:
                for ridx in self._recent_indices:
                    self._buffer_y[ridx] = 1.0

    def _num_envs_from_obs(self, obs):
        key = self.obs_key or self.default_obs_key
        if key is None or key not in obs:
            return 1
        return obs[key].shape[0]
