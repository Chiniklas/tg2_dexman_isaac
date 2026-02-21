"""Bellman-style failure risk critic for intervention gating.

What this module is for
-----------------------
This predictor estimates how likely a state-action pair is to eventually end
in a failure event. The output is used online to decide whether to intervene
(for example, switch from student action to teacher action).

The class in this file is the "new" predictor path:
    FailurePredictorCritic
The old horizon-label predictor was moved to:
    failure_predictor_legacy.py

Core idea
---------
We model a safety critic F(s, a) in [0, 1], interpreted as a discounted
eventual failure probability:

    F(s, a) ~= P(failure eventually | s, a, policy rollouts)

Training uses Bellman bootstrapping with twin critics and target networks:

    y = f_now + gamma * (1 - d_eff) * min(F1_targ(s', a'), F2_targ(s', a'))

where:
- f_now is immediate failure signal (1 only on failure terminal transition)
- d_eff is terminal mask (or "cannot bootstrap" mask)
- a' is the next action observed from rollout (SARSA-style target)

Data flow in training loop
--------------------------
1) The distillation loop calls add_step(obs, action, next_obs, done, info) once
   per environment step.
2) add_step converts observations to features, infers done/failure masks, and
   writes transitions into a ring replay buffer.
3) Every update_interval calls, train_step() samples replay minibatches and
   updates twin critics by MSE to Bellman targets.
4) At inference time, predict_risk()/should_intervene() use min(q1, q2) as a
   conservative risk estimate.

Why there is "open transition" state
------------------------------------
The replay stores (s, a, s', a_next, f_now, done). At time t we know (s_t, a_t)
and s_{t+1}, but we only know a_{t+1} when the next step arrives. To support
this, each env keeps one "open" transition reference. On the next call, that
previous transition is closed by writing its a_next.

Failure labeling semantics
--------------------------
This predictor treats failure as terminal event supervision inferred from
info/done fields. It does not backtrace N steps (legacy behavior). The useful
signal propagates backward through Bellman bootstrapping over replay updates.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

class _FailureQFunction(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = obs_dim + act_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class FailurePredictorCritic:
    """Bellman-style failure risk critic with twin Q networks."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        device: str = "cpu",
        default_obs_key: Optional[str] = None,
        rank: int = 0,
    ):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.device = torch.device(device)
        self.rank = rank
        self.supports_next_obs = True

        self.obs_key = cfg.get("obs_key", None)
        self.default_obs_key = default_obs_key

        self.hidden_sizes = cfg.get("hidden_sizes", [256, 128])
        self.lr = float(cfg.get("lr", 1e-3))
        self.dropout = float(cfg.get("dropout", 0.0))
        self.gamma = float(cfg.get("gamma", 0.99))
        # Reference alignment: use fixed target-network averaging factor.
        self.polyak = 0.995
        self.failure_threshold = float(cfg.get("failure_threshold", 0.5))
        self.pos_weight = cfg.get("pos_weight", None)
        self.pos_fraction = cfg.get("pos_fraction", 0.1)
        if self.pos_fraction is None:
            self.pos_fraction = 0.0
        self.pos_fraction = float(self.pos_fraction)
        if not (0.0 <= self.pos_fraction <= 1.0):
            raise ValueError(
                f"pos_fraction must be in [0, 1], got {self.pos_fraction}."
            )

        self.buffer_size = int(cfg.get("buffer_size", 100_000))
        self.batch_size = int(cfg.get("batch_size", 128))
        self.min_samples = int(cfg.get("min_samples", 10_000))
        self.update_interval = int(cfg.get("update_interval", 1_000))
        self.train_steps = int(cfg.get("train_steps", 1))
        if self.train_steps <= 0:
            self.train_steps = 1

        self.return_debug_dict = bool(cfg.get("return_debug_dict", False))
        self.debug_print_interval = int(cfg.get("debug_print_interval", 0))
        self.last_train_stats: dict[str, float | int] = {}
        self._used_next_obs_fallback = 0

        self._initialized = False
        self._steps = 0
        self._q_update_steps = 0
        self._buf_idx = 0
        self._buf_count = 0
        self._token_counter = 0
        self._num_envs = None

        self._obs_dim = None
        self._act_dim = None
        self._q1 = None
        self._q2 = None
        self._q1_targ = None
        self._q2_targ = None
        self._optim = None

        self._obs_buf = None
        self._act_buf = None
        self._obs2_buf = None
        self._next_act_buf = None
        self._has_next_act = None
        self._fail_buf = None
        self._done_buf = None
        self._token_buf = None
        self._open_refs = None

        if self.rank == 0 and self.enabled:
            print(
                "FailurePredictorCritic enabled: "
                f"buffer_size={self.buffer_size}, min_samples={self.min_samples}, "
                f"update_interval={self.update_interval}, train_steps={self.train_steps}, "
                f"gamma={self.gamma}, polyak={self.polyak}, failure_threshold={self.failure_threshold}, "
                f"pos_fraction={self.pos_fraction}",
                flush=True,
            )

    def add_step(self, obs, action, next_obs=None, reward=None, done=None, info=None):
        """Add one vectorized transition batch from rollout.

        Expected call semantics:
        - obs/action are (s_t, a_t)
        - next_obs is s_{t+1}
        - done/info correspond to transition outcome at t -> t+1

        This function writes transitions into replay and links each previous
        transition to the current action as SARSA next-action target.
        """
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
        done_mask, failure_done_mask = self._compute_done_and_failure_masks(
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

            fail_now = 1.0 if bool(failure_done_mask[env_id].item()) else 0.0
            done_now = 1.0 if bool(done_mask[env_id].item()) else 0.0
            curr_ref = self._store_transition(
                obs_cpu[env_id], act_cpu[env_id], next_cpu[env_id], fail_now, done_now
            )
            if done_now > 0.5:
                self._set_next_action(curr_ref, torch.zeros_like(act_cpu[env_id]))
            else:
                self._open_refs[env_id] = curr_ref

        self._steps += 1
        if self._steps % self.update_interval == 0:
            return self.train_step()
        return None

    def train_step(self):
        """Run one or more critic gradient updates from replay.

        Returns:
        - None if disabled, uninitialized, or replay has fewer than min_samples.
        - float loss_total by default.
        - dict of debug stats if return_debug_dict is enabled.
        """
        if not self.enabled or not self._initialized:
            return None
        if self._buf_count < self.min_samples:
            return None

        losses = []
        q1_losses = []
        q2_losses = []
        boot_frac = []
        target_means = []
        pred_means = []
        sampled_pos_frac = []
        replay_pos_frac = float(
            (self._fail_buf[: self._buf_count] > 0.5).to(dtype=torch.float32).mean().item()
        )
        for _ in range(self.train_steps):
            idx = self._sample_batch_indices(self.batch_size)
            obs = self._obs_buf[idx].to(self.device)
            act = self._act_buf[idx].to(self.device)
            obs2 = self._obs2_buf[idx].to(self.device)
            next_act = self._next_act_buf[idx].to(self.device)
            has_next_act = self._has_next_act[idx].to(self.device)
            fail = self._fail_buf[idx].to(self.device)
            done = self._done_buf[idx].to(self.device)
            sampled_pos_frac.append(float((fail > 0.5).to(dtype=torch.float32).mean().item()))

            d_eff = torch.maximum(done, (~has_next_act).to(dtype=torch.float32))
            with torch.no_grad():
                q1_t = self._q1_targ(obs2, next_act)
                q2_t = self._q2_targ(obs2, next_act)
                backup = fail + self.gamma * (1.0 - d_eff) * torch.minimum(q1_t, q2_t)
                backup = torch.clamp(backup, 0.0, 1.0)

            q1 = self._q1(obs, act)
            q2 = self._q2(obs, act)
            if self.pos_weight is not None:
                w = torch.ones_like(backup)
                w[backup > 0.5] = float(self.pos_weight)
                loss_q1 = (w * (q1 - backup) ** 2).mean()
                loss_q2 = (w * (q2 - backup) ** 2).mean()
            else:
                loss_q1 = ((q1 - backup) ** 2).mean()
                loss_q2 = ((q2 - backup) ** 2).mean()
            loss = loss_q1 + loss_q2

            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
            self._q_update_steps += 1
            # Reference alignment: update target networks every 2 Q updates.
            if self._q_update_steps % 2 == 0:
                self._polyak_update()

            losses.append(float(loss.detach().item()))
            q1_losses.append(float(loss_q1.detach().item()))
            q2_losses.append(float(loss_q2.detach().item()))
            boot_frac.append(float((1.0 - d_eff).mean().item()))
            target_means.append(float(backup.mean().item()))
            pred_means.append(float(torch.minimum(q1, q2).mean().item()))

        if len(losses) == 0:
            return None

        self.last_train_stats = {
            "loss_total": float(sum(losses) / len(losses)),
            "loss_q1": float(sum(q1_losses) / len(q1_losses)),
            "loss_q2": float(sum(q2_losses) / len(q2_losses)),
            "bootstrap_frac": float(sum(boot_frac) / len(boot_frac)),
            "target_mean": float(sum(target_means) / len(target_means)),
            "pred_mean": float(sum(pred_means) / len(pred_means)),
            "buffer_size": int(self._buf_count),
            "used_next_obs_fallback": int(self._used_next_obs_fallback),
            "sampled_pos_frac": float(sum(sampled_pos_frac) / len(sampled_pos_frac)),
            "replay_pos_frac": replay_pos_frac,
        }
        if self.debug_print_interval > 0 and (self._steps % self.debug_print_interval == 0) and self.rank == 0:
            print(f"[FailurePredictorCritic] {self.last_train_stats}", flush=True)

        if self.return_debug_dict:
            return dict(self.last_train_stats)
        return self.last_train_stats["loss_total"]

    def predict_risk(self, obs, action):
        if not self.enabled or not self._initialized:
            return None
        feats = self._flatten_features(self._extract_features(obs))
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        act = act.reshape(feats.shape[0], self._act_dim)
        with torch.no_grad():
            q1 = self._q1(feats, act)
            q2 = self._q2(feats, act)
            return torch.minimum(q1, q2)

    def should_intervene(self, obs, action):
        if not self.enabled:
            return None
        if self._buf_count < self.min_samples:
            return None
        risk = self.predict_risk(obs, action)
        if risk is None:
            return None
        return risk > self.failure_threshold

    def compute_failure_label(self, obs, reward, done, info):
        num_envs = self._num_envs_from_obs(obs)
        _, failure_done = self._compute_done_and_failure_masks(
            obs=obs, reward=reward, done=done, info=info, num_envs=num_envs
        )
        return failure_done.to(dtype=torch.float32)

    def _ensure_initialized(self, obs_dim: int, act_dim: int, num_envs: int):
        if self._initialized:
            if self._num_envs != num_envs:
                raise ValueError(
                    f"FailurePredictorCritic num_envs mismatch: expected {self._num_envs}, got {num_envs}."
                )
            return

        self._obs_dim = int(obs_dim)
        self._act_dim = int(act_dim)
        self._num_envs = int(num_envs)
        self._q1 = _FailureQFunction(self._obs_dim, self._act_dim, self.hidden_sizes, self.dropout).to(self.device)
        self._q2 = _FailureQFunction(self._obs_dim, self._act_dim, self.hidden_sizes, self.dropout).to(self.device)
        self._q1_targ = _FailureQFunction(self._obs_dim, self._act_dim, self.hidden_sizes, self.dropout).to(self.device)
        self._q2_targ = _FailureQFunction(self._obs_dim, self._act_dim, self.hidden_sizes, self.dropout).to(self.device)
        self._q1_targ.load_state_dict(self._q1.state_dict())
        self._q2_targ.load_state_dict(self._q2.state_dict())

        q_params = list(self._q1.parameters()) + list(self._q2.parameters())
        self._optim = torch.optim.Adam(q_params, lr=self.lr)

        self._obs_buf = torch.zeros((self.buffer_size, self._obs_dim), dtype=torch.float32, device="cpu")
        self._act_buf = torch.zeros((self.buffer_size, self._act_dim), dtype=torch.float32, device="cpu")
        self._obs2_buf = torch.zeros((self.buffer_size, self._obs_dim), dtype=torch.float32, device="cpu")
        self._next_act_buf = torch.zeros((self.buffer_size, self._act_dim), dtype=torch.float32, device="cpu")
        self._has_next_act = torch.zeros((self.buffer_size,), dtype=torch.bool, device="cpu")
        self._fail_buf = torch.zeros((self.buffer_size,), dtype=torch.float32, device="cpu")
        self._done_buf = torch.zeros((self.buffer_size,), dtype=torch.float32, device="cpu")
        self._token_buf = torch.full((self.buffer_size,), -1, dtype=torch.long, device="cpu")
        self._open_refs = [None for _ in range(self._num_envs)]
        self._initialized = True

    def _polyak_update(self):
        with torch.no_grad():
            for p, p_t in zip(self._q1.parameters(), self._q1_targ.parameters()):
                p_t.data.mul_(self.polyak)
                p_t.data.add_((1.0 - self.polyak) * p.data)
            for p, p_t in zip(self._q2.parameters(), self._q2_targ.parameters()):
                p_t.data.mul_(self.polyak)
                p_t.data.add_((1.0 - self.polyak) * p.data)

    def _store_transition(self, obs, act, obs2, fail, done):
        idx = self._buf_idx
        token = self._token_counter
        self._token_counter += 1

        self._obs_buf[idx] = obs
        self._act_buf[idx] = act
        self._obs2_buf[idx] = obs2
        self._next_act_buf[idx] = 0.0
        self._has_next_act[idx] = False
        self._fail_buf[idx] = float(fail)
        self._done_buf[idx] = float(done)
        self._token_buf[idx] = token

        self._buf_idx = (self._buf_idx + 1) % self.buffer_size
        self._buf_count = min(self.buffer_size, self._buf_count + 1)
        return (idx, int(token))

    def _set_next_action(self, ref, next_action):
        idx, token = ref
        if idx < 0 or idx >= self.buffer_size:
            return
        if int(self._token_buf[idx].item()) != int(token):
            return
        self._next_act_buf[idx] = torch.as_tensor(next_action, dtype=torch.float32, device="cpu")
        self._has_next_act[idx] = True

    def _sample_batch_indices(self, batch_size: int):
        if self._buf_count <= 0:
            raise ValueError("Cannot sample replay indices: empty buffer.")
        if self.pos_fraction <= 0.0:
            return torch.randint(0, self._buf_count, (batch_size,), dtype=torch.long)
        fail_mask = (self._fail_buf[: self._buf_count] > 0.5)
        pos_idx = torch.nonzero(fail_mask, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(~fail_mask, as_tuple=False).squeeze(-1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return torch.randint(0, self._buf_count, (batch_size,), dtype=torch.long)

        pos_count = int(batch_size * self.pos_fraction)
        if self.pos_fraction > 0.0 and pos_count == 0:
            pos_count = 1
        pos_count = min(max(pos_count, 1), batch_size)
        neg_count = batch_size - pos_count

        pos_pick = pos_idx[
            torch.randint(0, pos_idx.numel(), (pos_count,), dtype=torch.long)
        ]
        if neg_count > 0:
            neg_pick = neg_idx[
                torch.randint(0, neg_idx.numel(), (neg_count,), dtype=torch.long)
            ]
            idx = torch.cat([pos_pick, neg_pick], dim=0)
        else:
            idx = pos_pick
        perm = torch.randperm(idx.shape[0])
        return idx[perm]

    def _extract_features(self, obs):
        if obs is None:
            raise ValueError("obs is required for failure predictor.")
        if isinstance(obs, dict):
            key = self.obs_key or self.default_obs_key
            if key is None:
                raise ValueError("obs_key is not set and no default_obs_key provided.")
            if key not in obs:
                raise KeyError(f"obs_key '{key}' not found in obs.")
            feats = obs[key]
        else:
            feats = obs
        return torch.as_tensor(feats, dtype=torch.float32, device=self.device)

    def _flatten_features(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim == 0:
            return feats.reshape(1, 1)
        if feats.ndim == 1:
            return feats.unsqueeze(0)
        return feats.reshape(feats.shape[0], -1)

    def _to_bool_tensor(self, value, num_envs: int):
        if value is None:
            return torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        t = torch.as_tensor(value, device=self.device).flatten()
        if t.numel() == 1:
            t = t.repeat(num_envs)
        if t.numel() < num_envs:
            padded = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
            padded[: t.numel()] = t.to(dtype=torch.bool)
            return padded
        return t[:num_envs].to(dtype=torch.bool)

    def _num_envs_from_obs(self, obs):
        feats = self._flatten_features(self._extract_features(obs))
        return int(feats.shape[0])

    def _compute_done_and_failure_masks(self, obs, reward, done, info, num_envs: int):
        """Infer per-env terminal and failure-terminal masks.

        Priority for failure signal:
        1) explicit keys in info (failure/failed/task_failed/unsafe/out_of_reach)
        2) inverse of in_success_region if available
        3) fallback to done_mask
        """
        done_mask = self._to_bool_tensor(done, num_envs)
        failure_mask = None
        if isinstance(info, dict):
            for key in ("failure", "failed", "task_failed", "unsafe", "out_of_reach"):
                if key in info:
                    failure_mask = self._to_bool_tensor(info[key], num_envs)
                    break
            if "done" in info:
                done_mask = done_mask | self._to_bool_tensor(info["done"], num_envs)
            if "terminated" in info:
                done_mask = done_mask | self._to_bool_tensor(info["terminated"], num_envs)
            if "timed_out" in info:
                done_mask = done_mask | self._to_bool_tensor(info["timed_out"], num_envs)
            if "time_out" in info:
                done_mask = done_mask | self._to_bool_tensor(info["time_out"], num_envs)
            if failure_mask is None and "in_success_region" in info:
                success = self._to_bool_tensor(info["in_success_region"], num_envs)
                failure_mask = ~success
        if failure_mask is None:
            failure_mask = done_mask.clone()
        failure_done = done_mask & failure_mask
        return done_mask, failure_done
