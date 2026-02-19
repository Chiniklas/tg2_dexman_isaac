"""Distillation warm-start bootstrap utilities."""

import torch

from dextrah_lab.distillation_new.loss_utils import gaussian_kl, gaussian_nll


class DistillWarmStart:
    """Implements a 3-phase warm start: collect, BC pretrain, safety-model fit."""

    def __init__(self, agent):
        self.a = agent

    def _expand_rnn_state_batch(self, state, batch_size):
        if state.dim() == 2:
            curr = state.shape[0]
            if curr == batch_size:
                return state
            if curr == 1:
                return state.repeat(batch_size, 1)
            idx = torch.arange(batch_size, device=state.device) % curr
            return state.index_select(0, idx)
        curr = state.shape[1]
        if curr == batch_size:
            return state
        if curr == 1:
            rep_shape = [1] * state.dim()
            rep_shape[1] = batch_size
            return state.repeat(*rep_shape)
        idx = torch.arange(batch_size, device=state.device) % curr
        return state.index_select(1, idx)

    def _default_student_rnn_states_for_batch(self, batch_size):
        states = self.a.student_model.get_default_rnn_state()
        states = [s.to(self.a.device) for s in states]
        return [self._expand_rnn_state_batch(s, batch_size) for s in states]

    def _build_student_batch_from_obs(self, obs_batch, is_train=True):
        batch_size = int(obs_batch[self.a.student_obs_type].shape[0])
        batch_dict = {
            "is_train": bool(is_train),
            "obs": obs_batch[self.a.student_obs_type].to(self.a.device),
            "prev_actions": torch.zeros(
                (batch_size, self.a.num_actions_student),
                dtype=torch.float32,
                device=self.a.device,
            ),
            "finetune_backbone": False,
        }
        if "img" in obs_batch:
            batch_dict["img"] = obs_batch["img"].to(self.a.device)
            if "rgb" in obs_batch:
                batch_dict["rgb_data"] = obs_batch["rgb"].to(self.a.device)
                batch_dict["rgb"] = obs_batch["rgb"].to(self.a.device)
        if "img_left" in obs_batch:
            batch_dict["img_left"] = obs_batch["img_left"].to(self.a.device)
            batch_dict["img_right"] = obs_batch["img_right"].to(self.a.device)
        if self.a.is_rnn:
            batch_dict["rnn_states"] = self._default_student_rnn_states_for_batch(batch_size)
            batch_dict["seq_length"] = 1
            batch_dict["rnn_masks"] = None
        return batch_dict

    def _compute_imitation_loss_from_teacher_targets(
        self, actions_student, teacher_mus, teacher_sigmas, teacher_actions
    ):
        weights = 1 / teacher_sigmas[0]
        weights = weights ** 2
        if self.a.imitation_loss_type == "kl":
            kl_per_env, _, _ = gaussian_kl(
                actions_student["mus"],
                actions_student["sigmas"],
                teacher_mus,
                teacher_sigmas,
            )
            return self.a.reduce_loss(kl_per_env)
        if self.a.imitation_loss_type == "nll":
            nll_per_env = gaussian_nll(
                actions_student["mus"],
                actions_student["sigmas"],
                teacher_actions,
            )
            return self.a.reduce_loss(nll_per_env)
        if self.a.imitation_loss_type == "mse":
            mse_per_env = torch.mean(
                (actions_student["actions"] - teacher_actions) ** 2, dim=-1
            )
            return self.a.reduce_loss(mse_per_env)
        mu_loss = self.a.loss(
            actions_student["mus"], teacher_mus, fn="weighted_l2", weights=weights
        )
        sigma_loss = self.a.loss(actions_student["sigmas"], teacher_sigmas)
        return mu_loss + sigma_loss

    def _snapshot_warm_obs(self, obs, env_indices):
        obs_snapshot = {}
        capture_keys = [self.a.student_obs_type]
        if self.a.warm_start_bc_record_images:
            for key in ("img", "rgb", "img_left", "img_right"):
                if key in obs:
                    capture_keys.append(key)
        for key in capture_keys:
            if key not in obs:
                continue
            tensor = obs[key].detach().index_select(0, env_indices).to("cpu")
            if key in {"img", "rgb", "img_left", "img_right"}:
                tensor = tensor.to(dtype=torch.float16)
            obs_snapshot[key] = tensor
        return obs_snapshot

    def _warm_start_collect(self, obs):
        warm_samples = []
        record_steps = min(self.a.warm_start_collect_steps, self.a.warm_start_bc_record_steps)
        record_env_ids = torch.arange(
            self.a.warm_start_bc_record_envs, dtype=torch.long, device=self.a.device
        )
        if self.a.rank == 0:
            print(
                f"[WarmStart] Collecting bootstrap data for {self.a.warm_start_collect_steps} steps.",
                flush=True,
            )

        with torch.no_grad():
            for step in range(self.a.warm_start_collect_steps):
                teacher_out = self.a.get_actions(obs, "teacher")
                teacher_actions = teacher_out["actions"].detach()

                if step < record_steps and self.a.warm_start_bc_updates > 0:
                    warm_samples.append(
                        {
                            "obs": self._snapshot_warm_obs(obs, record_env_ids),
                            "teacher_mus": teacher_out["mus"].detach().index_select(0, record_env_ids).to("cpu"),
                            "teacher_sigmas": teacher_out["sigmas"].detach().index_select(0, record_env_ids).to("cpu"),
                            "teacher_actions": teacher_out["actions"].detach().index_select(0, record_env_ids).to("cpu"),
                        }
                    )

                if self.a.ood_classifier is not None and self.a.ood_classifier.enabled:
                    obs["ood_policy_embed"] = obs[self.a.student_obs_type]
                    try:
                        student_out = self.a.get_actions(obs, "student")
                        if student_out.get("embeds") is not None:
                            embeds = student_out["embeds"].detach()
                            obs["ood_embed"] = embeds
                            obs["ood_policy_embed"] = torch.cat(
                                [obs[self.a.student_obs_type], embeds], dim=-1
                            )
                    except Exception as err:
                        if self.a.rank == 0 and step == 0:
                            print(f"[WarmStart] OOD embed prep failed; using obs key only. Error: {err}", flush=True)
                    key = self.a.ood_classifier.obs_key or self.a.ood_classifier.default_obs_key
                    if not self.a.ood_classifier.initialized and key is not None and key in obs:
                        self.a.ood_classifier.init_buffer(obs)
                    self.a.ood_classifier.check_ood(obs, self.a.device)

                prev_obs = obs
                obs, rew, out_of_reach, timed_out, info = self.a.env.step(teacher_actions)

                if self.a.failure_predictor is not None and self.a.failure_predictor.enabled:
                    done_mask = out_of_reach | timed_out
                    fp_info = dict(info) if isinstance(info, dict) else {}
                    fp_info.setdefault("out_of_reach", out_of_reach)
                    fp_info.setdefault("timed_out", timed_out)
                    if getattr(self.a.failure_predictor, "supports_next_obs", False):
                        self.a.failure_predictor.add_step(
                            obs=prev_obs,
                            action=teacher_actions,
                            next_obs=obs,
                            reward=rew,
                            done=done_mask,
                            info=fp_info,
                        )
                    else:
                        self.a.failure_predictor.add_step(
                            obs=prev_obs,
                            action=teacher_actions,
                            reward=rew,
                            done=done_mask,
                            info=fp_info,
                        )

                done_idx = (out_of_reach | timed_out).nonzero(as_tuple=False)
                if self.a.is_teacher_rnn and len(done_idx) > 0:
                    if self.a.multi_teacher:
                        for states in self.a.teacher_hidden_states_pool:
                            self.a._zero_rnn_states(states, done_idx)
                    else:
                        for s in self.a.teacher_hidden_states:
                            s[:, done_idx, ...] *= 0.0
                if self.a.is_rnn and len(done_idx) > 0 and hasattr(self.a, "student_hidden_states"):
                    self.a._zero_rnn_states(self.a.student_hidden_states, done_idx)

        if self.a.rank == 0:
            print(
                f"[WarmStart] Collected {self.a.warm_start_collect_steps} rollout steps; "
                f"cached {len(warm_samples)} BC snapshots.",
                flush=True,
            )
        return obs, warm_samples

    def _warm_start_pretrain_bc(self, warm_samples):
        if self.a.warm_start_bc_updates <= 0:
            return
        if len(warm_samples) == 0:
            if self.a.rank == 0:
                print("[WarmStart] Skipping BC pretrain: no cached warm-start samples.", flush=True)
            return

        if self.a.rank == 0:
            print(f"[WarmStart] Running initial BC pretrain for {self.a.warm_start_bc_updates} updates.", flush=True)

        self.a.student_model.train()
        losses = []
        for _ in range(self.a.warm_start_bc_updates):
            sample_idx = int(torch.randint(0, len(warm_samples), (1,), device=self.a.device).item())
            sample = warm_samples[sample_idx]
            obs_batch = {}
            for key, value in sample["obs"].items():
                tensor = value.to(self.a.device)
                if tensor.dtype in {torch.float16, torch.bfloat16}:
                    tensor = tensor.to(dtype=torch.float32)
                obs_batch[key] = tensor
            if self.a.student_obs_type not in obs_batch:
                continue
            try:
                batch_dict = self._build_student_batch_from_obs(obs_batch, is_train=True)
                student_res = self.a.student_model(batch_dict)
                student_mus = student_res["mus"]
                student_sigmas = student_res["sigmas"]
                distr = torch.distributions.Normal(student_mus, student_sigmas, validate_args=False)
                student_actions = torch.clamp(distr.sample(), -1.0, 1.0)
                actions_student = {
                    "mus": student_mus,
                    "sigmas": student_sigmas,
                    "actions": student_actions,
                }
            except Exception as err:
                if self.a.rank == 0 and len(losses) == 0:
                    print(
                        "[WarmStart] BC pretrain batch build failed; "
                        "consider enabling warm_start.bc_record_images if vision inputs are required. "
                        f"Error: {err}",
                        flush=True,
                    )
                break

            teacher_mus = sample["teacher_mus"].to(self.a.device)
            teacher_sigmas = sample["teacher_sigmas"].to(self.a.device)
            teacher_actions = sample["teacher_actions"].to(self.a.device)
            bc_loss = self._compute_imitation_loss_from_teacher_targets(
                actions_student, teacher_mus, teacher_sigmas, teacher_actions
            )

            self.a.optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.a.student_model.parameters(), 1.0)
            self.a.optimizer.step()
            losses.append(float(bc_loss.detach().item()))

        if self.a.rank == 0:
            if len(losses) > 0:
                print(f"[WarmStart] Initial BC done. mean_loss={sum(losses) / len(losses):.6f}", flush=True)
            else:
                print("[WarmStart] Initial BC finished with no successful updates.", flush=True)

    def _warm_start_fit_safety_models(self):
        predictor_losses = []
        if (
            self.a.failure_predictor is not None
            and self.a.failure_predictor.enabled
            and self.a.warm_start_predictor_train_steps > 0
        ):
            for _ in range(self.a.warm_start_predictor_train_steps):
                out = self.a.failure_predictor.train_step()
                if out is None:
                    continue
                if isinstance(out, dict):
                    loss_val = out.get("loss_total", None)
                else:
                    loss_val = out
                if loss_val is not None:
                    predictor_losses.append(float(loss_val))

        ood_refit_done = False
        if (
            self.a.ood_classifier is not None
            and self.a.ood_classifier.enabled
            and self.a.ood_classifier.initialized
            and self.a.warm_start_ood_force_refit
        ):
            for fn_name in ("_refit_stats", "_refit_pca", "_train_classifier"):
                if hasattr(self.a.ood_classifier, fn_name):
                    try:
                        getattr(self.a.ood_classifier, fn_name)()
                        ood_refit_done = True
                    except Exception as err:
                        if self.a.rank == 0:
                            print(f"[WarmStart] OOD forced refit failed for {fn_name}: {err}", flush=True)
                    break

        if self.a.rank == 0:
            if len(predictor_losses) > 0:
                print(
                    f"[WarmStart] Predictor pretrain done. mean_loss={sum(predictor_losses) / len(predictor_losses):.6f}",
                    flush=True,
                )
            elif self.a.failure_predictor is not None and self.a.failure_predictor.enabled:
                print("[WarmStart] Predictor pretrain skipped or not enough samples yet.", flush=True)
            if self.a.ood_classifier is not None and self.a.ood_classifier.enabled:
                print(f"[WarmStart] OOD warm fit status: {'done' if ood_refit_done else 'no-op'}", flush=True)

    def run(self, obs):
        if not self.a.warm_start_enabled or self.a.warm_start_collect_steps <= 0:
            return obs
        if self.a.rank == 0:
            print("[WarmStart] Phase 1/3: collect warm-start rollouts.", flush=True)
        obs, warm_samples = self._warm_start_collect(obs)
        if self.a.rank == 0:
            print("[WarmStart] Phase 2/3: initial BC pretrain.", flush=True)
        self._warm_start_pretrain_bc(warm_samples)
        if self.a.rank == 0:
            print("[WarmStart] Phase 3/3: warm-fit safety models (predictor/OOD).", flush=True)
        self._warm_start_fit_safety_models()
        obs = self.a.env.reset()[0]
        self.a.init_tensors()
        if self.a.rank == 0:
            print("[WarmStart] Completed. Starting normal intervention pipeline.", flush=True)
        return obs
