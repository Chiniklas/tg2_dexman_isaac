"""Distillation warm-start bootstrap utilities.

Warm-start pipeline (2 phases only):
1) Collect teacher rollout data and save it explicitly.
2) Fit safety models (OOD / failure predictor) from that collected data.

BC is intentionally removed from warm-start.
"""

import os
import time

import torch


class DistillWarmStart:
    """Implements a 2-phase warm start: collect dataset, then fit safety models."""

    def __init__(self, agent):
        self.a = agent

    def _tb_add_scalar(self, tag, value, step=0):
        if self.a.rank != 0:
            return
        if not hasattr(self.a, "writer") or self.a.writer is None:
            return
        self.a.writer.add_scalar(tag, float(value), int(step))

    def _prepare_ood_policy_embed(self, obs, step):
        """Populate obs['ood_policy_embed'] = concat(policy, embeds) with strict checks."""
        if self.a.student_obs_type not in obs:
            raise KeyError(
                f"[WarmStart] Missing student obs key '{self.a.student_obs_type}' at step {step}."
            )
        student_out = self.a.get_actions(obs, "student")
        embeds = student_out.get("embeds", None)
        if embeds is None:
            raise RuntimeError(
                "[WarmStart] Failed to build ood_policy_embed: student policy returned no embeds "
                f"at step {step}. Fail-fast mode: no fallback is allowed."
            )
        if not torch.is_tensor(embeds):
            raise TypeError(
                "[WarmStart] Failed to build ood_policy_embed: 'embeds' is not a tensor "
                f"(type={type(embeds)}), step={step}."
            )
        embeds = embeds.detach()
        obs["ood_embed"] = embeds
        obs["ood_policy_embed"] = torch.cat([obs[self.a.student_obs_type], embeds], dim=-1)

    def _snapshot_warm_obs(self, obs, env_indices, include_images=False):
        obs_snapshot = {}
        capture_keys = [self.a.student_obs_type, "ood_policy_embed"]
        if include_images:
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

    def _save_collected_data(self, collected_samples):
        if self.a.rank != 0:
            return
        if self.a.warm_start_save_path is None:
            print("[WarmStart] save_collected_data is enabled but no save_path was resolved.", flush=True)
            return
        image_keys = {"img", "rgb", "img_left", "img_right"}
        samples_to_save = []
        for sample in collected_samples:
            obs_src = sample.get("obs", {})
            obs_dst = {}
            if "ood_policy_embed" in obs_src:
                obs_dst["ood_policy_embed"] = obs_src["ood_policy_embed"]
            if self.a.student_obs_type in obs_src:
                obs_dst[self.a.student_obs_type] = obs_src[self.a.student_obs_type]
            if self.a.warm_start_save_images:
                for key in image_keys:
                    if key in obs_src:
                        obs_dst[key] = obs_src[key]
            sample_dst = dict(sample)
            sample_dst["obs"] = obs_dst
            samples_to_save.append(sample_dst)
        payload = {
            "metadata": {
                "student_obs_type": self.a.student_obs_type,
                "safety_obs_key": "ood_policy_embed",
                "collect_steps": int(self.a.warm_start_collect_steps),
                "saved_steps": int(self.a.warm_start_save_steps),
                "saved_envs": int(self.a.warm_start_save_envs),
                "save_images": bool(self.a.warm_start_save_images),
                "num_saved_samples": int(len(samples_to_save)),
            },
            "samples": samples_to_save,
        }
        save_dir = os.path.dirname(self.a.warm_start_save_path)
        if len(save_dir) > 0:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(payload, self.a.warm_start_save_path)
        print(
            f"[WarmStart] Saved collected rollout snapshots to {self.a.warm_start_save_path} "
            f"(samples={len(samples_to_save)}).",
            flush=True,
        )

    def _warm_start_collect(self, obs):
        collected_samples = []
        save_steps = min(self.a.warm_start_collect_steps, self.a.warm_start_save_steps)
        save_env_ids = torch.arange(
            self.a.warm_start_save_envs, dtype=torch.long, device=self.a.device
        )
        if self.a.rank == 0:
            print(
                f"[WarmStart] Collecting bootstrap data for {self.a.warm_start_collect_steps} steps.",
                flush=True,
            )
        collect_start_t = time.time()
        progress_interval = max(1, min(200, self.a.warm_start_collect_steps // 20))

        with torch.no_grad():
            for step in range(self.a.warm_start_collect_steps):
                self._prepare_ood_policy_embed(obs, step)
                teacher_out = self.a.get_actions(obs, "teacher")
                teacher_actions = teacher_out["actions"].detach()

                if step < save_steps:
                    obs_snapshot = self._snapshot_warm_obs(
                        obs,
                        save_env_ids,
                        include_images=self.a.warm_start_save_images,
                    )
                    collected_samples.append(
                        {
                            "step": int(step),
                            "obs": obs_snapshot,
                            "teacher_mus": teacher_out["mus"].detach().index_select(0, save_env_ids).to("cpu"),
                            "teacher_sigmas": teacher_out["sigmas"].detach().index_select(0, save_env_ids).to("cpu"),
                            "teacher_actions": teacher_out["actions"].detach().index_select(0, save_env_ids).to("cpu"),
                        }
                    )

                obs, rew, out_of_reach, timed_out, _ = self.a.env.step(teacher_actions)
                if step < save_steps and len(collected_samples) > 0:
                    sample = collected_samples[-1]
                    sample["reward"] = rew.detach().index_select(0, save_env_ids).to("cpu")
                    sample["out_of_reach"] = out_of_reach.detach().index_select(0, save_env_ids).to("cpu")
                    sample["timed_out"] = timed_out.detach().index_select(0, save_env_ids).to("cpu")

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
                if self.a.rank == 0 and (
                    (step + 1) % progress_interval == 0
                    or (step + 1) == self.a.warm_start_collect_steps
                ):
                    elapsed = time.time() - collect_start_t
                    print(
                        f"[WarmStart] Collect progress: {step + 1}/{self.a.warm_start_collect_steps} "
                        f"steps ({100.0 * (step + 1) / max(1, self.a.warm_start_collect_steps):.1f}%), "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

        if self.a.rank == 0:
            print(
                f"[WarmStart] Collected {self.a.warm_start_collect_steps} rollout steps; "
                f"persisted {len(collected_samples)} dataset snapshots.",
                flush=True,
            )
        return obs, collected_samples

    def _warm_start_fit_safety_models(self, collected_samples):
        if len(collected_samples) > 0:
            for step, sample in enumerate(collected_samples):
                obs_dict = sample.get("obs", {})
                if "ood_policy_embed" not in obs_dict:
                    raise KeyError(
                        f"[WarmStart] Collected dataset sample {step} is missing 'ood_policy_embed'."
                    )
                current_obs = {"ood_policy_embed": obs_dict["ood_policy_embed"]}
                next_obs = current_obs
                if step + 1 < len(collected_samples):
                    next_obs_dict = collected_samples[step + 1].get("obs", {})
                    if "ood_policy_embed" in next_obs_dict:
                        next_obs = {"ood_policy_embed": next_obs_dict["ood_policy_embed"]}

                if self.a.ood_classifier is not None and self.a.ood_classifier.enabled:
                    key = self.a.ood_classifier.obs_key or self.a.ood_classifier.default_obs_key
                    if not self.a.ood_classifier.initialized and key is not None and key in current_obs:
                        self.a.ood_classifier.init_buffer(current_obs)
                    ood_unsafe = self.a.ood_classifier.check_ood(current_obs, self.a.device)
                    if ood_unsafe is not None:
                        self._tb_add_scalar(
                            "warmstart/ood/unsafe_fraction",
                            float(ood_unsafe.float().mean().item()),
                            step,
                        )
                    ood_threshold = getattr(self.a.ood_classifier, "threshold", None)
                    if ood_threshold is not None:
                        self._tb_add_scalar("warmstart/ood/threshold", float(ood_threshold), step)
                    ood_buf_count = getattr(self.a.ood_classifier, "buf_count", None)
                    if ood_buf_count is not None:
                        self._tb_add_scalar("warmstart/ood/buffer_count", int(ood_buf_count), step)

                if self.a.failure_predictor is not None and self.a.failure_predictor.enabled:
                    teacher_actions = sample["teacher_actions"]
                    reward = sample.get("reward", torch.zeros_like(teacher_actions[:, 0]))
                    out_of_reach = sample.get(
                        "out_of_reach",
                        torch.zeros((teacher_actions.shape[0],), dtype=torch.bool),
                    )
                    timed_out = sample.get(
                        "timed_out",
                        torch.zeros((teacher_actions.shape[0],), dtype=torch.bool),
                    )
                    done_mask = out_of_reach | timed_out
                    fp_info = {
                        "out_of_reach": out_of_reach,
                        "timed_out": timed_out,
                    }
                    if getattr(self.a.failure_predictor, "supports_next_obs", False):
                        self.a.failure_predictor.add_step(
                            obs=current_obs,
                            action=teacher_actions,
                            next_obs=next_obs,
                            reward=reward,
                            done=done_mask,
                            info=fp_info,
                        )
                    else:
                        self.a.failure_predictor.add_step(
                            obs=current_obs,
                            action=teacher_actions,
                            reward=reward,
                            done=done_mask,
                            info=fp_info,
                        )

        predictor_losses = []
        predictor_updates = 0
        predictor_fit_status = 0
        predictor_fit_enabled = (
            self.a.failure_predictor is not None
            and self.a.failure_predictor.enabled
            and self.a.warm_start_predictor_train_steps > 0
        )
        if predictor_fit_enabled:
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
                    self._tb_add_scalar(
                        "warmstart/predictor/loss",
                        predictor_losses[-1],
                        predictor_updates,
                    )
                    predictor_updates += 1

        # Encoded refit status:
        # 0 = fail/no-op, 1 = gaussian(_refit_stats), 2 = pca(_refit_pca), 3 = mlp(_train_classifier)
        ood_refit_status = 0
        if (
            self.a.ood_classifier is not None
            and self.a.ood_classifier.enabled
            and self.a.ood_classifier.initialized
            and self.a.warm_start_ood_force_refit
        ):
            fn_to_status = {
                "_refit_stats": 1,
                "_refit_pca": 2,
                "_train_classifier": 3,
            }
            for fn_name in ("_refit_stats", "_refit_pca", "_train_classifier"):
                if hasattr(self.a.ood_classifier, fn_name):
                    try:
                        getattr(self.a.ood_classifier, fn_name)()
                        ood_refit_status = fn_to_status[fn_name]
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
                status_name = {
                    0: "fail/no-op",
                    1: "gaussian",
                    2: "pca",
                    3: "mlp",
                }.get(ood_refit_status, str(ood_refit_status))
                print(
                    f"[WarmStart] OOD warm fit status: {status_name} (code={ood_refit_status})",
                    flush=True,
                )
        if predictor_fit_enabled and len(predictor_losses) > 0:
            self._tb_add_scalar(
                "warmstart/predictor/loss_mean",
                sum(predictor_losses) / len(predictor_losses),
                0,
            )
        self._tb_add_scalar("warmstart/ood/refit_done", ood_refit_status, 0)
        if predictor_fit_enabled and predictor_updates > 0:
            predictor_class_name = self.a.failure_predictor.__class__.__name__.lower()
            if "critic" in predictor_class_name:
                predictor_fit_status = 5
            else:
                predictor_fit_status = 4
        # Unified warm-start fitting status code:
        # 0 = fail/no-op
        # 1 = gaussian OOD, 2 = pca OOD, 3 = mlp OOD
        # 4 = predictor legacy, 5 = predictor critic
        model_fit_status = predictor_fit_status if predictor_fit_status > 0 else ood_refit_status
        self._tb_add_scalar("warmstart/model_fit/status_code", model_fit_status, 0)

    def run_offline_stage(self, obs):
        if not self.a.warm_start_enabled or self.a.warm_start_collect_steps <= 0:
            return obs
        if not self.a.warm_start_save_collected_data:
            raise ValueError(
                "Warm-start pipeline requires explicit dataset saving. "
                "Set warm_start.save_collected_data=true."
            )
        if self.a.rank == 0:
            print("[WarmStart] Phase 1/2: collect warm-start rollouts.", flush=True)
        obs, collected_samples = self._warm_start_collect(obs)
        self._save_collected_data(collected_samples)
        if self.a.rank == 0:
            print("[WarmStart] Phase 2/2: warm-fit safety models (predictor/OOD).", flush=True)
        self._warm_start_fit_safety_models(collected_samples)
        obs = self.a.env.reset()[0]
        self.a.init_tensors()
        if self.a.rank == 0 and hasattr(self.a, "writer") and self.a.writer is not None:
            self.a.writer.flush()
        if self.a.rank == 0:
            print("[WarmStart] Completed. Starting normal intervention pipeline.", flush=True)
        return obs

    def run(self, obs):
        """Backward-compatible alias."""
        return self.run_offline_stage(obs)
