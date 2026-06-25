"""Distillation warm-start bootstrap utilities.

Warm-start pipeline (2 phases only):
1) Collect teacher rollout data and save it explicitly.
2) Fit the success-value critic from that collected data.

BC is intentionally removed from warm-start.
"""

import os
import time

import torch

from dexsafedagger_lab.distillation.utils.loss_utils import weighted_l2


class DistillWarmStart:
    """Implements a 2-phase warm start: collect data, then fit the success critic."""

    def __init__(self, agent):
        self.a = agent

    def _tb_add_scalar(self, tag, value, step=0):
        if self.a.rank != 0:
            return
        if not hasattr(self.a, "writer") or self.a.writer is None:
            return
        self.a.writer.add_scalar(tag, float(value), int(step))

    def _print_summary(self, title, rows, *, prefix="[WarmStart]"):
        if self.a.rank != 0:
            return
        print(f"{prefix} {title}", flush=True)
        for key, value in rows:
            print(f"{prefix}   {key:<28} {value}", flush=True)

    def _snapshot_warm_obs(self, obs, env_indices):
        obs_snapshot = {}
        capture_keys = [
            "predictor_transition",
            self.a.student_obs_type,
            self.a.teacher_obs_type,
        ]
        for key in capture_keys:
            if key not in obs:
                raise KeyError(f"[WarmStart] Required obs key '{key}' is missing.")
            tensor = obs[key].detach().index_select(0, env_indices).to("cpu")
            obs_snapshot[key] = tensor
        return obs_snapshot

    def _vlm_advisor_enabled(self):
        advisor = getattr(self.a, "vlm_threshold_advisor", None)
        return advisor is not None and bool(getattr(advisor, "enabled", False))

    def _warm_start_l2_per_env(self, obs, teacher_out):
        if not self._vlm_advisor_enabled():
            return None
        student_out = self.a.get_actions(obs, "student")
        weights = 1 / teacher_out["sigmas"][0]
        weights = weights ** 2
        return weighted_l2(student_out["mus"], teacher_out["mus"], weights)

    def _seed_vlm_visual_buffer_from_warm_start(self, *, obs, l2_loss_per_env, out_of_reach, step):
        advisor = getattr(self.a, "vlm_threshold_advisor", None)
        if advisor is None or not advisor.enabled or not getattr(advisor, "visual_buffer_enabled", True):
            return
        if l2_loss_per_env is None or not isinstance(obs, dict):
            return

        image_key = getattr(advisor, "visual_image_key", "img_left")
        image_batch = obs.get(image_key)
        if image_batch is None and image_key != "img_left":
            image_batch = obs.get("img_left")
            image_key = "img_left"
        if image_batch is None:
            image_batch = obs.get("rgb")
            image_key = "rgb"
        if image_batch is None or not torch.is_tensor(image_batch) or image_batch.ndim != 4:
            return

        unsafe_flat = torch.as_tensor(out_of_reach, device=self.a.device, dtype=torch.bool).reshape(-1)
        unsafe_ids = unsafe_flat.nonzero(as_tuple=False).flatten()
        if unsafe_ids.numel() == 0:
            return

        l2_flat = torch.as_tensor(l2_loss_per_env, device=self.a.device, dtype=torch.float32).reshape(-1)
        max_new = max(1, int(getattr(advisor, "visual_captures_per_step", 2)))
        if unsafe_ids.numel() > max_new:
            unsafe_l2 = l2_flat[unsafe_ids]
            order = torch.argsort(unsafe_l2, descending=True)[:max_new]
            unsafe_ids = unsafe_ids[order]

        for env_id_t in unsafe_ids:
            env_id = int(env_id_t.item())
            if env_id >= int(image_batch.shape[0]):
                continue
            data_url = self.a._vlm_image_to_data_url(image_batch[env_id], advisor)
            if data_url is None:
                continue
            sample = {
                "image_data_url": data_url,
                "image_key": image_key,
                "source": "warmstart_unsafe",
                "step": int(step),
                "frame": int(getattr(self.a, "frame", 0)),
                "env_id": env_id,
                "teacher_student_l2": (
                    float(l2_flat[env_id].item()) if env_id < l2_flat.numel() else None
                ),
                "predictor_success": None,
                "unsafe": True,
                "l2_threshold": float(self.a._current_unsafe_l2_threshold()),
                "success_threshold": (
                    float(self.a.failure_predictor.success_threshold)
                    if self.a.failure_predictor is not None and self.a.failure_predictor.enabled
                    else None
                ),
                "stage": "warmstart",
            }
            if hasattr(self.a, "_vlm_object_metadata"):
                sample.update(self.a._vlm_object_metadata(env_id))
            self.a._vlm_visual_buffer.append(sample)

        max_size = int(getattr(advisor, "visual_buffer_size", 64))
        if len(self.a._vlm_visual_buffer) > max_size:
            self.a._vlm_visual_buffer = self.a._vlm_visual_buffer[-max_size:]

    def _save_collected_data(self, collected_samples):
        if self.a.rank != 0:
            return
        if self.a.warm_start_save_path is None:
            raise ValueError("[WarmStart] save_collected_data is enabled but no save_path was resolved.")
        samples_to_save = []
        positive_count = 0
        total_count = 0
        positive_label_key = "out_of_reach"
        for sample in collected_samples:
            if positive_label_key not in sample:
                raise KeyError(
                    f"[WarmStart] Collected sample is missing required label key '{positive_label_key}'."
                )
            samples_to_save.append(sample)
            positive_label = sample[positive_label_key]
            mask = positive_label.detach().to(dtype=torch.bool).reshape(-1)
            positive_count += int(mask.sum().item())
            total_count += int(mask.numel())
        payload = {
            "metadata": {
                "student_obs_type": self.a.student_obs_type,
                "teacher_obs_type": self.a.teacher_obs_type,
                "safety_obs_key": "predictor_transition",
                "collect_steps": int(self.a.warm_start_collect_steps),
                "num_saved_samples": int(len(samples_to_save)),
                "positive_label_key": positive_label_key,
            },
            "samples": samples_to_save,
        }
        save_dir = os.path.dirname(self.a.warm_start_save_path)
        if len(save_dir) > 0:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(payload, self.a.warm_start_save_path)
        positive_pct = 100.0 * positive_count / max(1, total_count)
        self._print_summary(
            "Snapshot dataset saved",
            [
                ("steps", len(samples_to_save)),
                ("env_step_samples", total_count),
                ("positive_label_key", positive_label_key),
                ("positive_labels", f"{positive_count}/{total_count} ({positive_pct:.2f}%)"),
                ("path", self.a.warm_start_save_path),
            ],
        )

    def _warm_start_collect(self, obs):
        collected_samples = []
        all_env_ids = torch.arange(self.a.num_envs, dtype=torch.long, device=self.a.device)
        episode_ids = torch.zeros(self.a.num_envs, dtype=torch.long, device=self.a.device)
        if self.a.rank == 0:
            print(
                f"[WarmStart] Collecting bootstrap data for {self.a.warm_start_collect_steps} steps.",
                flush=True,
            )
        collect_start_t = time.time()
        progress_interval = max(1, self.a.warm_start_collect_steps // 10)

        with torch.no_grad():
            for step in range(self.a.warm_start_collect_steps):
                pre_step_obs = obs
                teacher_out = self.a.get_actions(obs, "teacher")
                teacher_actions = teacher_out["actions"].detach()
                l2_loss_per_env = self._warm_start_l2_per_env(obs, teacher_out)
                obs_snapshot = self._snapshot_warm_obs(obs, all_env_ids)
                collected_samples.append(
                    {
                        "step": int(step),
                        "obs": obs_snapshot,
                        "episode_id": episode_ids.detach().to("cpu", dtype=torch.long),
                        "teacher_mus": teacher_out["mus"].detach().to("cpu"),
                        "teacher_sigmas": teacher_out["sigmas"].detach().to("cpu"),
                        "teacher_actions": teacher_actions.to("cpu"),
                    }
                )

                obs, rew, out_of_reach, timed_out, _ = self.a.env.step(teacher_actions)
                lift_success = self.a._compute_lift_success_mask(
                    out_of_reach=out_of_reach,
                    timed_out=timed_out,
                )
                sample = collected_samples[-1]
                sample["reward"] = rew.detach().to("cpu")
                sample["out_of_reach"] = out_of_reach.detach().to("cpu")
                sample["timed_out"] = timed_out.detach().to("cpu")
                sample["lift_success"] = lift_success.detach().to("cpu")
                self._seed_vlm_visual_buffer_from_warm_start(
                    obs=pre_step_obs,
                    l2_loss_per_env=l2_loss_per_env,
                    out_of_reach=out_of_reach,
                    step=step,
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
                if len(done_idx) > 0:
                    episode_ids[done_idx.flatten()] += 1
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
            elapsed = time.time() - collect_start_t
            env_steps = self.a.warm_start_collect_steps * self.a.num_envs
            print(
                f"[WarmStart] Collection complete: steps={self.a.warm_start_collect_steps}, "
                f"envs={self.a.num_envs}, env_step_samples={env_steps}, elapsed={elapsed:.1f}s",
                flush=True,
            )
            if self._vlm_advisor_enabled():
                print(
                    "[WarmStart] VLM visual buffer seeded: "
                    f"{len(getattr(self.a, '_vlm_visual_buffer', []))} samples.",
                    flush=True,
                )
        return obs, collected_samples

    def _run_predictor_overfit_test(self):
        if not getattr(self.a, "warm_start_predictor_overfit_test", False):
            return None
        fp = getattr(self.a, "failure_predictor", None)
        if fp is None or not getattr(fp, "enabled", False):
            if self.a.rank == 0:
                print("[WarmStart] Predictor overfit test skipped: predictor disabled.", flush=True)
            return None
        if not getattr(fp, "_initialized", False):
            if self.a.rank == 0:
                print("[WarmStart] Predictor overfit test skipped: predictor not initialized.", flush=True)
            return None

        buf_count = int(getattr(fp, "_buf_count", 0))
        if buf_count <= 0:
            if self.a.rank == 0:
                print("[WarmStart] Predictor overfit test skipped: empty replay buffer.", flush=True)
            return None

        max_samples = int(getattr(self.a, "warm_start_predictor_overfit_max_samples", 8192))
        chunk_size = int(getattr(self.a, "warm_start_predictor_overfit_chunk_size", 1024))
        sample_count = min(buf_count, max_samples)
        if sample_count < buf_count:
            idx = torch.randperm(buf_count, device="cpu")[:sample_count]
        else:
            idx = torch.arange(buf_count, dtype=torch.long, device="cpu")

        obs = fp._obs_buf[idx]
        act = fp._act_buf[idx]
        labels = fp._success_buf[idx].to(dtype=torch.float32).reshape(-1)

        pred_chunks = []
        with torch.no_grad():
            for start in range(0, sample_count, chunk_size):
                end = min(sample_count, start + chunk_size)
                pred = fp.predict_success(obs[start:end], act[start:end])
                if pred is None:
                    raise RuntimeError(
                        "[WarmStart] Predictor overfit test failed: predict_success returned None."
                    )
                pred_chunks.append(pred.detach().to(device="cpu", dtype=torch.float32).reshape(-1))
        preds = torch.cat(pred_chunks, dim=0)

        label_pos = labels > 0.5
        label_neg = ~label_pos
        mse = float(((preds - labels) ** 2).mean().item())
        mae = float((preds - labels).abs().mean().item())
        acc = float(((preds > 0.5) == label_pos).to(dtype=torch.float32).mean().item())
        pos_frac = float(label_pos.to(dtype=torch.float32).mean().item())
        pred_pos_mean = float(preds[label_pos].mean().item()) if bool(label_pos.any().item()) else None
        pred_neg_mean = float(preds[label_neg].mean().item()) if bool(label_neg.any().item()) else None

        metrics = {
            "sample_count": int(sample_count),
            "buffer_count": int(buf_count),
            "mse": mse,
            "mae": mae,
            "acc_at_0_5": acc,
            "label_pos_frac": pos_frac,
        }
        if pred_pos_mean is not None:
            metrics["pred_pos_mean"] = pred_pos_mean
        if pred_neg_mean is not None:
            metrics["pred_neg_mean"] = pred_neg_mean

        self._tb_add_scalar("warmstart/predictor_overfit/mse", mse, 0)
        self._tb_add_scalar("warmstart/predictor_overfit/mae", mae, 0)
        self._tb_add_scalar("warmstart/predictor_overfit/acc_at_0_5", acc, 0)
        self._tb_add_scalar("warmstart/predictor_overfit/label_pos_frac", pos_frac, 0)
        self._tb_add_scalar("warmstart/predictor_overfit/sample_count", sample_count, 0)
        self._tb_add_scalar("warmstart/predictor_overfit/buffer_count", buf_count, 0)
        if pred_pos_mean is not None:
            self._tb_add_scalar("warmstart/predictor_overfit/pred_pos_mean", pred_pos_mean, 0)
        if pred_neg_mean is not None:
            self._tb_add_scalar("warmstart/predictor_overfit/pred_neg_mean", pred_neg_mean, 0)
        return metrics

    def _warm_start_fit_failure_predictor(self, collected_samples):
        if len(collected_samples) > 0:
            safety_obs_key = "predictor_transition"
            for step, sample in enumerate(collected_samples):
                obs_dict = sample["obs"]
                if safety_obs_key not in obs_dict:
                    raise KeyError(
                        f"[WarmStart] Collected dataset sample {step} is missing '{safety_obs_key}'."
                    )
                # Fit the success critic on full warm-start transitions (all envs/steps).
                current_obs = {
                    safety_obs_key: obs_dict[safety_obs_key],
                }
                next_obs = current_obs
                if step + 1 < len(collected_samples):
                    next_obs_dict = collected_samples[step + 1]["obs"]
                    next_obs = {
                        safety_obs_key: next_obs_dict[safety_obs_key],
                    }

                if self.a.failure_predictor is not None and self.a.failure_predictor.enabled:
                    teacher_actions = sample["teacher_actions"]
                    reward = sample["reward"]
                    out_of_reach = sample["out_of_reach"]
                    timed_out = sample["timed_out"]
                    done_mask = out_of_reach | timed_out
                    fp_info = {
                        "out_of_reach": out_of_reach,
                        "timed_out": timed_out,
                        "lift_success": sample["lift_success"],
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
        predictor_overfit_metrics = self._run_predictor_overfit_test()

        if self.a.rank == 0:
            fp = getattr(self.a, "failure_predictor", None)
            replay_count = int(getattr(fp, "_buf_count", 0)) if fp is not None else 0
            train_status = "disabled"
            if predictor_fit_enabled:
                train_status = "trained" if len(predictor_losses) > 0 else "skipped_not_enough_samples"
            mean_loss = (
                f"{sum(predictor_losses) / len(predictor_losses):.6f}"
                if len(predictor_losses) > 0
                else "n/a"
            )
            rows = [
                ("status", train_status),
                ("replay_samples", replay_count),
                ("requested_updates", int(self.a.warm_start_predictor_train_steps)),
                ("effective_updates", predictor_updates),
                ("mean_loss", mean_loss),
            ]
            if predictor_overfit_metrics is not None:
                rows.extend(
                    [
                        (
                            "overfit_samples",
                            f"{predictor_overfit_metrics.get('sample_count', 0)}/"
                            f"{predictor_overfit_metrics.get('buffer_count', 0)}",
                        ),
                        ("overfit_mse", f"{predictor_overfit_metrics.get('mse', 0.0):.6f}"),
                        ("overfit_mae", f"{predictor_overfit_metrics.get('mae', 0.0):.6f}"),
                        ("overfit_acc@0.5", f"{predictor_overfit_metrics.get('acc_at_0_5', 0.0):.4f}"),
                        ("label_pos_frac", f"{predictor_overfit_metrics.get('label_pos_frac', 0.0):.4f}"),
                    ]
                )
                if "pred_pos_mean" in predictor_overfit_metrics:
                    rows.append(("pred_pos_mean", f"{predictor_overfit_metrics['pred_pos_mean']:.4f}"))
                if "pred_neg_mean" in predictor_overfit_metrics:
                    rows.append(("pred_neg_mean", f"{predictor_overfit_metrics['pred_neg_mean']:.4f}"))
            self._print_summary("Predictor fitting summary", rows)
        if predictor_fit_enabled and len(predictor_losses) > 0:
            self._tb_add_scalar(
                "warmstart/predictor/loss_mean",
                sum(predictor_losses) / len(predictor_losses),
                0,
            )
        if predictor_fit_enabled and predictor_updates > 0:
            predictor_fit_status = 4
        # Unified warm-start fitting status code:
        # 0 = fail/no-op
        # 4 = success-value critic
        self._tb_add_scalar("warmstart/model_fit/status_code", predictor_fit_status, 0)

    def run_offline_stage(self, obs):
        if not self.a.warm_start_enabled or self.a.warm_start_collect_steps <= 0:
            return obs
        if not self.a.warm_start_save_collected_data:
            raise ValueError(
                "Warm-start pipeline requires explicit dataset saving. "
                "Set warm_start.save_collected_data=true."
            )
        # Hardcode warm-start predictor features to consume predictor transition obs.
        if self.a.failure_predictor is not None and self.a.failure_predictor.enabled:
            self.a.failure_predictor.obs_key = "predictor_transition"
            self.a.failure_predictor.default_obs_key = "predictor_transition"
        if self.a.rank == 0:
            print("[WarmStart] Phase 1/2: collect warm-start rollouts.", flush=True)
        obs, collected_samples = self._warm_start_collect(obs)
        self._save_collected_data(collected_samples)
        if self.a.rank == 0:
            print("[WarmStart] Phase 2/2: warm-fit success-value critic.", flush=True)
        self._warm_start_fit_failure_predictor(collected_samples)
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
