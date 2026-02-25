import torch
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as vutils
import yaml
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import pathlib
import time
import math

from rl_games.common import a2c_common 
from rl_games.algos_torch import torch_ext 
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs 
from rl_games.algos_torch import central_value 
from rl_games.common import common_losses 
from rl_games.common import datasets
from rl_games.common import tr_helpers
from rl_games.common import vecenv
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.a2c_common import swap_and_flatten01
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.model_builder import ModelBuilder
from datetime import datetime
from tensorboardX import SummaryWriter
import wandb

from typing import Dict

from ood_classifier import OODGaussianBuffer, OODPCABuffer, OODMLPBuffer
from failure_predictor import FailurePredictorCritic
from failure_predictor_success_label import FailurePredictor
from dextrah_lab.distillation_new.loss_utils import (
    l2,
    weighted_l2,
)
from dextrah_lab.distillation_new.distill_warm_start import DistillWarmStart
from dextrah_lab.distillation_new.eval_utils import (
    UNSAFE_REASON_NAMES,
    classify_out_of_reach_reasons,
)

# Imitation loss:
# - "l2": weighted L2 on mus (by 1/sigma^2) + L2 on sigmas.
#
# Optional OOD config (simple diagonal Gaussian density model on a buffer):
# - ood.enabled: bool
# - ood.obs_key: observation key to use for OOD features (default: student obs_type)
# - ood.buffer_size: max number of feature vectors to store
# - ood.min_samples: warm-start threshold; below this, always mark unsafe
# - ood.update_interval: steps between refitting mean/var and threshold
# - ood.threshold_quantile: quantile for OOD threshold on buffer scores
# - ood.diag_eps: variance floor for stability
#
# Optional failure predictor config (state-action risk model):
# - failure_predictor.enabled: bool
# - failure_predictor.obs_key: observation key used by predictor (default: student obs_type)
# - failure_predictor.failure_threshold: intervention threshold in [0, 1]
# - failure_predictor.type: "critic" (default) | "legacy"
# - failure_predictor.horizon_steps: failure-within-horizon labeling window
# - failure_predictor.gamma / failure_predictor.polyak: used by critic mode
# - failure_predictor.warm_start_model_path: optional checkpoint path used to
#   save predictor after warmstart and load predictor for non-warmstart runs
#
# Optional warm-start config (2-phase bootstrap):
# - warm_start.enabled: bool
# - warm_start.collect_steps: env steps collected before normal intervention loop
# - warm_start.predictor_train_steps: offline train_step() calls for failure predictor
# - warm_start.ood_force_refit: force one post-collection OOD refit pass
# - warm_start.save_collected_data: whether to persist warm-start rollout snapshots to disk
# - warm_start.save_path: output file for saved warm-start data
#   (default: <run_dir>/bc_dataset/*.pt)
# - warm_start.save_steps: number of warm-start steps to persist (default: collect_steps)
# - warm_start.save_envs: number of env slots to persist per saved step
# - warm_start.save_images: whether saved warm-start snapshots include image keys
#
# Optional intervention switching config:
# - switch_back_min_teacher_steps: minimum consecutive teacher-control steps after
#   an unsafe trigger before allowing switch-back checks (default: 10)


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    adjusted_state_dict = {}
    num_elems = 0
    for key, value in checkpoint_state_dict.items():
        num_elems += value.numel()
        # If the key is in the model's state_dict, use it directly
        if key in model_state_dict:
            adjusted_state_dict[key] = value
        else:
            # Try inserting '_orig_mod' in different positions based on key structure
            parts = key.split(".")
            new_key_with_orig_mod = None
            
            # Try inserting '_orig_mod' before the last layer index for different cases
            parts.insert(2, "_orig_mod")
            new_key_with_orig_mod = ".".join(parts)
            
            # If adding '_orig_mod' matches a key in the model, use the modified key
            if new_key_with_orig_mod in model_state_dict:
                adjusted_state_dict[new_key_with_orig_mod] = value
            else:
                # check if removing orig_mod works
                key_no_orig_mod = key.replace("_orig_mod.", "")
                if key_no_orig_mod in model_state_dict:
                    adjusted_state_dict[key_no_orig_mod] = value
                else:
                    # Log the key that couldn't be matched, for debugging purposes
                    print(f"Could not match key: {key} -> {new_key_with_orig_mod}")
                    # If neither works, retain the original key as a fallback
                    adjusted_state_dict[key] = value
        
    print(f"Number of elements in adjusted state dict: {num_elems}")
    return adjusted_state_dict


def _reason_counts_checked(
    reason_idx: torch.Tensor,
    unsafe_mask: torch.Tensor,
    reason_names,
    reason_to_idx: dict[str, int],
    warn_label: str | None = None,
) -> tuple[dict[str, int], int]:
    unsafe_mask = unsafe_mask.to(dtype=torch.bool)
    counts = {
        str(name): int(((reason_idx == int(reason_to_idx[str(name)])) & unsafe_mask).sum().item())
        for name in reason_names
    }
    total_unsafe = int(unsafe_mask.sum().item())
    classified_total = int(sum(counts.values()))
    unknown_count = max(0, total_unsafe - classified_total)
    if unknown_count > 0:
        label = warn_label if warn_label is not None else "unsafe reason classification"
        raise RuntimeError(
            f"{label}: found {unknown_count} unclassified unsafe episodes "
            f"(unsafe_total={total_unsafe}, classified_total={classified_total}). "
            "Fail-fast mode is enabled; no fallback mapping is allowed."
        )
    return counts, total_unsafe



class SafeDagger:
    def __init__(self, env, config, summaries_dir, nn_dir, eval_env=None):
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            wp.set_device(f"cuda:{self.local_rank}")
        else:
            wp.set_device("cpu")

        self.env = env
        self.ov_env = env.env
        self.eval_env = eval_env
        self.num_envs = self.ov_env.num_envs
        self.num_actions = self.ov_env.num_actions
        self.device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.eval_every = int(self.config.get("eval_every", 0) or 0)
        self.eval_num_episodes = int(self.config.get("eval_num_episodes", 5) or 0)
        self.eval_max_steps = self.config.get("eval_max_steps", None)
        self.eval_lift_hold_s = max(0.0, float(self.config.get("eval_lift_hold_s", 0.5) or 0.0))
        base_dir = str(pathlib.Path(__file__).parent.resolve())
        self.nn_dir = os.path.join(base_dir, nn_dir)
        self.run_dir = os.path.dirname(self.nn_dir)
        if self.rank == 0:
            print(
                "SafeDagger "
                f"eval_every={self.eval_every}, "
                f"eval_num_episodes={self.eval_num_episodes}, "
                f"eval_lift_hold_s={self.eval_lift_hold_s}",
                flush=True,
            )
        self.student_network_params = self.load_param_dict(self.config["student"]["cfg"])["params"]
        self.teacher_network_params = self.load_param_dict(self.config["teacher"]["cfg"])["params"]
        self.student_network = self.load_networks(self.student_network_params)
        self.teacher_network = self.load_networks(self.teacher_network_params)

        self.value_size = 1
        self.horizon_length = self.student_network_params["config"]["horizon_length"]
        self.normalize_value = self.student_network_params["config"]["normalize_value"]
        self.normalize_input = self.student_network_params["config"]["normalize_input"]

        # get student and teacher models
        self.num_actions_student = self.num_actions
        self.student_model_config = {
            "actions_num": self.num_actions_student,
            "input_shape": (self.ov_env.num_observations,),
            "batch_size": self.num_envs,
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.teacher_model_config = {
            "actions_num": self.num_actions,
            "input_shape": (self.ov_env.num_teacher_observations,),
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            'normalize_value': self.normalize_value, 
            'normalize_input': self.normalize_input,
        }
        self.student_model = self.student_network.build(self.student_model_config).to(self.device)
        self.teacher_models = None
        self.teacher_model = None
        self.teacher_ckpt_dir = None
        self.multi_teacher = False
        teacher_ckpt = self.config["teacher"]["ckpt"]
        if teacher_ckpt is not None and os.path.isdir(teacher_ckpt):
            self.teacher_ckpt_dir = teacher_ckpt
            self.multi_teacher = True
            self.teacher_models = self._build_teacher_pool(teacher_ckpt)
            self.teacher_model = self.teacher_models[0]
        else:
            self.teacher_model = self.teacher_network.build(self.teacher_model_config).to(self.device)
        configured_loss = str(self.config.get("imitation_loss_type", "l2")).lower()
        if configured_loss != "l2":
            raise ValueError(
                f"Unsupported imitation_loss_type: {configured_loss}. Only 'l2' is supported."
            )
        self.imitation_loss_type = "l2"
        if self.rank == 0:
            print(f"Using imitation loss: {self.imitation_loss_type}")
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-4, eps=1e-8) # default lr = 1e-4
        self.num_iters = int(self.config.get("num_iters", 100_000) or 100_000)

        # load weights for student and teacher
        if self.config["student"]["ckpt"] is not None:
            self.set_weights(self.config["student"]["ckpt"], "student")
        if not self.multi_teacher and self.config["teacher"]["ckpt"] is not None:
            self.set_weights(self.config["teacher"]["ckpt"], "teacher")
        # get the observation type of the student and teacher
        self.student_obs_type = self.config["student"]["obs_type"]
        self.teacher_obs_type = self.config["teacher"]["obs_type"]
        self.ood_classifier = self._build_ood_classifier(self.config.get("ood", {}))
        self.failure_predictor = self._build_failure_predictor(
            self.config.get("failure_predictor", {})
        )
        self.is_rnn = self.student_model.is_rnn()
        self.is_teacher_rnn = self.teacher_model.is_rnn()
        if self.is_rnn:
            self.seq_length = self.student_network_params["config"]["seq_length"]
            self.seq_length = 1
            print("USING RNN")
        if self.is_teacher_rnn:
            print("USING TEACHER RNN")
        if hasattr(self.student_model.a2c_network, "is_aux") and self.student_model.a2c_network.is_aux:
            self.is_aux = True
            print("USING AUX")
        else:
            self.is_aux = False
        self.step_student_actions = True
        self.play_policy = self.config["play_policy"]
        if self.play_policy is True:
            self.step_student_actions = True

        # logging
        self.games_to_track = 100
        self.frame = 0
        self.epoch_num = 0
        self.unsafe_reason_names = UNSAFE_REASON_NAMES
        self.unsafe_reason_to_idx = {
            name: idx for idx, name in enumerate(self.unsafe_reason_names)
        }
        object_names = list(getattr(self.ov_env, "object_names", []))
        if len(object_names) == 0:
            object_names = ["object_0"]
        self.metric_object_names = tuple(str(name) for name in object_names)
        self.metric_object_tag_names = {
            name: str(name).replace("/", "_")
            for name in self.metric_object_names
        }
        object_idx = getattr(self.ov_env, "multi_object_idx", None)
        if object_idx is None:
            object_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        else:
            object_idx = torch.as_tensor(object_idx, dtype=torch.long, device=self.device).flatten()
            if object_idx.numel() < self.num_envs:
                padded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
                padded[: object_idx.numel()] = object_idx
                object_idx = padded
            elif object_idx.numel() > self.num_envs:
                object_idx = object_idx[: self.num_envs]
        self.env_object_idx = torch.clamp(
            object_idx, min=0, max=len(self.metric_object_names) - 1
        )
        self.game_rewards = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
        self.game_unsafe_terminated = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
        self.game_unsafe_reason = {
            name: torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
            for name in self.unsafe_reason_names
        }
        self.game_unsafe_terminated_by_object = {
            object_name: torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
            for object_name in self.metric_object_names
        }
        self.game_unsafe_reason_by_object = {
            object_name: {
                reason_name: torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
                for reason_name in self.unsafe_reason_names
            }
            for object_name in self.metric_object_names
        }

        if self.rank == 0:
            self.writer = SummaryWriter(summaries_dir)
            self.use_wandb = False
            summaries_dir = os.path.join(base_dir, summaries_dir)
            self.debug_dir = os.path.join(os.path.dirname(self.nn_dir), "debug")
            os.makedirs(self.debug_dir, exist_ok=True)
            if self.use_wandb:
                wandb.login(key=os.environ["WANDB_API_KEY"])
                # wandb.tensorboard.patch(root_logdir=summaries_dir)
                wandb.init(
                    project=os.environ["WANDB_PROJECT"],
                    entity=os.environ["WANDB_ENTITY"],
                    name=os.environ["WANDB_NAME"],
                    notes=os.environ["WANDB_NOTES"],
                    # sync_tensorboard=True,
                )
        else:
            self.use_wandb = False
            self.debug_dir = None
        self.debug_save_interval_s = 1.0
        sim_dt = getattr(self.ov_env.cfg, "sim_dt", None)
        if sim_dt is None and hasattr(self.ov_env.cfg, "sim"):
            sim_dt = getattr(self.ov_env.cfg.sim, "dt", 0.0)
        decimation = getattr(self.ov_env.cfg, "decimation", 1)
        self._debug_step_dt = float(sim_dt * decimation) if sim_dt is not None else 0.0
        self._debug_sim_time = 0.0
        self._debug_max_images_per_channel = 20
        self._debug_saved_left = 0
        self._debug_saved_right = 0
        self.scaler = GradScaler()
        wp.init()
        self.aux_coeff = self.ov_env.cfg.aux_coeff
        self.stereo = self.ov_env.cfg.simulate_stereo
        self.unsafe_mode = self.config.get("unsafe_mode", "l2")
        self.unsafe_l2_threshold = float(self.config.get("unsafe_l2_threshold", 0.5))
        self.switch_back_min_teacher_steps = int(
            self.config.get("switch_back_min_teacher_steps", 10)
        )
        if self.switch_back_min_teacher_steps < 0:
            self.switch_back_min_teacher_steps = 0
        if self.rank == 0 and self.unsafe_mode == "l2":
            print(
                f"Unsafe L2 threshold fixed at {self.unsafe_l2_threshold}",
                flush=True,
            )
        if self.rank == 0 and self.switch_back_min_teacher_steps > 0:
            print(
                "Teacher switch-back hold enabled: "
                f"min_teacher_steps={self.switch_back_min_teacher_steps}",
                flush=True,
            )
        if self.rank == 0 and self.unsafe_mode == "failure_predictor":
            if self.failure_predictor is None or not self.failure_predictor.enabled:
                print(
                    "Warning: unsafe_mode=failure_predictor but failure predictor is disabled.",
                    flush=True,
                )
        fp_cfg = self.config.get("failure_predictor", {}) or {}
        fp_ws_model_path = fp_cfg.get("warm_start_model_path", None)
        fp_kind = str(fp_cfg.get("type", "critic")).strip().lower()
        if fp_kind not in {"critic", "legacy"}:
            fp_kind = "critic"
        self.failure_predictor_warm_start_model_path = os.path.join(
            self.nn_dir,
            f"fp_warmstart_{fp_kind}.pt",
        )
        if self.rank == 0 and isinstance(fp_ws_model_path, str) and len(str(fp_ws_model_path).strip()) > 0:
            print(
                "Ignoring failure_predictor.warm_start_model_path and using run nn-dir path "
                "for predictor warm-start checkpoint.",
                flush=True,
            )
        if self.rank == 0 and self.failure_predictor_warm_start_model_path is not None:
            print(
                "Failure predictor warm-start model path: "
                f"{self.failure_predictor_warm_start_model_path}",
                flush=True,
            )
        warm_cfg = self.config.get("warm_start", {}) or {}
        self.warm_start_enabled = bool(warm_cfg.get("enabled", False))
        self.warm_start_collect_steps = int(warm_cfg.get("collect_steps", 0))
        self.warm_start_predictor_train_steps = int(warm_cfg.get("predictor_train_steps", 0))
        self.warm_start_predictor_overfit_test = bool(
            warm_cfg.get("predictor_overfit_test", False)
        )
        self.warm_start_predictor_overfit_max_samples = int(
            warm_cfg.get("predictor_overfit_max_samples", 8192)
        )
        self.warm_start_predictor_overfit_chunk_size = int(
            warm_cfg.get("predictor_overfit_chunk_size", 1024)
        )
        self.warm_start_ood_force_refit = bool(warm_cfg.get("ood_force_refit", True))
        self.warm_start_save_collected_data = bool(warm_cfg.get("save_collected_data", False))
        self.warm_start_save_steps = int(
            warm_cfg.get("save_steps", self.warm_start_collect_steps)
        )
        self.warm_start_save_envs = int(
            warm_cfg.get("save_envs", min(self.num_envs, 8))
        )
        self.warm_start_save_images = bool(
            warm_cfg.get("save_images", False)
        )
        save_path_cfg = warm_cfg.get("save_path", None)
        self.warm_start_save_path = None
        if isinstance(save_path_cfg, str) and len(save_path_cfg.strip()) > 0:
            save_path_resolved = save_path_cfg.strip()
            if self.rank == 0 and not os.path.isabs(save_path_resolved):
                save_path_resolved = os.path.join(self.run_dir, save_path_resolved)
            self.warm_start_save_path = save_path_resolved
        if self.warm_start_collect_steps < 0:
            self.warm_start_collect_steps = 0
        if self.warm_start_predictor_overfit_max_samples <= 0:
            self.warm_start_predictor_overfit_max_samples = 1
        if self.warm_start_predictor_overfit_chunk_size <= 0:
            self.warm_start_predictor_overfit_chunk_size = 1
        if self.warm_start_save_steps < 0:
            self.warm_start_save_steps = 0
        self.warm_start_save_steps = min(self.warm_start_save_steps, self.warm_start_collect_steps)
        if self.warm_start_save_envs <= 0:
            self.warm_start_save_envs = 1
        self.warm_start_save_envs = min(self.warm_start_save_envs, self.num_envs)
        if self.warm_start_save_collected_data and self.warm_start_save_path is None and self.rank == 0:
            warm_start_dir = os.path.join(
                self.run_dir,
                "bc_dataset",
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.warm_start_save_path = os.path.join(
                warm_start_dir, f"warm_start_collection_{timestamp}.pt"
            )
        if self.rank == 0 and self.warm_start_enabled:
            print(
                "Warm start enabled: "
                f"collect_steps={self.warm_start_collect_steps}, "
                f"predictor_train_steps={self.warm_start_predictor_train_steps}, "
                f"predictor_overfit_test={self.warm_start_predictor_overfit_test}, "
                f"predictor_overfit_max_samples={self.warm_start_predictor_overfit_max_samples}, "
                f"predictor_overfit_chunk_size={self.warm_start_predictor_overfit_chunk_size}, "
                f"ood_force_refit={self.warm_start_ood_force_refit}, "
                f"save_collected_data={self.warm_start_save_collected_data}, "
                f"save_steps={self.warm_start_save_steps}, "
                f"save_envs={self.warm_start_save_envs}, "
                f"save_images={self.warm_start_save_images}, "
                f"save_path={self.warm_start_save_path}",
                flush=True,
            )
        self.distill_warm_start = DistillWarmStart(self)
        self.finetune_backbone = False
        self.viz_imgs = False
        if self.viz_imgs:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))

            x = np.linspace(0, 50., num=self.ov_env.cfg.img_width)
            y = np.linspace(0, 50., num=self.ov_env.cfg.img_height)
            X, Y = np.meshgrid(x, y)
            if self.stereo:
                titles = ["Left RGB", "Right RGB"]
            else:
                titles = ["RGB", "Depth"]

            # Set up the first depth map visualization
            self.rendered_img1 = self.ax1.imshow(np.zeros((self.ov_env.cfg.img_height, self.ov_env.cfg.img_width, 3)), vmin=0., vmax=1.)
            self.ax1.set_title(titles[0])

            # Set up the second depth map visualization
            if self.stereo:
                self.rendered_img2 = self.ax2.imshow(np.zeros((self.ov_env.cfg.img_height, self.ov_env.cfg.img_width, 3)), vmin=0., vmax=1.)
            else:
                self.rendered_img2 = self.ax2.imshow(X, vmin=0, vmax=1.4, cmap='Greys')
            self.ax2.set_title(titles[1])

            self.fig.canvas.draw()
            plt.show(block=False)

        self.init_tensors()

    def init_tensors(self):
        # dummy variable so that calculating neglogp doesn't give error (we don't care about the value)
        self.prev_actions_student = torch.zeros((self.num_envs, self.num_actions_student), dtype=torch.float32).to(self.device)
        self.prev_actions_teacher = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32).to(self.device)

        self.current_rewards = torch.zeros(
            (self.num_envs, self.value_size), dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.current_unsafe_terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_unsafe_reason_idx = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.dones = torch.ones(
            (self.num_envs,), dtype=torch.uint8, device=self.device
        )

        self.actions_teacher = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self.device
        )

        if self.is_rnn:
            self.student_hidden_states = self.student_model.get_default_rnn_state()
            self.student_hidden_states = [s.to(self.device) for s in self.student_hidden_states]
            self.hidden_state_means = [
                RunningMeanStd((s.shape[0], s.shape[-1])).to(device=self.device, dtype=s.dtype)
                for s in self.student_hidden_states
            ]
            self.num_seqs = self.horizon_length // self.seq_length

        if self.is_teacher_rnn:
            if self.multi_teacher:
                self.teacher_hidden_states_pool = []
                for model in self.teacher_models:
                    states = model.get_default_rnn_state()
                    self.teacher_hidden_states_pool.append([s.to(self.device) for s in states])
                self.teacher_hidden_states = None
            else:
                self.teacher_hidden_states = self.teacher_model.get_default_rnn_state()
                self.teacher_hidden_states = [s.to(self.device) for s in self.teacher_hidden_states]
            # self.num_seqs = self.horizon_length // self.seq_length

        self.env_counter = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.unsafe = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.teacher_takeover_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._predictor_lift_hold_counts = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        if self.stereo:
            self.rgb_buffers_left = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            self.rgb_buffers_right = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            self.depth_buffers_left = torch.zeros(
                (self.num_envs, 1, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            self.depth_buffers_right = torch.zeros(
                (self.num_envs, 1, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
        else:
            self.rgb_buffers = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            self.depth_buffers = torch.zeros(
                (self.num_envs, 1, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)

    def _run_warm_start(self, obs):
        return self.distill_warm_start.run_offline_stage(obs)

    def _finalize_loggers(self):
        if self.rank == 0 and self.use_wandb:
            wandb.finish()
        if self.rank == 0 and hasattr(self, "writer") and self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def run_warm_start_stage(self, obs=None):
        """Run offline warm-start bootstrap only (collect/BC/safety-fit)."""
        self.student_model.train()
        if self.multi_teacher:
            for model in self.teacher_models:
                model.eval()
        else:
            self.teacher_model.eval()
        if hasattr(self, "_predictor_lift_hold_counts") and self._predictor_lift_hold_counts is not None:
            self._predictor_lift_hold_counts.zero_()
        if obs is None:
            obs = self.env.reset()[0]
        return self._run_warm_start(obs)

    def run_online_stage(self, obs=None):
        """Run online SafeDAgger intervention training and metrics logging."""
        self.student_model.train()
        if self.multi_teacher:
            for model in self.teacher_models:
                model.eval()
        else:
            self.teacher_model.eval()
        # Hardcode online failure-predictor features to student proprio only.
        if self.failure_predictor is not None and self.failure_predictor.enabled:
            self.failure_predictor.obs_key = "policy"
            self.failure_predictor.default_obs_key = "policy"
        if hasattr(self, "_predictor_lift_hold_counts") and self._predictor_lift_hold_counts is not None:
            self._predictor_lift_hold_counts.zero_()
        if obs is None:
            obs = self.env.reset()[0]

        log_counter = 0
        total_loss = 0.

        self.optimizer.zero_grad()

        num_iters = self.num_iters
        pending_fp_step = None
        if self.rank == 0:
            if self.eval_every > 0 and self.eval_num_episodes > 0:
                print(
                    f"Eval enabled: every {self.eval_every} iters, "
                    f"{self.eval_num_episodes} episodes per eval.",
                    flush=True,
                )
            else:
                print(
                    f"Eval disabled (eval_every={self.eval_every}, "
                    f"eval_num_episodes={self.eval_num_episodes}).",
                    flush=True,
                )
        while log_counter < num_iters:
            beta = 0.0
            if self.play_policy:
                self.optimizer.param_groups[0]["lr"] = 0.0

            if log_counter < 5000:
                self.finetune_backbone = False
            else:
                self.finetune_backbone = True

            if self.viz_imgs:
                if self.stereo:
                    obj_uv_left = obs["obj_uv_left"][2].clone().detach().cpu().numpy()
                    obj_uv_right = obs["obj_uv_right"][2].clone().detach().cpu().numpy()
                    obj_uv_left[0] *= self.ov_env.cfg.img_width
                    obj_uv_left[1] *= self.ov_env.cfg.img_height
                    obj_uv_right[0] *= self.ov_env.cfg.img_width
                    obj_uv_right[1] *= self.ov_env.cfg.img_height
                    # plot object uv on top of rgb, need to be int
                    obj_uv_left = obj_uv_left.astype(np.int32)
                    obj_uv_right = obj_uv_right.astype(np.int32)
                    rgb_img = obs["img_left"][2].clone().detach().cpu().numpy().transpose(1, 2, 0)
                    # rgb_img[obj_uv_left[1]-4:obj_uv_left[1]+4, obj_uv_left[0]-4:obj_uv_left[0]+4, :] = [1, 1, 1]
                    self.rendered_img1.set_data(rgb_img)
                    rgb_img = obs["img_right"][2].clone().detach().cpu().numpy().transpose(1, 2, 0)
                    # rgb_img[obj_uv_right[1]-4:obj_uv_right[1]+4, obj_uv_right[0]-4:obj_uv_right[0]+4, :] = [1, 1, 1]
                    self.rendered_img2.set_data(rgb_img)
                else:
                    rgb_img = obs["rgb"][0].clone().detach().cpu().numpy().transpose(1, 2, 0)
                    self.rendered_img1.set_data(rgb_img)
                    self.rendered_img2.set_data(obs["img"][0, 0].detach().cpu().numpy())
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            
            # left_img = (obs["img_left"][0].clone().detach().cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
            # right_img = (obs["img_right"][0].clone().detach().cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
            # Image.fromarray(left_img).save("left_img.png")
            # Image.fromarray(right_img).save("right_img.png")
            # breakpoint()
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    actions_teacher = self.get_actions(obs, "teacher")
                    self.actions_teacher = actions_teacher["actions"]

                start_time = time.time()
                # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # imgs_real = torch.load('images1.pth') #[-0.7, 0.08, 0.295]
                # imgs_real = torch.load('images2.pth') #[-0.65, 0.25,  0.3]
                # real_img_left = imgs_real['left_img']
                # real_img_right = imgs_real['right_img']
                # obs['img_left'][:] = real_img_left #[:, torch.arange(3 - 1, -1, -1), :, :]
                # obs['img_right'][:] = real_img_right #[:, torch.arange(3 - 1, -1, -1), :, :]
                actions_student = self.get_actions(obs, "student")
                embeds = actions_student.get("embeds")
                if self._requires_ood_policy_embed():
                    if embeds is None or not torch.is_tensor(embeds):
                        raise RuntimeError(
                            "ood_policy_embed is required by enabled safety modules, "
                            "but student policy did not return tensor embeds. "
                            "Fail-fast mode: no fallback is allowed."
                        )
                    embeds = embeds.detach()
                    obs["ood_embed"] = embeds
                    obs["ood_policy_embed"] = torch.cat(
                        [obs[self.student_obs_type], embeds], dim=-1
                    )
                else:
                    obs["ood_policy_embed"] = obs[self.student_obs_type]
                    if embeds is not None and torch.is_tensor(embeds):
                        embeds = embeds.detach()
                        obs["ood_embed"] = embeds
                        obs["ood_policy_embed"] = torch.cat(
                            [obs[self.student_obs_type], embeds], dim=-1
                        )
                if (
                    self.ood_classifier is not None
                    and self.ood_classifier.enabled
                    and not self.ood_classifier.initialized
                ):
                    key = self.ood_classifier.obs_key or self.ood_classifier.default_obs_key
                    if key is not None and key in obs:
                        self.ood_classifier.init_buffer(obs)
                # Critic-style failure predictor needs next_obs features.
                # Feed one step later so next_obs has ood_policy_embed materialized.
                if (
                    self.failure_predictor is not None
                    and self.failure_predictor.enabled
                    and getattr(self.failure_predictor, "supports_next_obs", False)
                    and pending_fp_step is not None
                ):
                    fp_loss = self.failure_predictor.add_step(
                        obs=pending_fp_step["obs"],
                        action=pending_fp_step["action"],
                        next_obs=obs,
                        reward=pending_fp_step["reward"],
                        done=pending_fp_step["done"],
                        info=pending_fp_step["info"],
                    )
                    pending_fp_step = None
                    if self.rank == 0 and fp_loss is not None:
                        if isinstance(fp_loss, dict):
                            loss_total = fp_loss.get("loss_total", None)
                            if loss_total is not None:
                                self.writer.add_scalar("failure_predictor/loss", float(loss_total), self.frame)
                        else:
                            self.writer.add_scalar("failure_predictor/loss", float(fp_loss), self.frame)

                aux_loss = list() if self.is_aux else [0.]
                if actions_student["aux"] is not None:
                    aux_out = actions_student["aux"]
                    self.aux_loss_names = aux_out.keys()
                    aux_gt = obs["aux_info"]
                    mask = obs["mask_left"] if self.stereo else obs["mask"]
                    # invert binary mask for depth
                    mask = ~mask
                    for aux_name in self.aux_loss_names:
                        num_vals = aux_out[aux_name].shape[-1]
                        if 'img' in aux_name:
                            num_supervised_envs = aux_out[aux_name].shape[0]
                            if "depth" in aux_name:
                                depth_min = self.ov_env.cfg.d_min
                                depth_max = self.ov_env.cfg.d_max
                                aux_out[aux_name] = aux_out[aux_name]*(depth_max - depth_min) + depth_min
                            aux_loss.append(
                                torch.mean(
                                    torch.norm(
                                        ((aux_out[aux_name] - aux_gt[aux_name])), 
                                        p=2, dim=(1,2,3)),
                                )
                            )
                            # breakpoint()
                            if self.rank == 0:
                                self.log_img(aux_out[aux_name][:5], aux_gt[aux_name][:5])
                        elif "uv" in aux_name:
                            # find uvs that are between 0 and 1
                            uv_mask = (aux_out[aux_name] >= 0) & (aux_out[aux_name] <= 1)
                            uv_mask = uv_mask.all(dim=-1)
                            aux_loss.append(
                                self.loss(
                                    aux_out[aux_name][uv_mask],
                                    aux_gt[aux_name][uv_mask].reshape(
                                        len(uv_mask), -1
                                    )
                                )
                            )
                        else:
                            aux_loss.append(
                                self.loss(aux_out[aux_name], aux_gt[aux_name].reshape(self.num_envs, -1)) #/ num_vals
                            )

                weights = 1 / actions_teacher['sigmas'][0]
                weights = weights ** 2
                l2_loss_per_env = weighted_l2(
                    actions_student["mus"], actions_teacher["mus"], weights
                )
                l2_loss_mean = l2_loss_per_env.mean()
                mu_loss = self.loss(
                    actions_student["mus"], actions_teacher["mus"],
                    fn="weighted_l2", weights=weights
                )
                sigma_loss = self.loss(actions_student["sigmas"], actions_teacher["sigmas"])
                imitation_loss = mu_loss + sigma_loss
                aux_sum = sum(aux_loss) if aux_loss else 0.0
                aux_sum_tensor = aux_sum if torch.is_tensor(aux_sum) else torch.tensor(aux_sum, device=self.device)
                total_loss_step = imitation_loss + self.aux_coeff * aux_sum_tensor
                total_loss += total_loss_step
                current_l2_threshold = self._current_unsafe_l2_threshold()
                unsafe_raw = self.check_unsafe(
                    l2_loss_per_env=l2_loss_per_env,
                    obs=obs,
                    l2_threshold=current_l2_threshold,
                    student_action=actions_student["actions"],
                )
                if self.switch_back_min_teacher_steps > 0:
                    trigger_mask = unsafe_raw & (self.teacher_takeover_steps_remaining <= 0)
                    if bool(trigger_mask.any().item()):
                        self.teacher_takeover_steps_remaining[trigger_mask] = self.switch_back_min_teacher_steps
                    teacher_hold_mask = self.teacher_takeover_steps_remaining > 0
                    self.unsafe = unsafe_raw | teacher_hold_mask
                else:
                    self.unsafe = unsafe_raw
                beta = float(self.unsafe.float().mean().item())
            # pos = torch.tensor([
            #     [self.ov_env.cfg.x_center+self.ov_env.cfg.x_width/2, self.ov_env.cfg.y_center+self.ov_env.cfg.y_width/2, 0.5],
            #     [self.ov_env.cfg.x_center-self.ov_env.cfg.x_width/2, self.ov_env.cfg.y_center-self.ov_env.cfg.y_width/2, 0.5],
            # ]).to(self.device)
            # self.ov_env._set_pos_marker(pos)
            # print(aux_out["object_pos"])
            # self.ov_env._set_pos_marker(aux_out["object_pos"])

            if self.rank == 0:
                self.log_information(
                    log_counter,
                    total_loss_step,
                    imitation_loss,
                    aux_loss,
                    aux_sum_tensor,
                    beta,
                    l2_loss_mean,
                    mu_loss,
                    sigma_loss,
                    current_l2_threshold,
                )

            log_counter += 1
            self.env_counter += 1

            if self.is_rnn:
                if log_counter % self.seq_length == 0:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), 1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    for i, s in enumerate(self.student_hidden_states):
                        self.student_hidden_states[i] = s.detach()
                    total_loss = 0.
                    torch.cuda.empty_cache()
            else:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                total_loss = 0.
            end_time = time.time()
            # print(f"Time taken for backward and step: {end_time - start_time} seconds")

            stepping_actions = actions_student["actions"] if self.step_student_actions else actions_teacher["actions"]
            if self.unsafe.any():
                stepping_actions = stepping_actions.clone()
                stepping_actions[self.unsafe] = actions_teacher["actions"][self.unsafe].to(
                    dtype=stepping_actions.dtype
                )

            prev_obs = obs
            obs, rew, out_of_reach, timed_out, info = self.env.step(
                stepping_actions.detach()
            )
            if self.switch_back_min_teacher_steps > 0:
                active_hold = self.teacher_takeover_steps_remaining > 0
                if bool(active_hold.any().item()):
                    self.teacher_takeover_steps_remaining[active_hold] -= 1
            if self.failure_predictor is not None and self.failure_predictor.enabled:
                done_mask = out_of_reach | timed_out
                lift_success = self._compute_lift_success_mask(
                    out_of_reach=out_of_reach,
                    timed_out=timed_out,
                )
                if isinstance(info, dict):
                    fp_info = dict(info)
                else:
                    fp_info = {}
                fp_info.setdefault("out_of_reach", out_of_reach)
                fp_info.setdefault("timed_out", timed_out)
                fp_info["lift_success"] = lift_success
                if getattr(self.failure_predictor, "supports_next_obs", False):
                    fp_key = getattr(self.failure_predictor, "obs_key", None) or getattr(
                        self.failure_predictor, "default_obs_key", None
                    )
                    if fp_key is None or fp_key not in prev_obs:
                        raise KeyError(
                            f"Failure predictor expected obs key '{fp_key}' in prev_obs, "
                            "but it was not found."
                        )
                    pending_fp_step = {
                        "obs": {fp_key: prev_obs[fp_key].detach().clone()},
                        "action": stepping_actions.detach(),
                        "reward": rew,
                        "done": done_mask,
                        "info": {
                            "out_of_reach": out_of_reach.detach().clone(),
                            "timed_out": timed_out.detach().clone(),
                            "lift_success": lift_success.detach().clone(),
                        },
                    }
                    fp_loss = None
                else:
                    fp_loss = self.failure_predictor.add_step(
                        obs=prev_obs,
                        action=stepping_actions.detach(),
                        reward=rew,
                        done=done_mask,
                        info=fp_info,
                    )
                if self.rank == 0 and fp_loss is not None:
                    if isinstance(fp_loss, dict):
                        loss_total = fp_loss.get("loss_total", None)
                        if loss_total is not None:
                            self.writer.add_scalar("failure_predictor/loss", float(loss_total), self.frame)
                    else:
                        self.writer.add_scalar("failure_predictor/loss", float(fp_loss), self.frame)

            if self.rank == 0 and self.debug_dir is not None and self.stereo:
                self._debug_sim_time += self._debug_step_dt
                if self._debug_sim_time >= self.debug_save_interval_s:
                    self._debug_sim_time -= self.debug_save_interval_s
                    if self._debug_saved_left < self._debug_max_images_per_channel:
                        left = obs["img_left"][0].detach().cpu()
                        vutils.save_image(
                            left, os.path.join(self.debug_dir, f"left_env0_step{log_counter:06d}.png")
                        )
                        self._debug_saved_left += 1
                    if self._debug_saved_right < self._debug_max_images_per_channel:
                        right = obs["img_right"][0].detach().cpu()
                        vutils.save_image(
                            right, os.path.join(self.debug_dir, f"right_env0_step{log_counter:06d}.png")
                        )
                        self._debug_saved_right += 1

            self.frame += self.num_envs
            self.current_rewards += rew.unsqueeze(-1)
            self.current_lengths += 1
            reason_idx = classify_out_of_reach_reasons(
                ov_env=self.ov_env,
                out_of_reach=out_of_reach,
                reason_names=self.unsafe_reason_names,
                reason_to_idx=self.unsafe_reason_to_idx,
                device=self.device,
            )
            self.current_unsafe_terminated = self.current_unsafe_terminated | out_of_reach
            classified_out_of_reach = reason_idx >= 0
            new_reason_mask = (self.current_unsafe_reason_idx < 0) & out_of_reach & classified_out_of_reach
            self.current_unsafe_reason_idx[new_reason_mask] = reason_idx[new_reason_mask]
            self.dones = out_of_reach | timed_out
            all_done_indices = self.dones.nonzero(as_tuple=False)

            if self.is_rnn and len(all_done_indices) > 0:
                if total_loss > 1e-8:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), 1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    for i, s in enumerate(self.student_hidden_states):
                        self.student_hidden_states[i] = s.detach()
                    total_loss = 0.

                for i, s in enumerate(self.student_hidden_states):
                    with torch.no_grad():
                        self.hidden_state_means[i](
                            self.student_hidden_states[i][:, all_done_indices[0]].permute((1, 0, 2))
                        )
                    self.student_hidden_states[i][:, all_done_indices] *= 0.

                self.env_counter[all_done_indices] = 0

            if self.is_teacher_rnn and len(all_done_indices) > 0:
                if self.multi_teacher:
                    for states in self.teacher_hidden_states_pool:
                        self._zero_rnn_states(states, all_done_indices)
                else:
                    for s in self.teacher_hidden_states:
                        s[:, all_done_indices, ...] *= 0.

            done_indices = all_done_indices[:]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.game_unsafe_terminated.update(self.current_unsafe_terminated[done_indices].float())
            if len(done_indices) > 0:
                done_env_ids = done_indices.squeeze(-1) if done_indices.ndim > 1 else done_indices
                done_reason_idx = self.current_unsafe_reason_idx[done_env_ids].clone()
                done_unsafe_mask = self.current_unsafe_terminated[done_env_ids].to(dtype=torch.bool)
                unknown_done_mask = done_unsafe_mask & (done_reason_idx < 0)
                if bool(unknown_done_mask.any().item()):
                    unknown_count = int(unknown_done_mask.sum().item())
                    unsafe_total = int(done_unsafe_mask.sum().item())
                    raise RuntimeError(
                        "train-step unsafe reason classification: found "
                        f"{unknown_count} unclassified unsafe done episodes "
                        f"(unsafe_total={unsafe_total}). "
                        "Fail-fast mode is enabled; no fallback mapping is allowed."
                    )
                for name, idx in self.unsafe_reason_to_idx.items():
                    self.game_unsafe_reason[name].update((done_reason_idx == idx).float())
                done_obj_idx = self.env_object_idx[done_env_ids]
                for obj_idx, object_name in enumerate(self.metric_object_names):
                    obj_mask = done_obj_idx == obj_idx
                    if not obj_mask.any():
                        continue
                    obj_done_env_ids = done_env_ids[obj_mask]
                    self.game_unsafe_terminated_by_object[object_name].update(
                        self.current_unsafe_terminated[obj_done_env_ids].float()
                    )
                    obj_done_reason_idx = done_reason_idx[obj_mask]
                    for reason_name, reason_idx in self.unsafe_reason_to_idx.items():
                        self.game_unsafe_reason_by_object[object_name][reason_name].update(
                            (obj_done_reason_idx == reason_idx).float()
                        )
            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            self.current_unsafe_terminated = self.current_unsafe_terminated & self.dones.logical_not()
            self.current_unsafe_reason_idx[done_indices] = -1
            self.actions_teacher[done_indices] *= 0.
            if len(done_indices) > 0:
                self.unsafe[done_indices] = False
                if self.switch_back_min_teacher_steps > 0:
                    self.teacher_takeover_steps_remaining[done_indices] = 0

            if not self.play_policy:
                # if (
                #     log_counter % 10000 == 0 and
                #     log_counter > 10 and
                #     self.optimizer.param_groups[0]["lr"] > 1.2*1e-4
                # ):
                #     self.optimizer.param_groups[0]["lr"] /= 1.2
                if self.rank == 0 and log_counter % 5_000 == 0:
                    ckpt_path = os.path.join(
                        self.nn_dir,
                        f"dextrah_student_{log_counter}_iters"
                    )
                    self.save(ckpt_path)

            if (
                self.eval_every > 0
                and self.eval_num_episodes > 0
                and log_counter % self.eval_every == 0
                and log_counter > 0
            ):
                if self.rank == 0:
                    print(f"Running eval at iter {log_counter}...", flush=True)
                (
                    eval_lift,
                    eval_reward,
                    eval_unsafe,
                    eval_unsafe_reason_prop,
                    eval_per_object_metrics,
                ) = self.evaluate_student(self.eval_num_episodes)
                if self.rank == 0 and eval_lift is not None:
                    self.writer.add_scalar("eval/avg/lift_success", eval_lift, self.frame)
                    self.writer.add_scalar("eval/avg/avg_reward", eval_reward, self.frame)
                    self.writer.add_scalar("eval/avg/unsafe_episode_rate", eval_unsafe, self.frame)
                    for name in self.unsafe_reason_names:
                        self.writer.add_scalar(
                            f"eval/avg/unsafe_reason_prop/{name}",
                            eval_unsafe_reason_prop.get(name, 0.0),
                            self.frame,
                        )
                    for object_name, object_metrics in eval_per_object_metrics.items():
                        object_tag_name = str(object_name).replace("/", "_")
                        self.writer.add_scalar(
                            f"eval/{object_tag_name}/lift_success",
                            object_metrics.get("lift_success", 0.0),
                            self.frame,
                        )
                        self.writer.add_scalar(
                            f"eval/{object_tag_name}/unsafe_episode_rate",
                            object_metrics.get("unsafe_episode_rate", 0.0),
                            self.frame,
                        )
                        reason_prop = object_metrics.get("unsafe_reason_prop", {})
                        for reason_name in self.unsafe_reason_names:
                            self.writer.add_scalar(
                                f"eval/{object_tag_name}/unsafe_reason_prop/{reason_name}",
                                float(reason_prop.get(reason_name, 0.0)),
                                self.frame,
                            )
                    self.writer.flush()
                    print(
                        f"Eval lift_success: {eval_lift:.3f} | avg_reward: {eval_reward:.3f} | "
                        f"unsafe_episode_rate: {eval_unsafe:.3f} | "
                        + " | ".join(
                            [f"unsafe_reason_prop/{name}: {eval_unsafe_reason_prop.get(name, 0.0):.3f}"
                             for name in self.unsafe_reason_names]
                        ),
                        flush=True,
                    )
                if self.eval_env is None:
                    obs = self.env.reset()[0]
                    self.init_tensors()

        self._finalize_loggers()
        return log_counter

    def run_pipeline(self, pipeline="full"):
        """
        Training pipeline selector:
        - both: run warmstart stage then online intervention training.
        - warmstart: run warmstart stage only.
        - safedagger: run online intervention training only.
        """
        raw_mode = str(pipeline).lower()
        mode_aliases = {
            "both": "both",
            "warmstart": "warmstart",
            "safedagger": "safedagger",
            # Backward-compatible aliases.
            "full": "both",
            "online": "safedagger",
        }
        mode = mode_aliases.get(raw_mode, None)
        if mode is None:
            raise ValueError(
                f"Unsupported pipeline mode: {pipeline}. "
                "Expected one of: warmstart, safedagger, both."
            )
        warm_obs = None
        if mode in {"both", "warmstart"}:
            if self.rank == 0:
                print(f"[Pipeline] Running warmstart stage (mode={mode}).", flush=True)
            warm_obs = self.run_warm_start_stage()
            if mode == "both" and self.failure_predictor is not None and self.failure_predictor.enabled:
                # Enforce explicit checkpoint boundary:
                # warmstart must persist predictor, then both-mode online reloads that exact artifact.
                self._save_failure_predictor_warm_start_model()
                self._load_failure_predictor_warm_start_model()
            else:
                self._save_failure_predictor_warm_start_model()
        if mode == "warmstart":
            self._finalize_loggers()
            return 0
        if mode == "safedagger":
            self._load_failure_predictor_warm_start_model()
        if self.rank == 0:
            print(f"[Pipeline] Running online stage (mode={mode}).", flush=True)
        if mode == "both":
            return self.run_online_stage(obs=warm_obs)
        return self.run_online_stage(obs=None)

    def _save_failure_predictor_warm_start_model(self):
        if self.failure_predictor is None or not self.failure_predictor.enabled:
            return False
        path = self.failure_predictor_warm_start_model_path
        if path is None:
            return False
        if not hasattr(self.failure_predictor, "save_checkpoint"):
            if self.rank == 0:
                print(
                    "[Pipeline] Failure predictor does not expose save_checkpoint(); skipping warm-start save.",
                    flush=True,
                )
            return False
        return bool(self.failure_predictor.save_checkpoint(path))

    def _load_failure_predictor_warm_start_model(self):
        if self.failure_predictor is None or not self.failure_predictor.enabled:
            return False
        path = self.failure_predictor_warm_start_model_path
        if path is None:
            if self.rank == 0:
                print(
                    "[Pipeline] No failure_predictor.warm_start_model_path set; "
                    "starting predictor from scratch for safedagger mode.",
                    flush=True,
                )
            return False
        if not hasattr(self.failure_predictor, "load_checkpoint"):
            if self.rank == 0:
                print(
                    "[Pipeline] Failure predictor does not expose load_checkpoint(); cannot load warm-start model.",
                    flush=True,
                )
            return False
        return bool(self.failure_predictor.load_checkpoint(path))

    def distill(self):
        """Backward-compatible default entrypoint (equivalent to run_pipeline('both'))."""
        return self.run_pipeline("both")

    # --- Logging and Visualization ---
    def log_information(
        self,
        log_counter,
        total_loss,
        imitation_loss=None,
        aux_loss=None,
        aux_sum=None,
        beta=None,
        l2_loss_mean=None,
        mu_loss=None,
        sigma_loss=None,
        unsafe_l2_threshold=None,
    ):
        if imitation_loss is None:
            imitation_loss = total_loss if aux_loss is None else total_loss - self.aux_coeff * sum(aux_loss)
        if aux_sum is None:
            aux_sum = sum(aux_loss) if aux_loss else 0.0
        if aux_sum is not None and not torch.is_tensor(aux_sum):
            aux_sum = torch.tensor(aux_sum, device=self.device)
        if beta is None:
            beta = 0.

        if self.game_rewards.current_size > 0:
            mean_rewards = self.game_rewards.get_mean()
            mean_lengths = self.game_lengths.get_mean()
            mean_unsafe_terminated = self.game_unsafe_terminated.get_mean()
            unsafe_episode_rate = float(np.asarray(mean_unsafe_terminated).reshape(-1)[0])
            unsafe_reason_prop = {
                name: float(np.asarray(self.game_unsafe_reason[name].get_mean()).reshape(-1)[0])
                for name in self.unsafe_reason_names
            }
            unsafe_episode_rate_by_object = {}
            unsafe_reason_prop_by_object = {}
            for object_name in self.metric_object_names:
                object_unsafe_meter = self.game_unsafe_terminated_by_object[object_name]
                if object_unsafe_meter.current_size > 0:
                    object_unsafe_rate = float(np.asarray(object_unsafe_meter.get_mean()).reshape(-1)[0])
                else:
                    object_unsafe_rate = 0.0
                unsafe_episode_rate_by_object[object_name] = object_unsafe_rate
                object_reason_rates = {}
                for reason_name in self.unsafe_reason_names:
                    reason_meter = self.game_unsafe_reason_by_object[object_name][reason_name]
                    if reason_meter.current_size > 0:
                        object_reason_rates[reason_name] = float(np.asarray(reason_meter.get_mean()).reshape(-1)[0])
                    else:
                        object_reason_rates[reason_name] = 0.0
                unsafe_reason_prop_by_object[object_name] = {
                    reason_name: object_reason_rates[reason_name]
                    for reason_name in self.unsafe_reason_names
                }
            self.mean_rewards = mean_rewards[0]
            for i in range(self.value_size):
                rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                self.writer.add_scalar(
                    rewards_name + "/step", mean_rewards[i], self.frame
                )
                self.writer.add_scalar(
                    "total_loss", total_loss.detach().cpu().numpy(), self.frame
                )
                self.writer.add_scalar(
                    "imitation_loss", imitation_loss.detach().cpu().numpy(), self.frame
                )
                if aux_sum is not None:
                    self.writer.add_scalar(
                        "aux_loss_total", aux_sum.detach().cpu().numpy(), self.frame
                    )
                if mu_loss is not None:
                    self.writer.add_scalar(
                        "mu_loss", mu_loss.detach().cpu().numpy(), self.frame
                    )
                if sigma_loss is not None:
                    self.writer.add_scalar(
                        "sigma_loss", sigma_loss.detach().cpu().numpy(), self.frame
                    )
                self.writer.add_scalar(
                    "beta", beta, self.frame
                )
                self.writer.add_scalar(
                    "train/avg/unsafe_episode_rate", unsafe_episode_rate, self.frame
                )
                for name in self.unsafe_reason_names:
                    self.writer.add_scalar(
                        f"train/avg/unsafe_reason_prop/{name}",
                        unsafe_reason_prop[name],
                        self.frame,
                    )
                for object_name in self.metric_object_names:
                    object_tag_name = self.metric_object_tag_names[object_name]
                    self.writer.add_scalar(
                        f"train/{object_tag_name}/unsafe_episode_rate",
                        unsafe_episode_rate_by_object[object_name],
                        self.frame,
                    )
                    for reason_name in self.unsafe_reason_names:
                        self.writer.add_scalar(
                            f"train/{object_tag_name}/unsafe_reason_prop/{reason_name}",
                            unsafe_reason_prop_by_object[object_name][reason_name],
                            self.frame,
                        )
                self.writer.add_scalar(
                    "train/intervention_rate", beta, self.frame
                )
                if l2_loss_mean is not None:
                    self.writer.add_scalar(
                        "l2_loss_mean", l2_loss_mean.detach().cpu().numpy(), self.frame
                    )
                if unsafe_l2_threshold is not None:
                    self.writer.add_scalar(
                        "unsafe_l2_threshold", float(unsafe_l2_threshold), self.frame
                    )
                if beta > 0.95:
                    perf = self.ov_env.in_success_region.float().mean().cpu().numpy()
                else:
                    perf = self.ov_env.in_success_region.float().mean().cpu().numpy()
                self.writer.add_scalar(
                    "in_success_region", perf, self.frame
                )
                if self.use_wandb:
                    wandb.log({
                        "in_success_region": perf,
                        "imitation_loss": imitation_loss.detach().cpu().numpy(),
                        "total_loss": total_loss.detach().cpu().numpy(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "beta": beta,
                        "train/avg/unsafe_episode_rate": unsafe_episode_rate,
                        **{
                            f"train/avg/unsafe_reason_prop/{name}": unsafe_reason_prop[name]
                            for name in self.unsafe_reason_names
                        },
                        **{
                            f"train/{self.metric_object_tag_names[object_name]}/unsafe_episode_rate":
                            unsafe_episode_rate_by_object[object_name]
                            for object_name in self.metric_object_names
                        },
                        **{
                            f"train/{self.metric_object_tag_names[object_name]}/unsafe_reason_prop/{reason_name}":
                            unsafe_reason_prop_by_object[object_name][reason_name]
                            for object_name in self.metric_object_names
                            for reason_name in self.unsafe_reason_names
                        },
                        "train/intervention_rate": beta,
                        "unsafe_l2_threshold": float(unsafe_l2_threshold) if unsafe_l2_threshold is not None else self.unsafe_l2_threshold,
                        "iteration": self.frame
                    })
                    if aux_sum is not None:
                        wandb.log({
                            "aux_loss_total": aux_sum.detach().cpu().numpy(),
                            "iteration": self.frame
                        })
                    if mu_loss is not None:
                        wandb.log({
                            "mu_loss": mu_loss.detach().cpu().numpy(),
                            "iteration": self.frame
                        })
                    if sigma_loss is not None:
                        wandb.log({
                            "sigma_loss": sigma_loss.detach().cpu().numpy(),
                            "iteration": self.frame
                        })
                if self.is_aux:
                    for idx, name in enumerate(self.aux_loss_names):
                        self.writer.add_scalar(
                            f"aux_loss_{name}", aux_loss[i].detach().cpu().numpy(), self.frame
                        )
                        if self.use_wandb:
                            wandb.log({
                                f"aux_loss_{name}": aux_loss[i].detach().cpu().numpy(),
                                "iteration": self.frame
                            })

        if log_counter % 10 == 0:
            print("="*10)
            print("ITERATION:", log_counter)
            print("LR: ", self.optimizer.param_groups[0]["lr"])
            print("Imitation Loss: ", imitation_loss)
            if aux_sum is not None:
                print("Aux Loss (total): ", aux_sum)
            if mu_loss is not None:
                print("Mu Loss: ", mu_loss)
            if sigma_loss is not None:
                print("Sigma Loss: ", sigma_loss)
            if self.is_aux:
                print("Aux Loss: ", aux_loss)
            print("Total Loss: ", total_loss)
            print("Beta: ", beta)
            if l2_loss_mean is not None:
                print("L2 Loss Mean: ", l2_loss_mean)
            if unsafe_l2_threshold is not None:
                print("Unsafe L2 Threshold: ", float(unsafe_l2_threshold))
            if self.game_rewards.current_size > 0:
                print("\tMean Rewards: ", mean_rewards)
                print("\tMean Length: ", mean_lengths)
                print("\tin_success_region: ", perf)
                print("\tunsafe_episode_rate: ", mean_unsafe_terminated)
                for name in self.unsafe_reason_names:
                    print(f"\tunsafe_reason_prop/{name}: {unsafe_reason_prop[name]:.4f}")
                fp_pos_stats = self._failure_predictor_positive_stats()
                if fp_pos_stats is not None:
                    pos_count, total_count, pos_pct = fp_pos_stats
                    print(
                        "\tfailure_predictor_pos_labeled: "
                        f"{pos_count}/{total_count} ({pos_pct:.2f}%)"
                    )

    def log_img(self, pred_images, gt_images):
        combined_images = torch.cat((pred_images, gt_images), dim=0)
        image_grid = vutils.make_grid(combined_images, nrow=pred_images.shape[0], normalize=True, scale_each=True)
        self.writer.add_image('Predictions_vs_Ground_Truth', image_grid, global_step=self.frame)
        if self.use_wandb:
            images = wandb.Image(image_grid, caption="Top: Network Pred, Bottom: GT")
            wandb.log({"predictions vs ground truth": images})

    def _failure_predictor_positive_stats(self):
        """Return (pos_count, total_count, pos_pct) for current predictor ring buffer."""
        fp = self.failure_predictor
        if fp is None or not fp.enabled:
            return None

        # Critic predictor: positives are fail-labeled transitions in replay.
        if hasattr(fp, "_fail_buf") and hasattr(fp, "_buf_count") and fp._fail_buf is not None:
            total_count = int(fp._buf_count)
            if total_count <= 0:
                return (0, 0, 0.0)
            fail_buf = fp._fail_buf[:total_count]
            if torch.is_tensor(fail_buf):
                pos_count = int((fail_buf > 0.5).to(dtype=torch.int64).sum().item())
            else:
                pos_count = int(np.sum(np.asarray(fail_buf) > 0.5))
            pos_pct = 100.0 * pos_count / max(1, total_count)
            return (pos_count, total_count, pos_pct)

        # Legacy predictor: positives among finalized/labeled entries.
        if (
            hasattr(fp, "_buffer_y")
            and hasattr(fp, "_buffer_labeled")
            and fp._buffer_y is not None
            and fp._buffer_labeled is not None
        ):
            buf_count = int(getattr(fp, "_buf_count", 0))
            if buf_count <= 0:
                return (0, 0, 0.0)
            y = fp._buffer_y[:buf_count]
            labeled = fp._buffer_labeled[:buf_count]
            if torch.is_tensor(y) and torch.is_tensor(labeled):
                labeled = labeled.to(dtype=torch.bool)
                total_count = int(labeled.sum().item())
                pos_count = int(((y > 0.5) & labeled).sum().item())
            else:
                y_np = np.asarray(y)
                lab_np = np.asarray(labeled).astype(bool)
                total_count = int(np.sum(lab_np))
                pos_count = int(np.sum((y_np > 0.5) & lab_np))
            pos_pct = 100.0 * pos_count / max(1, total_count)
            return (pos_count, total_count, pos_pct)

        return None

    # --- Safety and Unsafe Detection ---
    def check_unsafe(
        self,
        l2_loss_per_env=None,
        obs=None,
        out_of_reach=None,
        timed_out=None,
        info=None,
        l2_threshold=None,
        student_action=None,
    ):
        """Unsafe logic controlled by unsafe_mode: none | l2 | ood | failure_predictor."""
        if self.unsafe_mode == "none":
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.unsafe_mode == "ood":
            if self.ood_classifier is not None and self.ood_classifier.enabled:
                unsafe = self.ood_classifier.check_ood(obs, self.device)
                if unsafe is not None:
                    return unsafe
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.unsafe_mode == "failure_predictor":
            if self.failure_predictor is not None and self.failure_predictor.enabled:
                unsafe_pred = self.failure_predictor.should_intervene(obs, student_action)
                if unsafe_pred is None:
                    unsafe_pred = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                else:
                    unsafe_pred = unsafe_pred.to(device=self.device, dtype=torch.bool)

                # Legacy predictor: use predictor AND action-disagreement gate.
                if isinstance(self.failure_predictor, FailurePredictor):
                    if l2_loss_per_env is None:
                        return unsafe_pred
                    threshold = self.unsafe_l2_threshold if l2_threshold is None else float(l2_threshold)
                    unsafe_l2 = l2_loss_per_env > threshold
                    # return unsafe_pred & unsafe_l2
                    return unsafe_l2
                # Critic predictor: use predictor AND action-disagreement gate.
                if isinstance(self.failure_predictor, FailurePredictorCritic):
                    if l2_loss_per_env is None:
                        return unsafe_pred
                    threshold = self.unsafe_l2_threshold if l2_threshold is None else float(l2_threshold)
                    unsafe_l2 = l2_loss_per_env > threshold
                    return unsafe_pred & unsafe_l2
                raise ValueError(
                    "Unsupported failure predictor class: "
                    f"{self.failure_predictor.__class__.__name__}. "
                    "Expected FailurePredictor (legacy) or FailurePredictorCritic."
                )
            if l2_loss_per_env is not None:
                threshold = self.unsafe_l2_threshold if l2_threshold is None else float(l2_threshold)
                return l2_loss_per_env > threshold
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if l2_loss_per_env is not None:
            threshold = self.unsafe_l2_threshold if l2_threshold is None else float(l2_threshold)
            return l2_loss_per_env > threshold
        if isinstance(info, dict):
            if "unsafe" in info:
                return torch.as_tensor(info["unsafe"], device=self.device, dtype=torch.bool)
            if "unsafe_flag" in info:
                return torch.as_tensor(info["unsafe_flag"], device=self.device, dtype=torch.bool)
            if "in_unsafe_region" in info:
                return torch.as_tensor(info["in_unsafe_region"], device=self.device, dtype=torch.bool)
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _current_unsafe_l2_threshold(self):
        return self.unsafe_l2_threshold

    def _requires_ood_policy_embed(self):
        """Return True if any enabled safety module reads from 'ood_policy_embed'."""
        needs = False
        if self.ood_classifier is not None and self.ood_classifier.enabled:
            ood_key = self.ood_classifier.obs_key or self.ood_classifier.default_obs_key
            needs = needs or (ood_key == "ood_policy_embed")
        if self.failure_predictor is not None and self.failure_predictor.enabled:
            fp_default = getattr(self.failure_predictor, "default_obs_key", None)
            fp_key = getattr(self.failure_predictor, "obs_key", None) or fp_default
            needs = needs or (fp_key == "ood_policy_embed")
        return needs

    def _compute_lift_success_mask(self, out_of_reach=None, timed_out=None, ov_env=None):
        """Compute per-env hold-gated lift success aligned with eval semantics."""
        env_ref = self.ov_env if ov_env is None else ov_env
        if env_ref is None or not hasattr(env_ref, "object_pos"):
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        num_envs = int(getattr(env_ref, "num_envs", self.num_envs))
        if (
            not hasattr(self, "_predictor_lift_hold_counts")
            or self._predictor_lift_hold_counts is None
            or int(self._predictor_lift_hold_counts.shape[0]) != num_envs
        ):
            self._predictor_lift_hold_counts = torch.zeros(
                num_envs, dtype=torch.long, device=self.device
            )

        done_mask = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        for v in (out_of_reach, timed_out):
            if v is None:
                continue
            t = torch.as_tensor(v, device=self.device).reshape(-1).to(dtype=torch.bool)
            if t.numel() == 1:
                t = t.repeat(num_envs)
            if t.numel() != num_envs:
                raise ValueError(
                    f"lift-success done mask size mismatch: expected {num_envs}, got {t.numel()}."
                )
            done_mask = done_mask | t

        table_center_z = env_ref.cfg.table_cfg.init_state.pos[2]
        table_top_z = table_center_z + 0.5 * env_ref.cfg.table_size_z
        lift_height_thresh = table_top_z + getattr(env_ref.cfg, "object_height_thresh", 0.0)
        lift_success_step = env_ref.object_pos[:, 2].to(device=self.device) > lift_height_thresh

        if hasattr(env_ref, "good_grasp_mask") and env_ref.good_grasp_mask is not None:
            contact_mask = env_ref.good_grasp_mask.to(device=self.device, dtype=torch.bool)
        elif hasattr(env_ref, "object_contact_counts") and env_ref.object_contact_counts is not None:
            contact_mask = env_ref.object_contact_counts.to(device=self.device) > 0.0
        else:
            contact_mask = torch.ones_like(lift_success_step, dtype=torch.bool)
        lift_success_step = lift_success_step & contact_mask

        sim_dt = getattr(env_ref.cfg, "sim_dt", None)
        if sim_dt is None and hasattr(env_ref.cfg, "sim"):
            sim_dt = getattr(env_ref.cfg.sim, "dt", None)
        decimation = getattr(env_ref.cfg, "decimation", 1)
        step_dt = float(sim_dt * decimation) if sim_dt is not None else 0.0
        hold_steps = 1
        if self.eval_lift_hold_s > 0.0 and step_dt > 0.0:
            hold_steps = max(1, int(math.ceil(self.eval_lift_hold_s / step_dt)))

        active_envs = ~done_mask
        self._predictor_lift_hold_counts = torch.where(
            active_envs & lift_success_step,
            self._predictor_lift_hold_counts + 1,
            torch.where(active_envs, torch.zeros_like(self._predictor_lift_hold_counts), self._predictor_lift_hold_counts),
        )
        lift_success_hold = self._predictor_lift_hold_counts >= hold_steps

        # Prevent leakage across auto-reset episode boundaries.
        self._predictor_lift_hold_counts = torch.where(
            done_mask,
            torch.zeros_like(self._predictor_lift_hold_counts),
            self._predictor_lift_hold_counts,
        )
        return lift_success_hold

    # --- Safety Model Builders ---
    def _build_failure_predictor(self, cfg):
        fp_cfg = cfg or {}
        device = str(fp_cfg.get("device", self.device))
        fp_type = str(fp_cfg.get("type", "critic")).lower()
        if fp_type == "legacy":
            fp_class = FailurePredictor
        elif fp_type == "critic":
            fp_class = FailurePredictorCritic
        else:
            raise ValueError(f"Unsupported failure_predictor.type: {fp_type}")
        return fp_class(
            fp_cfg,
            device=device,
            default_obs_key=self.student_obs_type,
            rank=self.rank,
        )

    def _build_ood_classifier(self, cfg):
        ood_cfg = cfg or {}
        ood_type = str(ood_cfg.get("type", "gaussian")).lower()
        if ood_type == "gaussian":
            klass = OODGaussianBuffer
        elif ood_type == "pca":
            klass = OODPCABuffer
        elif ood_type == "mlp":
            klass = OODMLPBuffer
        else:
            raise ValueError(f"Unsupported ood.type: {ood_type}")
        return klass(ood_cfg, default_obs_key=self.student_obs_type, rank=self.rank)

    # --- Teacher Policy and Action Selection ---
    def _build_teacher_pool(self, ckpt_root):
        if not hasattr(self.ov_env, "object_names"):
            raise ValueError("Environment does not expose object_names for multi-teacher loading.")
        object_names = list(self.ov_env.object_names)
        if len(object_names) == 0:
            raise ValueError("No object names found for multi-teacher loading.")
        ckpt_map = {}
        missing = []
        for name in object_names:
            subdir = os.path.join(ckpt_root, name)
            if not os.path.isdir(subdir):
                missing.append(name)
                continue
            candidates = sorted(glob.glob(os.path.join(subdir, "*.pth")))
            if len(candidates) == 0:
                missing.append(name)
                continue
            ckpt_map[name] = candidates[0]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing teacher checkpoints for objects: {missing_str} (root: {ckpt_root})")

        models = []
        for name in object_names:
            model = self.teacher_network.build(self.teacher_model_config).to(self.device)
            self.set_weights(ckpt_map[name], "teacher", model_override=model)
            models.append(model)
        return models

    def _select_rnn_states(self, states, indices):
        indices = indices.flatten()
        selected = []
        for s in states:
            if s.dim() == 2:
                selected.append(s.index_select(0, indices))
            else:
                selected.append(s.index_select(1, indices))
        return selected

    def _writeback_rnn_states(self, states, indices, new_states):
        indices = indices.flatten()
        for i, s in enumerate(states):
            ns = new_states[i]
            if ns.dtype != s.dtype:
                ns = ns.to(dtype=s.dtype)
            if s.dim() == 2:
                s.index_copy_(0, indices, ns)
            else:
                s.index_copy_(1, indices, ns)

    def _zero_rnn_states(self, states, indices):
        if indices.numel() == 0:
            return
        indices = indices.flatten()
        for s in states:
            if s.dim() == 2:
                s[indices] = 0
            else:
                s[:, indices, ...] = 0

    def _get_actions_multi_teacher(self, obs):
        mus = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        sigmas = torch.zeros_like(mus)
        obj_indices = self.ov_env.multi_object_idx

        for obj_idx, model in enumerate(self.teacher_models):
            env_mask = obj_indices == obj_idx
            if not torch.any(env_mask):
                continue
            idx = env_mask.nonzero(as_tuple=False).flatten()
            batch_dict = {
                "is_train": False,
                "obs": obs[self.teacher_obs_type][idx],
                "prev_actions": self.prev_actions_teacher[idx],
            }
            if self.is_teacher_rnn:
                states = self._select_rnn_states(self.teacher_hidden_states_pool[obj_idx], idx)
                batch_dict["rnn_states"] = states
                batch_dict["seq_length"] = 1
                batch_dict["rnn_masks"] = None
            res_dict = model(batch_dict)
            if self.is_teacher_rnn:
                self._writeback_rnn_states(
                    self.teacher_hidden_states_pool[obj_idx],
                    idx,
                    res_dict["rnn_states"],
                )
            mus[idx] = res_dict["mus"].to(dtype=mus.dtype)
            sigmas[idx] = res_dict["sigmas"].to(dtype=sigmas.dtype)
        return mus, sigmas

    def get_actions(self, obs, policy_type):
        aux = None
        embeds = None
        if policy_type == "student":
            batch_dict = {
                "is_train": True,
                # "obs": real_obs["proprio"].to(self.device).repeat(2,1),
                "obs": obs[self.student_obs_type],
                # "observations": obs[self.student_obs_type],
                "prev_actions": self.prev_actions_student,
            }
            if "img" in obs:
                # mean_tensor = torch.mean(obs["img"], dim=(2, 3), keepdim=True)
                batch_dict["img"] = obs["img"] #- mean_tensor
                batch_dict["rgb_data"] = obs["rgb"]
                batch_dict["rgb"] = obs["rgb"]
            if "img_left" in obs:
                batch_dict["img_left"] = obs["img_left"]
                batch_dict["img_right"] = obs["img_right"]
                # batch_dict["img_left"] = real_obs["left_img"].repeat(2,1,1,1).to(self.device)
                # batch_dict["img_right"] = real_obs["right_img"].repeat(2,1,1,1).to(self.device)
            if self.is_rnn:
                # batch_dict["rnn_states"] = [real_obs["hidden_state_1"], real_obs["hidden_state_2"]]
                batch_dict["rnn_states"] = self.student_hidden_states
                batch_dict["seq_length"] = 1
                batch_dict["rnn_masks"] = None
            batch_dict["finetune_backbone"] = self.finetune_backbone
            res_dict = self.student_model(batch_dict)
            mus = res_dict["mus"]
            sigmas = res_dict["sigmas"]
            rnn_states = res_dict.get("rnn_states", None)
            if isinstance(rnn_states, (tuple, list)) and len(rnn_states) >= 3:
                if torch.is_tensor(rnn_states[2]):
                    embeds = rnn_states[2]
            # self.ov_env._set_gt_pos_marker(gt_pos.repeat(self.num_envs, 1))
            # breakpoint()
            if self.is_rnn:
                if isinstance(rnn_states, (tuple, list)):
                    states = rnn_states[0]
                else:
                    states = rnn_states
                if self.is_aux and isinstance(states, (tuple, list)):
                    self.student_hidden_states = [s for s in states]
                elif states is not None:
                    self.student_hidden_states = [s for s in states]
            if self.is_aux:
                if isinstance(rnn_states, (tuple, list)) and len(rnn_states) >= 2:
                    aux = rnn_states[1]
        else:
            if self.multi_teacher:
                mus, sigmas = self._get_actions_multi_teacher(obs)
            else:
                batch_dict = {
                    "is_train": False,
                    "obs": obs[self.teacher_obs_type],
                    "prev_actions": self.prev_actions_teacher,
                }
                if self.is_teacher_rnn:
                    batch_dict["rnn_states"] = self.teacher_hidden_states
                    batch_dict["seq_length"] = 1
                    batch_dict["rnn_masks"] = None
                res_dict = self.teacher_model(batch_dict)
                if self.is_teacher_rnn:
                    self.teacher_hidden_states = res_dict["rnn_states"]
                mus = res_dict["mus"]
                sigmas = res_dict["sigmas"]
        distr = torch.distributions.Normal(mus, sigmas, validate_args=False)
        selected_action = distr.sample().squeeze()
        # clamp selected action between 1 and -1
        selected_action = torch.clamp(selected_action, -1., 1.)

        return {
            "mus": mus,
            "sigmas": sigmas,
            "actions": selected_action,
            "aux": aux,
            "embeds": embeds,
        }

    # --- Evaluation ---
    def _get_student_actions_eval(self, obs, prev_actions, hidden_states):
        batch_dict = {
            "is_train": False,
            "obs": obs[self.student_obs_type],
            "prev_actions": prev_actions,
        }
        if "img" in obs:
            batch_dict["img"] = obs["img"]
            batch_dict["rgb_data"] = obs["rgb"]
            batch_dict["rgb"] = obs["rgb"]
        if "img_left" in obs:
            batch_dict["img_left"] = obs["img_left"]
            batch_dict["img_right"] = obs["img_right"]
        if self.is_rnn:
            batch_dict["rnn_states"] = hidden_states
            batch_dict["seq_length"] = 1
            batch_dict["rnn_masks"] = None
        batch_dict["finetune_backbone"] = False
        res_dict = self.student_model(batch_dict)
        mus = res_dict["mus"]
        sigmas = res_dict["sigmas"]
        if self.is_rnn:
            if self.is_aux:
                hidden_states = [s for s in res_dict["rnn_states"][0]]
            else:
                hidden_states = [s for s in res_dict["rnn_states"]]
        distr = torch.distributions.Normal(mus, sigmas, validate_args=False)
        selected_action = distr.sample().squeeze()
        selected_action = torch.clamp(selected_action, -1., 1.)
        return selected_action.detach(), hidden_states

    def evaluate_student(self, num_episodes):
        if num_episodes <= 0:
            return None, None, None, None, {}
        eval_env = self.eval_env if self.eval_env is not None else self.env
        eval_ov_env = eval_env.env
        if eval_ov_env.num_envs != self.num_envs:
            if self.rank == 0:
                print(
                    "Skipping eval: eval_env num_envs must match training num_envs "
                    f"({eval_ov_env.num_envs} vs {self.num_envs})."
                )
            return None, None, None, None, {}
        num_envs = eval_ov_env.num_envs
        eval_object_names = list(getattr(eval_ov_env, "object_names", []))
        if len(eval_object_names) == 0:
            eval_object_names = list(self.metric_object_names)
        if len(eval_object_names) == 0:
            eval_object_names = ["object_0"]
        eval_object_names = [str(name) for name in eval_object_names]
        eval_object_idx = getattr(eval_ov_env, "multi_object_idx", None)
        if eval_object_idx is None:
            eval_object_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        else:
            eval_object_idx = torch.as_tensor(eval_object_idx, dtype=torch.long, device=self.device).flatten()
            if eval_object_idx.numel() < num_envs:
                padded = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
                padded[: eval_object_idx.numel()] = eval_object_idx
                eval_object_idx = padded
            elif eval_object_idx.numel() > num_envs:
                eval_object_idx = eval_object_idx[:num_envs]
        eval_object_idx = torch.clamp(eval_object_idx, min=0, max=len(eval_object_names) - 1)
        eval_object_env_counts = {
            object_name: int((eval_object_idx == idx).sum().item())
            for idx, object_name in enumerate(eval_object_names)
        }
        sim_dt = getattr(eval_ov_env.cfg, "sim_dt", None)
        if sim_dt is None and hasattr(eval_ov_env.cfg, "sim"):
            sim_dt = getattr(eval_ov_env.cfg.sim, "dt", None)
        decimation = getattr(eval_ov_env.cfg, "decimation", 1)
        step_dt = float(sim_dt * decimation) if sim_dt is not None else 0.0
        hold_steps = 1
        if self.eval_lift_hold_s > 0.0 and step_dt > 0.0:
            hold_steps = max(1, int(math.ceil(self.eval_lift_hold_s / step_dt)))
        if self.rank == 0:
            print(
                f"Eval lift hold gate: {hold_steps} steps (~{self.eval_lift_hold_s:.3f}s target, dt={step_dt:.5f}s)",
                flush=True,
            )
        prev_actions = torch.zeros(
            (num_envs, self.num_actions_student),
            dtype=torch.float32,
            device=self.device,
        )
        was_training = self.student_model.training
        self.student_model.eval()
        max_steps = self.eval_max_steps
        if max_steps is None:
            max_steps = getattr(eval_ov_env, "distill_max_episode_length", None)
        if max_steps is None:
            max_steps = getattr(eval_ov_env, "max_episode_length", None)
        if max_steps is None:
            max_steps = 1000
        success_rates = []
        reward_means = []
        unsafe_rates = []
        total_reason_counts = {name: 0 for name in self.unsafe_reason_names}
        per_object_lift_series = {
            object_name: [] for object_name in eval_object_names
        }
        per_object_unsafe_rate_series = {
            object_name: [] for object_name in eval_object_names
        }
        per_object_reason_counts_total = {
            object_name: {name: 0 for name in self.unsafe_reason_names}
            for object_name in eval_object_names
        }
        with torch.no_grad():
            for _ in range(num_episodes):
                obs = eval_env.reset()[0]
                dones = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
                ever_lifted = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
                ever_unsafe_terminated = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
                unsafe_reason_idx = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
                if self.is_rnn:
                    hidden_states = [s.to(self.device) for s in self.student_model.get_default_rnn_state()]
                else:
                    hidden_states = None
                prev_actions.zero_()
                steps = 0
                reward_sum = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)
                lift_hold_counts = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
                while steps < max_steps and not dones.all():
                    actions, hidden_states = self._get_student_actions_eval(
                        obs, prev_actions, hidden_states
                    )
                    obs, reward, out_of_reach, timed_out, info = eval_env.step(actions)
                    dones = out_of_reach | timed_out
                    reason_idx = classify_out_of_reach_reasons(
                        ov_env=eval_ov_env,
                        out_of_reach=out_of_reach,
                        reason_names=self.unsafe_reason_names,
                        reason_to_idx=self.unsafe_reason_to_idx,
                        device=self.device,
                    )
                    ever_unsafe_terminated = ever_unsafe_terminated | out_of_reach
                    classified_out_of_reach = reason_idx >= 0
                    new_reason_mask = (unsafe_reason_idx < 0) & out_of_reach & classified_out_of_reach
                    unsafe_reason_idx[new_reason_mask] = reason_idx[new_reason_mask]
                    prev_actions = actions.detach()
                    if self.is_rnn and dones.any():
                        self._zero_rnn_states(hidden_states, dones.nonzero(as_tuple=False))
                    reward_sum += reward
                    steps += 1
                    table_center_z = eval_ov_env.cfg.table_cfg.init_state.pos[2]
                    table_top_z = table_center_z + 0.5 * eval_ov_env.cfg.table_size_z
                    lift_height_thresh = table_top_z + getattr(eval_ov_env.cfg, "object_height_thresh", 0.0)
                    lift_success = eval_ov_env.object_pos[:, 2] > lift_height_thresh
                    if hasattr(eval_ov_env, "good_grasp_mask") and eval_ov_env.good_grasp_mask is not None:
                        contact_mask = eval_ov_env.good_grasp_mask.to(device=self.device, dtype=torch.bool)
                    elif (
                        hasattr(eval_ov_env, "object_contact_counts")
                        and eval_ov_env.object_contact_counts is not None
                    ):
                        contact_mask = eval_ov_env.object_contact_counts.to(device=self.device) > 0.0
                    else:
                        contact_mask = torch.ones_like(lift_success, dtype=torch.bool)
                    lift_success = lift_success & contact_mask
                    active_envs = ~dones
                    lift_hold_counts = torch.where(
                        active_envs & lift_success,
                        lift_hold_counts + 1,
                        torch.where(active_envs, torch.zeros_like(lift_hold_counts), lift_hold_counts),
                    )
                    lift_success = lift_hold_counts >= hold_steps
                    ever_lifted = ever_lifted | lift_success
                # If we hit the max step cap, treat remaining envs as done for reward aggregation.
                if steps >= max_steps:
                    dones = torch.ones_like(dones)
                success_rates.append(ever_lifted.float().mean().item())
                reward_means.append(reward_sum.mean().item())
                unsafe_rates.append(ever_unsafe_terminated.float().mean().item())
                reason_counts, _ = _reason_counts_checked(
                    reason_idx=unsafe_reason_idx,
                    unsafe_mask=ever_unsafe_terminated,
                    reason_names=self.unsafe_reason_names,
                    reason_to_idx=self.unsafe_reason_to_idx,
                    warn_label=None,
                )
                for name in self.unsafe_reason_names:
                    total_reason_counts[name] += int(reason_counts[name])
                for obj_idx, object_name in enumerate(eval_object_names):
                    obj_mask = eval_object_idx == obj_idx
                    if not obj_mask.any():
                        continue
                    per_object_lift_series[object_name].append(
                        ever_lifted[obj_mask].float().mean().item()
                    )
                    per_object_unsafe_rate_series[object_name].append(
                        ever_unsafe_terminated[obj_mask].float().mean().item()
                    )
                    obj_reason_counts, _ = _reason_counts_checked(
                        reason_idx=unsafe_reason_idx[obj_mask],
                        unsafe_mask=ever_unsafe_terminated[obj_mask],
                        reason_names=self.unsafe_reason_names,
                        reason_to_idx=self.unsafe_reason_to_idx,
                        warn_label=None,
                    )
                    for reason_name in self.unsafe_reason_names:
                        per_object_reason_counts_total[object_name][reason_name] += int(
                            obj_reason_counts[reason_name]
                        )
        if was_training:
            self.student_model.train()
        total_eval_episodes = max(1, int(num_episodes * num_envs))
        eval_reason_prop = {
            name: float(total_reason_counts.get(name, 0)) / float(total_eval_episodes)
            for name in self.unsafe_reason_names
        }
        eval_per_object_metrics = {}
        for object_name in eval_object_names:
            total_obj_episodes = max(1, int(num_episodes * eval_object_env_counts.get(object_name, 0)))
            obj_reason_prop = {
                name: float(per_object_reason_counts_total[object_name].get(name, 0))
                / float(total_obj_episodes)
                for name in self.unsafe_reason_names
            }
            eval_per_object_metrics[object_name] = {
                "lift_success": (
                    float(np.mean(per_object_lift_series[object_name]))
                    if len(per_object_lift_series[object_name]) > 0
                    else 0.0
                ),
                "unsafe_episode_rate": (
                    float(np.mean(per_object_unsafe_rate_series[object_name]))
                    if len(per_object_unsafe_rate_series[object_name]) > 0
                    else 0.0
                ),
                "unsafe_reason_prop": {
                    name: float(obj_reason_prop.get(name, 0.0))
                    for name in self.unsafe_reason_names
                },
            }
        return (
            float(np.mean(success_rates)),
            float(np.mean(reward_means)),
            float(np.mean(unsafe_rates)),
            eval_reason_prop,
            eval_per_object_metrics,
        )

    # --- Loss and Optimization Utilities ---
    def reduce_loss(self, loss_per_env):
        rnn_masks = None
        losses, _ = torch_ext.apply_masks([loss_per_env.unsqueeze(1)], rnn_masks)
        return losses[0]

    def loss(self, student_result, target_result, fn="l2", weights=None):
        if fn == "l2":
            loss = l2(student_result, target_result)
        else:
            loss = weighted_l2(student_result, target_result, weights)
        rnn_masks = None
        losses, sum_mask = torch_ext.apply_masks(
            [loss.unsqueeze(1)], rnn_masks
        )
        return losses[0]

    # --- Config and Checkpoint I/O ---
    def set_weights(self, ckpt, policy_type, model_override=None):
        """Set the weights of the model."""
        weights = torch_ext.load_checkpoint(ckpt)
        if policy_type == "student":
            weights["model"] = adjust_state_dict_keys(
                weights["model"],
                self.student_model.state_dict()
            )
            model = self.student_model
            # self.epoch_num = weights.get('epoch', 0)
            # self.optimizer.load_state_dict(weights['optimizer'])
            # self.frame = weights.get('frame', 0)
        else:
            model = model_override if model_override is not None else self.teacher_model
        model.load_state_dict(weights["model"])
        if self.normalize_input and 'running_mean_std' in weights:
            model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def save(self, filename):
        """Save the checkpoint to filename"""
        state = {
            "model": self.student_model.state_dict()
        }
        state['epoch'] = self.epoch_num
        state['optimizer'] = self.optimizer.state_dict()
        state['frame'] = self.frame
        torch_ext.save_checkpoint(filename, state)

    def load_networks(self, params):
        """Loads the network """
        builder = ModelBuilder()
        return builder.load(params)

    def load_param_dict(self, cfg_path) -> Dict:
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
