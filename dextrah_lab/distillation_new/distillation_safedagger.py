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

# Imitation loss options (imitation_loss_type):
# - "kl": KL(N_teacher || N_student) over action distributions.
# - "nll": -log pi_student(a_teacher_sample) using teacher sampled actions.
# - "mse": MSE between teacher sampled actions and student sampled actions.
# - "l2": legacy weighted L2 on mus (by 1/sigma^2) + L2 on sigmas.
#
# Optional debug-friendly config:
# - imitation_target: "action_distribution" | "sampled_action"
# - loss_type: "kl" | "nll" | "l2" | "mse"
# If both are provided, they override imitation_loss_type.
#
# Optional OOD config (simple diagonal Gaussian density model on a buffer):
# - ood.enabled: bool
# - ood.obs_key: observation key to use for OOD features (default: student obs_type)
# - ood.buffer_size: max number of feature vectors to store
# - ood.min_samples: warm-start threshold; below this, always mark unsafe
# - ood.update_interval: steps between refitting mean/var and threshold
# - ood.threshold_quantile: quantile for OOD threshold on buffer scores
# - ood.diag_eps: variance floor for stability

def l2(model, target):
    """Computes the L2 norm between model and target.
    """

    return torch.norm(model - target, p=2, dim=-1)

def weighted_l2(model, target, weights):
    return torch.sum((model - target) * (weights * (model - target)), dim=-1) ** 0.5


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

def gaussian_kl(mu_student, sigma_student, mu_teacher, sigma_teacher, eps=1e-6):
    """KL(N_teacher || N_student) per sample, summed over action dims."""
    sigma_student = torch.clamp(sigma_student, min=eps)
    sigma_teacher = torch.clamp(sigma_teacher, min=eps)
    var_student = sigma_student ** 2
    var_teacher = sigma_teacher ** 2
    mu_term = (mu_teacher - mu_student) ** 2 / (2.0 * var_student)
    sigma_term = torch.log(sigma_student / sigma_teacher) + var_teacher / (2.0 * var_student) - 0.5
    kl = mu_term + sigma_term
    return kl.sum(-1), mu_term.sum(-1), sigma_term.sum(-1)

def gaussian_nll(mu_student, sigma_student, actions, eps=1e-6):
    """Negative log-likelihood of actions under N(mu_student, sigma_student), summed over dims."""
    sigma_student = torch.clamp(sigma_student, min=eps)
    var_student = sigma_student ** 2
    nll = 0.5 * (
        ((actions - mu_student) ** 2) / var_student
        + 2.0 * torch.log(sigma_student)
        + math.log(2.0 * math.pi)
    )
    return nll.sum(-1)

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
        if self.rank == 0:
            print(
                f"SafeDagger eval_every={self.eval_every}, eval_num_episodes={self.eval_num_episodes}",
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
        self.imitation_loss_type = self._resolve_imitation_loss_type()
        if self.imitation_loss_type not in {"kl", "nll", "l2", "mse"}:
            raise ValueError(f"Unsupported imitation_loss_type: {self.imitation_loss_type}")
        if self.rank == 0:
            print(f"Using imitation loss: {self.imitation_loss_type}")
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-4, eps=1e-8) # default lr = 1e-4
        self.num_iters = 100_000

        # load weights for student and teacher
        if self.config["student"]["ckpt"] is not None:
            self.set_weights(self.config["student"]["ckpt"], "student")
        if not self.multi_teacher and self.config["teacher"]["ckpt"] is not None:
            self.set_weights(self.config["teacher"]["ckpt"], "teacher")
        # get the observation type of the student and teacher
        self.student_obs_type = self.config["student"]["obs_type"]
        self.teacher_obs_type = self.config["teacher"]["obs_type"]
        self.ood_classifier = self._build_ood_classifier(self.config.get("ood", {}))
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
        self.game_rewards = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)

        if self.rank == 0:
            self.writer = SummaryWriter(summaries_dir)
            self.use_wandb = False
            parent_path = str(pathlib.Path(__file__).parent.resolve())
            summaries_dir = os.path.join(parent_path, summaries_dir)
            self.nn_dir = os.path.join(parent_path, nn_dir)
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

    def distill(self):
        self.student_model.train()
        if self.multi_teacher:
            for model in self.teacher_models:
                model.eval()
        else:
            self.teacher_model.eval()
        # torch.set_float32_matmul_precision('high')

        obs = self.env.reset()[0]

        log_counter = 0
        total_loss = 0.

        self.optimizer.zero_grad()

        num_iters = self.num_iters
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
                if actions_student.get("embeds") is not None:
                    embeds = actions_student["embeds"].detach()
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
                mu_loss = None
                sigma_loss = None
                if self.imitation_loss_type == "kl":
                    kl_per_env, kl_mu_per_env, kl_sigma_per_env = gaussian_kl(
                        actions_student["mus"],
                        actions_student["sigmas"],
                        actions_teacher["mus"],
                        actions_teacher["sigmas"],
                    )
                    imitation_loss = self.reduce_loss(kl_per_env)
                    mu_loss = self.reduce_loss(kl_mu_per_env)
                    sigma_loss = self.reduce_loss(kl_sigma_per_env)
                elif self.imitation_loss_type == "nll":
                    nll_per_env = gaussian_nll(
                        actions_student["mus"],
                        actions_student["sigmas"],
                        actions_teacher["actions"],
                    )
                    imitation_loss = self.reduce_loss(nll_per_env)
                elif self.imitation_loss_type == "mse":
                    mse_per_env = torch.mean(
                        (actions_student["actions"] - actions_teacher["actions"]) ** 2, dim=-1
                    )
                    imitation_loss = self.reduce_loss(mse_per_env)
                    mu_loss = imitation_loss
                else:
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
                self.unsafe = self.check_unsafe(l2_loss_per_env=l2_loss_per_env, obs=obs)
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

            obs, rew, out_of_reach, timed_out, info = self.env.step(
                stepping_actions.detach()
            )

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
            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            self.actions_teacher[done_indices] *= 0.
            if len(done_indices) > 0:
                self.unsafe[done_indices] = False

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
                eval_lift, eval_reward, eval_unsafe = self.evaluate_student(self.eval_num_episodes)
                if self.rank == 0 and eval_lift is not None:
                    self.writer.add_scalar("eval/lift_success", eval_lift, self.frame)
                    self.writer.add_scalar("eval/avg_reward", eval_reward, self.frame)
                    self.writer.add_scalar("eval/in_unsafe_zone", eval_unsafe, self.frame)
                    self.writer.flush()
                    print(
                        f"Eval lift_success: {eval_lift:.3f} | avg_reward: {eval_reward:.3f} | "
                        f"in_unsafe_zone: {eval_unsafe:.3f}",
                        flush=True,
                    )
                if self.eval_env is None:
                    obs = self.env.reset()[0]
                    self.init_tensors()

        if self.rank == 0 and self.use_wandb:
            wandb.finish()

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
                if l2_loss_mean is not None:
                    self.writer.add_scalar(
                        "l2_loss_mean", l2_loss_mean.detach().cpu().numpy(), self.frame
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
            if self.game_rewards.current_size > 0:
                print("\tMean Rewards: ", mean_rewards)
                print("\tMean Length: ", mean_lengths)
                print("\tin_success_region: ", perf)

    def log_img(self, pred_images, gt_images):
        combined_images = torch.cat((pred_images, gt_images), dim=0)
        image_grid = vutils.make_grid(combined_images, nrow=pred_images.shape[0], normalize=True, scale_each=True)
        self.writer.add_image('Predictions_vs_Ground_Truth', image_grid, global_step=self.frame)
        if self.use_wandb:
            images = wandb.Image(image_grid, caption="Top: Network Pred, Bottom: GT")
            wandb.log({"predictions vs ground truth": images})

    def reduce_loss(self, loss_per_env):
        rnn_masks = None
        losses, _ = torch_ext.apply_masks([loss_per_env.unsqueeze(1)], rnn_masks)
        return losses[0]

    def check_unsafe(self, l2_loss_per_env=None, obs=None, out_of_reach=None, timed_out=None, info=None):
        """Unsafe logic controlled by unsafe_mode: none | l2 | ood."""
        if self.unsafe_mode == "none":
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.unsafe_mode == "ood":
            if self.ood_classifier is not None and self.ood_classifier.enabled:
                unsafe = self.ood_classifier.check_ood(obs, self.device)
                if unsafe is not None:
                    return unsafe
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if l2_loss_per_env is not None:
            return l2_loss_per_env > self.unsafe_l2_threshold
        if isinstance(info, dict):
            if "unsafe" in info:
                return torch.as_tensor(info["unsafe"], device=self.device, dtype=torch.bool)
            if "unsafe_flag" in info:
                return torch.as_tensor(info["unsafe_flag"], device=self.device, dtype=torch.bool)
            if "in_unsafe_region" in info:
                return torch.as_tensor(info["in_unsafe_region"], device=self.device, dtype=torch.bool)
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
            # real_world_idx = 1
            # real_world_names = ["obs.pth", "obs2.pth"]
            # gt_pos = [[-0.7, 0.08, 0.295], [-0.65, 0.25, 0.3]]
            # real_obs = torch.load(real_world_names[real_world_idx])
            # gt_pos = torch.tensor(gt_pos[real_world_idx]).reshape(1, 3).to(self.device)
            # arm_positions = [
            #     -0.7875749180783324, -0.4724239581655329, 0.6733201341853008,
            #     1.211750626464511, -1.7481902752912126, 0.9306101177413156, 0.6660046636382199
            # ]
            # hand_positions = [
            #     -0.018107680695800783, 0.22323929877421064, 0.7489833809370932, 0.9548251041408286,
            #     -0.013491997381184897, 0.34635377487752284, 0.8157332627176921, 0.8537238869226075,
            #     -0.0008876314066569011, 0.4619233840242513, 0.8937560633628338, 0.8243432873622641,
            #     1.175845324398397, 0.3547862732407634, 0.3690771388879395, 0.286438654928182
            # ]
            # robot_q = torch.tensor(arm_positions + hand_positions).to(self.device)
            # obs[self.student_obs_type][:, :len(robot_q)] = robot_q
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
            return None, None, None
        eval_env = self.eval_env if self.eval_env is not None else self.env
        eval_ov_env = eval_env.env
        if eval_ov_env.num_envs != self.num_envs:
            if self.rank == 0:
                print(
                    "Skipping eval: eval_env num_envs must match training num_envs "
                    f"({eval_ov_env.num_envs} vs {self.num_envs})."
                )
            return None, None, None
        num_envs = eval_ov_env.num_envs
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
        with torch.no_grad():
            for _ in range(num_episodes):
                obs = eval_env.reset()[0]
                dones = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
                ever_lifted = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
                ever_unsafe_terminated = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
                if self.is_rnn:
                    hidden_states = [s.to(self.device) for s in self.student_model.get_default_rnn_state()]
                else:
                    hidden_states = None
                prev_actions.zero_()
                steps = 0
                reward_sum = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)
                while steps < max_steps and not dones.all():
                    actions, hidden_states = self._get_student_actions_eval(
                        obs, prev_actions, hidden_states
                    )
                    obs, reward, out_of_reach, timed_out, info = eval_env.step(actions)
                    dones = out_of_reach | timed_out
                    ever_unsafe_terminated = ever_unsafe_terminated | out_of_reach
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
                    try:
                        lift_weight = eval_ov_env.dextrah_adr.get_custom_param_value(
                            "reward_weights", "lift_weight"
                        )
                    except Exception:
                        lift_weight = 1.0
                    if lift_weight == 0.0:
                        lift_success = torch.zeros_like(lift_success)
                    ever_lifted = ever_lifted | lift_success
                # If we hit the max step cap, treat remaining envs as done for reward aggregation.
                if steps >= max_steps:
                    dones = torch.ones_like(dones)
                success_rates.append(ever_lifted.float().mean().item())
                reward_means.append(reward_sum.mean().item())
                unsafe_rates.append(ever_unsafe_terminated.float().mean().item())
        if was_training:
            self.student_model.train()
        return (
            float(np.mean(success_rates)),
            float(np.mean(reward_means)),
            float(np.mean(unsafe_rates)),
        )

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

    def _resolve_imitation_loss_type(self):
        target = self.config.get("imitation_target", None)
        loss_type = self.config.get("loss_type", None)
        if target is None and loss_type is None:
            return self.config.get("imitation_loss_type", "kl")
        if target is None or loss_type is None:
            raise ValueError(
                "Both imitation_target and loss_type must be set together "
                "(e.g., --imitation_target action_distribution --loss_type l2)."
            )
        target = str(target).lower()
        loss_type = str(loss_type).lower()
        if target not in {"action_distribution", "sampled_action"}:
            raise ValueError(f"Unsupported imitation_target: {target}")
        if loss_type not in {"kl", "nll", "l2", "mse"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        if target == "action_distribution" and loss_type not in {"kl", "nll", "l2"}:
            raise ValueError(
                "loss_type 'mse' requires imitation_target 'sampled_action'."
            )
        if target == "sampled_action" and loss_type not in {"mse"}:
            raise ValueError(
                "sampled_action only supports loss_type 'mse' in this implementation."
            )
        if self.rank == 0:
            print(f"Using imitation_target={target}, loss_type={loss_type}")
        return loss_type

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
