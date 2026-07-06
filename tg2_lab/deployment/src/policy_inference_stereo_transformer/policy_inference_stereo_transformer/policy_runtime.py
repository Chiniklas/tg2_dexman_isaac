#!/usr/bin/env python3
"""Runtime wrapper for stereo transformer RL-Games policies."""

from __future__ import annotations

import yaml

import torch
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder


def load_param_dict(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def adjust_state_dict_keys(checkpoint_state_dict: dict, model_state_dict: dict) -> dict:
    """Best-effort key remapping for checkpoints saved with/without _orig_mod."""
    adjusted_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            adjusted_state_dict[key] = value
            continue

        parts = key.split(".")
        parts.insert(2, "_orig_mod")
        with_orig = ".".join(parts)
        if with_orig in model_state_dict:
            adjusted_state_dict[with_orig] = value
            continue

        no_orig = key.replace("_orig_mod.", "")
        if no_orig in model_state_dict:
            adjusted_state_dict[no_orig] = value
            continue

        adjusted_state_dict[key] = value
    return adjusted_state_dict


def register_stereo_transformer_builder() -> None:
    """Register network builder used by rl_games_ppo_stereo_transformer.yaml."""
    try:
        from tg2_lab.distillation.a2c_stereo_transformer import (
            A2CBuilder as A2CStereoTransformerBuilder,
        )
    except ImportError:
        from tg2_lab.distillation.a2c_stereo_transformer import (  # type: ignore
            A2CBuilder as A2CStereoTransformerBuilder,
        )

    model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)


class StereoTransformerPolicy:
    """Minimal deployment-safe wrapper around an rl_games stereo transformer policy."""

    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str | None,
        img_shape: tuple[int, int, int],
        num_proprio_obs: int,
        num_actions: int,
        device: str = "cuda",
        num_envs: int = 2,
    ) -> None:
        register_stereo_transformer_builder()

        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.num_proprio_obs = int(num_proprio_obs)
        self.num_actions = int(num_actions)
        self.img_shape = img_shape
        self.num_envs = int(num_envs)

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        network_params = load_param_dict(cfg_path)["params"]
        model_config = {
            "actions_num": self.num_actions,
            "input_shape": (self.num_proprio_obs,),
            "num_seqs": self.num_envs,
            "value_size": 1,
            "normalize_value": network_params["config"].get("normalize_value", True),
            "normalize_input": network_params["config"].get("normalize_input", True),
            "num_envs": self.num_envs,
        }

        builder = ModelBuilder()
        network = builder.load(network_params)
        self.model = network.build(model_config).to(self.device)
        self.model.eval()

        if ckpt_path:
            weights = torch_ext.load_checkpoint(ckpt_path)
            remapped = adjust_state_dict_keys(weights["model"], self.model.state_dict())
            self.model.load_state_dict(remapped)
            if model_config["normalize_input"] and "running_mean_std" in weights:
                self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

        self.hidden_states = None
        if self.model.is_rnn():
            self.hidden_states = [state.to(self.device) for state in self.model.get_default_rnn_state()]

        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self.device
        )

    def _validate_obs(
        self,
        proprio: torch.Tensor,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
    ) -> None:
        if tuple(proprio.shape) != (1, self.num_proprio_obs):
            raise ValueError(
                f"Expected proprio shape (1, {self.num_proprio_obs}), got {tuple(proprio.shape)}"
            )
        expected = (1, *self.img_shape)
        if tuple(left_img.shape) != expected:
            raise ValueError(f"Expected left image shape {expected}, got {tuple(left_img.shape)}")
        if tuple(right_img.shape) != expected:
            raise ValueError(f"Expected right image shape {expected}, got {tuple(right_img.shape)}")

    @torch.no_grad()
    def step(
        self,
        proprio: torch.Tensor,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        deterministic: bool = True,
    ) -> dict:
        self._validate_obs(proprio, left_img, right_img)

        obs = {
            "is_train": False,
            "obs": proprio.repeat(self.num_envs, 1),
            "img_left": left_img.repeat(self.num_envs, 1, 1, 1),
            "img_right": right_img.repeat(self.num_envs, 1, 1, 1),
            "prev_actions": self.prev_actions,
            "finetune_backbone": False,
        }

        if self.model.is_rnn():
            obs["rnn_states"] = self.hidden_states
            obs["seq_length"] = 1
            obs["rnn_masks"] = None

        res_dict = self.model(obs)
        mus = res_dict["mus"][0:1]
        sigmas = res_dict.get("sigmas", torch.zeros_like(mus))[0:1]

        result = {
            "mus": mus,
            "sigmas": sigmas,
            "selected_action": mus,
        }

        rnn_states = res_dict.get("rnn_states")
        if self.model.is_rnn() and rnn_states is not None:
            if isinstance(rnn_states, (list, tuple)) and len(rnn_states) > 0:
                candidate = rnn_states[0]
                if isinstance(candidate, (list, tuple)):
                    self.hidden_states = [state.detach() for state in candidate]

            if (
                isinstance(rnn_states, (list, tuple))
                and len(rnn_states) > 1
                and isinstance(rnn_states[1], dict)
                and "object_pos" in rnn_states[1]
            ):
                result["obj_pos"] = rnn_states[1]["object_pos"][0:1]

        if not deterministic:
            sigma = sigmas.abs().clamp(min=1e-6)
            result["selected_action"] = torch.distributions.Normal(mus, sigma).sample()

        return result

    def reset(self) -> None:
        if self.model.is_rnn() and self.hidden_states is not None:
            for state in self.hidden_states:
                state.zero_()
