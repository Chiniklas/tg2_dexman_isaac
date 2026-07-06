"""Gym registration for the SimToolReal TG2-InspireHand Isaac Lab task."""

import gymnasium as gym

from . import agents
from .simtoolreal_tg2_env import SimToolRealTg2Env
from .simtoolreal_tg2_env_cfg import SimToolRealTg2EnvCfg


gym.register(
    id="simtoolreal_tg2",
    entry_point="tg2_lab.tasks.simtoolreal_tg2.simtoolreal_tg2_env:SimToolRealTg2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SimToolRealTg2EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapo_cfg.yaml",
    },
)

gym.register(
    id="simtoolreal_tg2_pretrain_like",
    entry_point="tg2_lab.tasks.simtoolreal_tg2.simtoolreal_tg2_env:SimToolRealTg2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SimToolRealTg2EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapo_pretrain_like_cfg.yaml",
    },
)
