"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages
import gymnasium as gym

from . import agents
from .dextrah_kuka_inspirehand_env import DextrahKukaInspirehandEnv
from .dextrah_kuka_inspirehand_env_cfg import DextrahKukaInspirehandEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Dextrah-Kuka-Inspirehand",
    entry_point="dextrah_lab.tasks.dextrah_kuka_inspirehand.dextrah_kuka_inspirehand_env:DextrahKukaInspirehandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DextrahKukaInspirehandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# The blacklist is used to prevent importing configs from sub-packages
#_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
#import_packages(__name__, _BLACKLIST_PKGS)
