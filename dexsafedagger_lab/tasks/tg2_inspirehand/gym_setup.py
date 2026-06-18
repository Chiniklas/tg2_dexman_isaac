"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages
import gymnasium as gym

from . import agents
from .dexsafedagger_tg2_inspirehand_env import DexSafeDaggerTG2InspirehandEnv
from .dexsafedagger_tg2_inspirehand_env_cfg import DexSafeDaggerTG2InspirehandEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="dexsafedagger_tg2_inspirehand",
    entry_point="dexsafedagger_lab.tasks.tg2_inspirehand.dexsafedagger_tg2_inspirehand_env:DexSafeDaggerTG2InspirehandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DexSafeDaggerTG2InspirehandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# The blacklist is used to prevent importing configs from sub-packages
#_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
#import_packages(__name__, _BLACKLIST_PKGS)
