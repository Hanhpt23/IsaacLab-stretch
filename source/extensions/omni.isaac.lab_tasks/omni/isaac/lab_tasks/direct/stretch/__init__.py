# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Stretch-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .stretch_env import StretchReachEnvCfg, StretchReachEnv 
from .stretch_customed_env import StretchReachCustomEnvCfg, StretchReachCustomEnv 

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Stretch-Cabinet-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.stretch:StretchReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StretchReachEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.StretchCabinetPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Stretch-Cabinet-Direct-v1",
    entry_point="omni.isaac.lab_tasks.direct.stretch:StretchReachCustomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StretchReachCustomEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.StretchCabinetPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
