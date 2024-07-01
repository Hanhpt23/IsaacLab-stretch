# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python3 source/standalone/stretchRL/stretchconfig/stretch_env.py --num_envs 32
    

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.orbit.assets import Articulation
##
# Pre-defined configs
##
from omni.isaac.lab_assets import CARTPOLE_CFG, FRANKA_PANDA_CFG, RIDGEBACK_FRANKA_PANDA_CFG, STRETCH_CFG# isort:skip

def design_scene():
    """Designs the scene."""
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # add robots and return them

    return add_robots()

def add_robots() -> Articulation:
    """Adds robots to the scene."""
    robot_cfg = STRETCH_CFG
    # -- Spawn robot
    robot_cfg.spawn.func("/World/Robot_1", robot_cfg.spawn, translation=(0.0, -1.0, 0.0))
    robot_cfg.spawn.func("/World/Robot_2", robot_cfg.spawn, translation=(0.0, 1.0, 0.0))
    # -- Create interface
    robot = Articulation(cfg=robot_cfg.replace(prim_path="/World/Robot.*"))

    return robot

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    cartpole: ArticulationCfg = STRETCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
