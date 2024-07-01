# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/03_scene/create_scene.py --num_envs 32

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
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
##
# Pre-defined configs
##
from omni.isaac.lab_assets import CARTPOLE_CFG, FRANKA_PANDA_CFG, RIDGEBACK_FRANKA_PANDA_CFG, STRETCH_CFG# isort:skip


@configclass
class StretchSceneCfg(InteractiveSceneCfg):
    """Configuration for a stretch scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    stretch: ArticulationCfg = STRETCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities

    robot = scene["stretch"]
    print('Joint names: ', robot.joint_names)
    print('Body names: ', robot.body_names)
    print('Joint position limits: ', robot.data.soft_joint_pos_limits)
    print('Joint velocity limits: ', robot.data.soft_joint_vel_limits)

    '''
    Stretch joints
    ['caster_joint', 'joint_left_wheel', 'joint_right_wheel', 'joint_lift', 'joint_head_pan', 
     'joint_head_tilt', 'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 'joint_wrist_yaw', 
     'joint_gripper_finger_left', 'joint_gripper_finger_right']

    '''
    
    target_base_names = [
        # "caster_joint",
        "joint_left_wheel",
        "joint_right_wheel"
    ]

    target_base_index = [robot.data.joint_names.index(name) for name in target_base_names]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 1000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state) # set the root state

            #set all position to zeros and lift with a height of 0.2 m
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            
            joint_pos_0 = torch.zeros_like(joint_pos)
            joint_pos_0[:, 3] = 0.2
            joint_vel_0 = torch.zeros_like(joint_vel)
            
            robot.write_joint_state_to_sim(joint_pos_0, joint_vel_0)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

            # reset command
            # actions = torch.rand_like(joint_pos)
            actions = torch.zeros_like(joint_pos)
            # lift
            actions[:, 3] = 0.2
            # print(actions)


        # Apply specific actions for wheels between count 200 and 300
        if 100 <= count < 800:
            actions[:, 1:3] = 50.0
        else:
            actions[:, 1:3] = 0.0

        # Increase arms (joints) 
        if 0 <= count < 250:
            actions[:, 6:10] += 0.001
        else:
            actions[:, 6:10] -= 0.001

        print(actions.shape)


        # Apply action
        robot.set_joint_velocity_target(actions[:, 1:3], joint_ids=target_base_index)
        robot.set_joint_position_target(actions[:, 3:], joint_ids=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        print(robot.data.joint_vel)
        


        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = StretchSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print("Sence: ", scene)
    # scene.articulations["stretch"] = scene
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
