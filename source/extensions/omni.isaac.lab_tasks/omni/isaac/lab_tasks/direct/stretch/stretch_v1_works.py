# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab_assets import STRETCH_CFG
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg



@configclass
class StretchReachCustomEnvCfg(DirectRLEnvCfg):

    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    num_actions = 13 # 13 joints
    num_observations = 31
    num_states = 0

    action_scale = 0.5
    dof_velocity_scale = 1.0

    # simulation
    # sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=2.5,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = STRETCH_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0.0, 0.40),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )
    

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="multiply",
        #     restitution_combine_mode="multiply",
        #     static_friction=0.5,
        #     dynamic_friction=0.5,
        #     restitution=0.0,
        # ),
    )

    # reward scales
    dist_reward_scale = 2.0
    rot_reward_scale = 0.5
    around_handle_reward_scale = 0.0
    open_reward_scale = 7.5
    action_penalty_scale = 0.01
    finger_dist_reward_scale = 0.0
    finger_close_reward_scale = 10.0


class StretchReachCustomEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: StretchReachCustomEnvCfg
    cfg1: DirectRLEnvCfg

    def __init__(self, cfg: StretchReachCustomEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # print(self._robot.joint_names)
        # ['caster_joint', 'joint_left_wheel', 'joint_right_wheel', 'joint_lift', 'joint_head_pan', 'joint_head_tilt', 
        #  'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 'joint_wrist_yaw', 
        #  'joint_gripper_finger_left', 'joint_gripper_finger_right']

        # print(self._robot.body_names)
        # ['base_link', 'caster_link', 'link_aruco_left_base', 'link_aruco_right_base', 'laser', 'link_left_wheel', 
        #  'link_mast', 'link_right_wheel', 'link_head', 'link_lift', 'respeaker_base', 'link_head_pan', 'link_arm_l4', 
        #  'link_aruco_shoulder', 'link_head_tilt', 'link_arm_l3', 'camera_bottom_screw_frame', 'link_arm_l2', 'camera_link', 
        #  'link_arm_l1', 'camera_accel_frame', 'camera_color_frame', 'camera_depth_frame', 'camera_gyro_frame', 'camera_infra1_frame', 
        #  'camera_infra2_frame', 'link_arm_l0', 'camera_accel_optical_frame', 'camera_color_optical_frame', 
        #  'camera_depth_optical_frame', 'camera_gyro_optical_frame', 'camera_infra1_optical_frame', 
        #  'camera_infra2_optical_frame', 'link_aruco_inner_wrist', 'link_aruco_top_wrist', 'link_wrist_yaw', 
        #  'link_gripper', 'link_grasp_center', 'link_gripper_finger_left', 'link_gripper_finger_right', 
        #  'link_gripper_fingertip_left', 'link_gripper_fingertip_right']



        self.target_base_names = [
            # "caster_joint",
            "joint_left_wheel",
            "joint_right_wheel"
        ]
        self.target_base_index = [self._robot.data.joint_names.index(name) for name in self.target_base_names]
        
        # create auxiliary variables for computing applied action, observations and rewards
        # position
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_lower_limits[0] = -99999999
        self.robot_dof_upper_limits[0] =  99999999
        print('position limitation: ',self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # tensor([-0.6000,    -inf,    -inf,  0.0000, -3.9000, -1.5300,  0.0000,  0.0000, 0.0000,  0.0000, -1.7500, -0.6000, -0.6000], device='cuda:0') 
        # tensor([0.6000,    inf,    inf, 1.1000, 1.5000, 0.7900, 0.1300, 0.1300, 0.1300, 0.1300, 4.0000, 0.6000, 0.6000], device='cuda:0')


        # # velocity
        # self.robot_dof_upper_limits_vel = self._robot.data.soft_joint_vel_limits.to(device=self.device)
        # self.robot_dof_lower_limits_vel = torch.zeros_like(self.robot_dof_upper_limits_vel)
        # print('velocity limitation: ', self._robot.data.soft_joint_vel_limits, self._robot.data.joint_vel)
        # # joit velocity limits are zeros, we need to take into account

        
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("joint_gripper_finger_left")[0]] = 0.13
        self.robot_dof_speed_scales[self._robot.find_joints("joint_gripper_finger_right")[0]] = 0.13
        print('self.robot_dof_speed_scales', self.robot_dof_speed_scales)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/link_arm_l3")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/link_gripper_fingertip_left")), # link_gripper_finger_left
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/link_gripper_fingertip_right")), # link_gripper_finger_right
            self.device,
        )

        left_wheel = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/link_left_wheel")),
            self.device,
        )
        right_wheel = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/link_right_wheel")),
            self.device,
        )
        # Determine hand pose root and position
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        # determin local grasp root and position
        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        # robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # position and root of cabinet grasper
        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, -1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("link_arm_l3")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("link_gripper_fingertip_left")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("link_gripper_fingertip_right")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone() #* self.cfg.action_scale

        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        # print('self.robot_dof_targets: ', self.robot_dof_targets)
        # print('Actions: ', self.actions)


    def _apply_action(self):
        # Apply action
        # self._robot.set_joint_velocity_target(self.actions[:, 1:3], joint_ids= self.target_base_index)
        # self._robot.set_joint_position_target(self.actions[:, 3:], joint_ids=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        self._robot.set_joint_velocity_target(self.robot_dof_targets[:, 1:3], joint_ids= self.target_base_index)
        self._robot.set_joint_position_target(self.robot_dof_targets[:, 3:], joint_ids=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._robot.data.joint_pos[:, 3] > 0.19
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        

        # Creating a tensor with all True values on GPU
        # terminated = torch.zeros_like( self._robot.data.joint_pos[:, 3], dtype=torch.bool, device=self.device)
        # Creating a tensor with all False values on GPU
        # truncated = torch.zeros_like( self._robot.data.joint_pos, dtype=torch.bool, device=self.device)

        # print('Termination and truncation: ',terminated, truncated)
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        
        pass
        # # Refresh the intermediate values after the physics steps
        # self._compute_intermediate_values()
        # robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        # robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        # return self._compute_rewards(
        #     self.actions,
        #     self._cabinet.data.joint_pos,
        #     self.robot_grasp_pos,
        #     self.drawer_grasp_pos,
        #     self.robot_grasp_rot,
        #     self.drawer_grasp_rot,
        #     robot_left_finger_pos,
        #     robot_right_finger_pos,
        #     self.gripper_forward_axis,
        #     self.drawer_inward_axis,
        #     self.gripper_up_axis,
        #     self.drawer_up_axis,
        #     self.num_envs,
        #     self.cfg.dist_reward_scale,
        #     self.cfg.rot_reward_scale,
        #     self.cfg.around_handle_reward_scale,
        #     self.cfg.open_reward_scale,
        #     self.cfg.finger_dist_reward_scale,
        #     self.cfg.action_penalty_scale,
        #     self._robot.data.joint_pos,
        #     self.cfg.finger_close_reward_scale,
        # )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        pass
        # robot state
        # joint_pos = self._robot.data.default_joint_pos[env_ids] 
        # print('Reset joint position: ', joint_pos)
        # # joint_pos[:, 3] = 0.5
        # joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # joint_vel = torch.zeros_like(joint_pos)
        # self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # # cabinet state
        # zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        # self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # # Need to refresh the intermediate values so that _get_observations() can use the latest values
        # self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # dof_pos_scaled = (
        #     2.0
        #     * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
        #     / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
        #     - 1.0
        # )
        to_target = self.drawer_grasp_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                self._robot.data.joint_pos,  # dof_pos_scaled, # 13
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale, # 13
                to_target, # 3
                self._cabinet.data.joint_pos[:, 3].unsqueeze(-1), # 1
                self._cabinet.data.joint_vel[:, 3].unsqueeze(-1), # 1
            ),
            dim=-1,
        )
        print('Observation: ', obs.shape, obs)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        around_handle_reward_scale,
        open_reward_scale,
        finger_dist_reward_scale,
        action_penalty_scale,
        joint_positions,
        finger_close_reward_scale,
    ):
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # bonus if left finger is above the drawer handle and right below
        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(
            franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
            torch.where(
                franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2], around_handle_reward + 0.5, around_handle_reward
            ),
            around_handle_reward,
        )
        # reward for distance of each finger from the drawer
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        finger_dist_reward = torch.where(
            franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
            torch.where(
                franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
                finger_dist_reward,
            ),
            finger_dist_reward,
        )

        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(
            d <= 0.03, (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]), finger_close_reward
        )

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + around_handle_reward_scale * around_handle_reward
            + open_reward_scale * open_reward
            + finger_dist_reward_scale * finger_dist_reward
            - action_penalty_scale * action_penalty
            + finger_close_reward * finger_close_reward_scale
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "around_handle_reward": (around_handle_reward_scale * around_handle_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "finger_dist_reward": (finger_dist_reward_scale * finger_dist_reward).mean(),
            "action_penalty": (action_penalty_scale * action_penalty).mean(),
            "finger_close_reward": (finger_close_reward * finger_close_reward_scale).mean(),
        }

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
