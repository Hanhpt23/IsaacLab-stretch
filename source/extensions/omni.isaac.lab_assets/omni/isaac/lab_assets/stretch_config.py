# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`STRETCH_CFG`: Stretch robot
* :obj:`STRETCH_PD_CFG`: Stretch robot with with stiffer PD control


"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
## 

STRETCH_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/hanhubuntu/IsaacLab/source/standalone/stretchRL/stretch_URDF_caster_revolute/stretch/stretch.usd", 
        activate_contact_sensors=False,
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #     disable_gravity=False,
            # max_depenetration_velocity=5.0,
        # ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            # solver_position_iteration_count=8, 
            # solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     contact_offset=0.005, 
        #     rest_offset=0.0),

    ),
    init_state=ArticulationCfg.InitialStateCfg(

		# Here is a list of joints and their recommended position limits:
		############################# JOINT LIMITS #############################
		# joint_lift:      lower_limit =  0.15,  upper_limit =  1.10  # in meters
		# wrist_extension: lower_limit =  0.00,  upper_limit =  0.50  # in meters
		# joint_wrist_yaw: lower_limit = -1.75,  upper_limit =  4.00  # in radians
		# joint_head_pan:  lower_limit = -2.80,  upper_limit =  2.90  # in radians
		# joint_head_tilt: lower_limit = -1.60,  upper_limit =  0.40  # in radians
		# joint_gripper_finger_left:  lower_limit = -0.35,  upper_limit =  0.165  # in radians
		#
		# INCLUDED JOINTS IN POSITION MODE
		# translate_mobile_base: No lower or upper limit. Defined by a step size in meters
		# rotate_mobile_base:    No lower or upper limit. Defined by a step size in radians
		########################################################################


        # Stretch joints
        # ['caster_joint', 'joint_left_wheel', 'joint_right_wheel', 'joint_lift', 'joint_head_pan', 
        # 'joint_head_tilt', 'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 'joint_wrist_yaw', 
        # 'joint_gripper_finger_left', 'joint_gripper_finger_right']


        joint_pos={
            # head
            "joint_head_pan": 1.0,
            "joint_head_tilt": 0.0,
            # lift
            "joint_lift": 0.5,
            # arm
            "joint_arm_l0": 0.0,
            "joint_arm_l1": 0.0,
            "joint_arm_l2": 0.0,
            "joint_arm_l3": 0.0,
            # wrist 
            "joint_wrist_yaw": 3.4,
            # gripper
            "joint_gripper_finger_left": -0.1,
            "joint_gripper_finger_right": 0.1,
            # base
            "joint_left_wheel": 0.0,
            "joint_right_wheel": 0.0,
            "caster_joint": 0
        },
        pos=(2.0, 0.0, 0.0)
    ),
    actuators={
        # Stretch head
        "stretch_head_pan": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_pan"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "stretch_head_tilt": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_tilt"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        # Stretch lift
        "stretch_lift": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift"],
            effort_limit=200.0,
            velocity_limit=0.1,
            stiffness=2e3,
            damping=1e2,
        ),
        # Stretch arm
        "stretch_extension_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_arm_l[0-3]"],
            effort_limit=87.0,
            velocity_limit=1,
            stiffness=80.0,
            damping=4.0,
        ),
        # Stretch wrist
        "stretch_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_wrist_yaw"],
            effort_limit=100.0,
            velocity_limit=100,
            stiffness=800,
            damping=4.0,
        ),
        # Stretch gripper
        "stretch_gripper_left": ImplicitActuatorCfg(
            joint_names_expr=["joint_gripper_finger_left"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "stretch_gripper_right": ImplicitActuatorCfg(
            joint_names_expr=["joint_gripper_finger_right"],
            effort_limit=200.0,
            velocity_limit=1.0,
            stiffness=2e3,
            damping=1e2,
        ),
        # Stretch Base
        "stretch_base_right": ImplicitActuatorCfg(
            joint_names_expr=["joint_right_wheel"],
            effort_limit=200.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1e2,
        ),
        "stretch_base_left": ImplicitActuatorCfg(
            joint_names_expr=["joint_left_wheel"],
            effort_limit=200.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1e2,
        ),
        "caster": ImplicitActuatorCfg(
            joint_names_expr=["caster_joint"],
            effort_limit=200.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Stretch robot."""



"""Configuration of Stretch robot with stiffer PD control.

The following control configuration is used:

* Base: velocity control
* Arm: position control with damping
* Lift: position control with damping
* Wrist, Gripper: position control with damping
"""
