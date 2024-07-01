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
    spawn=sim_utils.UsdFileCfg(
        usd_path="StretchGripper_USD/stretchgripper.usd", 
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, 
            rest_offset=0.0),

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

        joint_pos={
            "joint_lift": 0.2,
            "wrist_extension": 0,
            "joint_wrist_yaw": 3.4,
            "joint_head_pan": 1,
            "joint_head_tilt": 0.0,
            "joint_gripper_finger_left": 0.04,
        },
    ),
    actuators={
        "stretch_extension_arm": ImplicitActuatorCfg(
            joint_names_expr=["wrist_extension"],
            effort_limit=87.0,
            velocity_limit=0.5,
            stiffness=80.0,
            damping=4.0,
        ),
        "stretch_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_wrist_yaw"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "stretch_head": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "stretch_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint_gripper_finger_.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "stretch_lift": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        # Stretch Base
        "stretch_base_right": ImplicitActuatorCfg(
            joint_names_expr=["joint_right_wheel"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "stretch_base_left": ImplicitActuatorCfg(
            joint_names_expr=["joint_left_wheel"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Stretch robot."""


STRETCH_PD_CFG = STRETCH_CFG.copy()
STRETCH_PD_CFG.spawn.rigid_props.disable_gravity = True
STRETCH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
STRETCH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
STRETCH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
STRETCH_PD_CFG.actuators["panda_forearm"].damping = 80.0


"""Configuration of Stretch robot with stiffer PD control.

# This configuration is useful for task-space control using differential IK.
# """
