import numpy as np


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 left_shoulder_rgb: np.ndarray,
                 left_shoulder_depth: np.ndarray,
                 left_shoulder_mask: np.ndarray,
                 left_shoulder_point_cloud: np.ndarray,
                 right_shoulder_rgb: np.ndarray,
                 right_shoulder_depth: np.ndarray,
                 right_shoulder_mask: np.ndarray,
                 right_shoulder_point_cloud: np.ndarray,
                 overhead_rgb: np.ndarray,
                 overhead_depth: np.ndarray,
                 overhead_mask: np.ndarray,
                 overhead_point_cloud: np.ndarray,
                 overhead_rgb_0: np.ndarray,
                 overhead_depth_0: np.ndarray,
                 overhead_mask_0: np.ndarray,
                 overhead_point_cloud_0: np.ndarray,
                 wrist_rgb: np.ndarray,
                 wrist_depth: np.ndarray,
                 wrist_mask: np.ndarray,
                 wrist_point_cloud: np.ndarray,
                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                 front_mask: np.ndarray,
                 front_point_cloud: np.ndarray,
                 front_rgb_0: np.ndarray,
                 front_depth_0: np.ndarray,
                 front_mask_0: np.ndarray,
                 front_point_cloud_0: np.ndarray,
                 joint_velocities: np.ndarray,
                 joint_positions: np.ndarray,
                 joint_forces: np.ndarray,
                 gripper_open: float,
                 gripper_pose: np.ndarray,
                 gripper_matrix: np.ndarray,
                 gripper_joint_positions: np.ndarray,
                 gripper_touch_forces: np.ndarray,
                 joint_velocities_2: np.ndarray,
                 joint_positions_2: np.ndarray,
                 joint_forces_2: np.ndarray,
                 gripper_open_2: float,
                 gripper_pose_2: np.ndarray,
                 gripper_matrix_2: np.ndarray,
                 gripper_joint_positions_2: np.ndarray,
                 gripper_touch_forces_2: np.ndarray,
                 task_low_dim_state: np.ndarray,
                 misc: dict):
        self.left_shoulder_rgb = left_shoulder_rgb
        self.left_shoulder_depth = left_shoulder_depth
        self.left_shoulder_mask = left_shoulder_mask
        self.left_shoulder_point_cloud = left_shoulder_point_cloud
        self.right_shoulder_rgb = right_shoulder_rgb
        self.right_shoulder_depth = right_shoulder_depth
        self.right_shoulder_mask = right_shoulder_mask
        self.right_shoulder_point_cloud = right_shoulder_point_cloud
        self.overhead_rgb = overhead_rgb
        self.overhead_depth = overhead_depth
        self.overhead_mask = overhead_mask
        self.overhead_point_cloud = overhead_point_cloud
        self.overhead_rgb_0 = overhead_rgb_0
        self.overhead_depth_0 = overhead_depth_0
        self.overhead_mask_0 = overhead_mask_0
        self.overhead_point_cloud_0 = overhead_point_cloud_0
        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        self.wrist_mask = wrist_mask
        self.wrist_point_cloud = wrist_point_cloud
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.front_rgb_0 = front_rgb_0
        self.front_depth_0 = front_depth_0
        self.front_mask_0 = front_mask_0
        self.front_point_cloud_0 = front_point_cloud_0
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = gripper_matrix
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = gripper_touch_forces
        self.joint_velocities_2 = joint_velocities_2
        self.joint_positions_2 = joint_positions_2
        self.joint_forces_2 = joint_forces_2
        self.gripper_open_2 = gripper_open_2
        self.gripper_pose_2 = gripper_pose_2
        self.gripper_matrix_2 = gripper_matrix_2
        self.gripper_joint_positions_2 = gripper_joint_positions_2
        self.gripper_touch_forces_2 = gripper_touch_forces_2
        self.task_low_dim_state = task_low_dim_state
        self.misc = misc

    # def get_low_dim_data(self) -> np.ndarray:
    #     """Gets a 1D array of all the low-dimensional obseervations.

    #     :return: 1D array of observations.
    #     """
    #     low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
    #     for data in [self.joint_velocities, self.joint_positions,
    #                  self.joint_forces,
    #                  self.gripper_pose, self.gripper_joint_positions,
    #                  self.gripper_touch_forces, self.task_low_dim_state]:
    #         if data is not None:
    #             low_dim_data.append(data)
    #     return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
    

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.gripper_pose, self.gripper_joint_positions,
                     self.gripper_touch_forces, 
                     self.joint_velocities_2, self.joint_positions_2,
                     self.joint_forces_2,
                     self.gripper_pose_2, self.gripper_joint_positions_2,
                     self.gripper_touch_forces_2, 
                     self.task_low_dim_state, 
                     ]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
