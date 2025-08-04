from typing import List, Callable

import numpy as np
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.const import ObjectType
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from absl import flags

from rlbench.backend.exceptions import (
    WaypointError, BoundaryError, NoWaypointsError, DemoError)
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.noise_model import NoiseModel
from rlbench.observation_config import ObservationConfig, CameraConfig

import threading
import imageio
import re
import os
from datetime import datetime


import os
import shutil


STEPS_BEFORE_EPISODE_START = 10


# FLAGS = flags.FLAGS

# flags.DEFINE_string(
#     'save_dir', '/tmp/rlbench_videos/',
#     'Where to save the generated videos.')
# flags.DEFINE_list(
#     'tasks', [], 'The tasks to record. If empty, all tasks are recorded.')
# flags.DEFINE_boolean(
#     'individual', True, 'One long clip of all the tasks, or individual videos.')
# flags.DEFINE_boolean(
#     'domain_randomization', False, 'If domain randomization should be applied.')
# flags.DEFINE_string(
#     'textures_path', '',
#     'Where to locate textures if using domain randomization.')
# flags.DEFINE_boolean('headless', True, 'Run in headless mode.')
# flags.DEFINE_list(
#     'camera_resolution', [1280, 720], 'The camera resolution')



class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy,
                 speed: float, init_rotation: float = np.deg2rad(0)):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])
        # self._last_angle = self.origin.get_orientation()[2]  # Z-axis rotation
        self.origin.set_orientation([0.0, 0.0, 0.240])
    
    def get_speed(self):
        pass
        

    def step(self):
        pass
        # self.origin.set_orientation([0.0, 0.0, 0.240])
        # self.origin.rotate([0, 0, self.speed])
        # self._last_angle = self.origin.get_orientation()[2]






class Scene(object):
    """Controls what is currently in the vrep scene. This is used for making
    sure that the tasks are easily reachable. This may be just replaced by
    environment. Responsible for moving all the objects. """

    def __init__(self,
                 pyrep: PyRep,
                 robot: Robot,
                 robot_2: Robot = None,
                 obs_config: ObservationConfig = ObservationConfig(),
                 robot_setup: str = 'panda',
                 save_directory:str = "/home/sujan/VoxPoser_Feb 28/VoxPoser/data/test/"):
        
        self.save_directory = save_directory
        
        self._camera_lock = threading.Lock()
        self._scene_lock = threading.Lock()

        print(f'Save directory from scene: {self.save_directory}')

        # Overwrite the directory if it exists
        if os.path.exists(self.save_directory):
            shutil.rmtree(self.save_directory)
        os.makedirs(self.save_directory)
        # # Get list of existing 'run_X' directories
        # existing_runs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and re.match(r'run_\d+', d)]

        # # Extract numbers and find the next available one
        # run_numbers = [int(re.search(r'run_(\d+)', d).group(1)) for d in existing_runs]
        # next_run = max(run_numbers, default=0) + 1

        # # Create the new directory
        # new_run_dir = os.path.join(base_path, f'run_{next_run}')
        # os.makedirs(new_run_dir)

        # print(f"Created run directory: {new_run_dir}")

        # # You can now store the path or use it for logging, saving files, etc.
        # self.run_dir = new_run_dir
        
        # #save_directory_for_each_run
        # self.run_dir = save_directory



        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam = VisionSensor.create([1280, 720])
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)
        cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)

        self._frame_counter = 0


        self._cam_motion = cam_motion
        self._fps = 30
        self._snaps = []
        self._current_snaps = []
        
        self.pyrep = pyrep
        self.robot = robot
        self.robot_setup = robot_setup

        if robot_2:
            self.robot_2 = robot_2
            self.robot_setup_2 = 'franka'
            self._start_arm_joint_pos_2 = robot_2.arm.get_joint_positions()
            self._starting_gripper_joint_pos_2 = robot_2.gripper.get_joint_positions()

        self.task = None
        self._obs_config = obs_config
        self._initial_task_state = None
        self._start_arm_joint_pos = robot.arm.get_joint_positions()
        self._starting_gripper_joint_pos = robot.gripper.get_joint_positions()
        self._workspace = Shape('workspace')
        self._workspace_boundary = SpawnBoundary([self._workspace])
        self._cam_over_shoulder_left = VisionSensor('cam_over_shoulder_left')
        self._cam_over_shoulder_right = VisionSensor('cam_over_shoulder_right')
        self._cam_overhead = VisionSensor('cam_overhead')
        self._cam_overhead_0 = VisionSensor('cam_overhead0')
        self._cam_wrist = VisionSensor('cam_wrist')
        self._cam_front = VisionSensor('cam_front')
        self._cam_front_0 = VisionSensor('cam_front0') #cam_front0
        

        self._cam_over_shoulder_left_mask = VisionSensor(
            'cam_over_shoulder_left_mask')
        self._cam_over_shoulder_right_mask = VisionSensor(
            'cam_over_shoulder_right_mask')
        self._cam_overhead_mask = VisionSensor('cam_overhead_mask')
        self._cam_overhead_mask_0 = VisionSensor('cam_overhead_mask_0')
        self._cam_wrist_mask = VisionSensor('cam_wrist_mask')
        self._cam_front_mask = VisionSensor('cam_front_mask')
        self._cam_front_mask_0 = VisionSensor('cam_front_mask0') #cam_front0
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

        self._initial_robot_state = (robot.arm.get_configuration_tree(),
                                     robot.gripper.get_configuration_tree())
        
        
        self._initial_robot_state_2 = (robot_2.arm.get_configuration_tree(),
                                     robot_2.gripper.get_configuration_tree())

        # Set camera properties from observation config
        self._set_camera_properties()

        x, y, z = self._workspace.get_position()
        minx, maxx, miny, maxy, _, _ = self._workspace.get_bounding_box()
        self._workspace_minx = x - np.fabs(minx) - 0.2
        self._workspace_maxx = x + maxx + 0.2
        self._workspace_miny = y - np.fabs(miny) - 0.2
        self._workspace_maxy = y + maxy + 0.2
        self._workspace_minz = z
        self._workspace_maxz = z + 1.0  # 1M above workspace

        self.target_workspace_check = Dummy.create()
        self._step_callback = None

        self._robot_shapes = self.robot.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE)
        self._joint_position_action = None


    
    def take_snap(self):
        self._cam_motion.step()
            # Capture and format the frame
        frame = (self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8)
        
        # Save in memory
        self._current_snaps.append(frame)
        self._snaps.extend(self._current_snaps)

        
        filename = os.path.join(self.save_directory, f"frame_{self._frame_counter:05d}.png")
        imageio.imwrite(filename, frame)
        self._frame_counter += 1
        
        
        # # Generate a filename using timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # filename = os.path.join(self.save_directory, f"frame_{timestamp}.png")
        
        # # Save the image
        # imageio.imwrite(filename, frame)

    def save(self, path):
        print('Converting to video ...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # OpenCV QT version can conflict with PyRep, so import here
        import cv2
        video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self._fps,
                tuple(self._cam_motion.cam.get_resolution([1280, 720])))
        for image in self._snaps:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()
        self._snaps = []

    def load(self, task: Task) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.

        """
        task.load()  # Load the task in to the scene

        # Set at the centre of the workspace
        task.get_base().set_position(self._workspace.get_position())

        self._initial_task_state = task.get_state()
        self.task = task
        self._initial_task_pose = task.boundary_root().get_orientation()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks. """
        if self.task is not None:
            self.robot.gripper.release()
            if self.robot_2:
                self.robot_2.gripper.release()
            if self._has_init_task:
                self.task.cleanup_()
            self.task.unload()
        self.task = None
        self._variation_index = 0

    def init_task(self) -> None:
        self.task.init_task()
        self._initial_task_state = self.task.get_state()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(self, index: int, randomly_place: bool=True,
                     max_attempts: int = 5, place_demo: bool = False) -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace.
        """

        self._variation_index = index

        if not self._has_init_task:
            self.init_task()

        # Try a few times to init and place in the workspace
        self._attempts = 0
        descriptions = None
        while self._attempts < max_attempts:
            descriptions = self.task.init_episode(index)
            try:
                if (randomly_place and
                        not self.task.is_static_workspace()):
                    
                    self._place_task()
                    # if self.robot.arm.check_arm_collision():
                    #     raise BoundaryError()
                if not place_demo:
                    self.task.validate()
                    break
                else:
                    # Placing demo, run the number of attempts for correct demo reset
                    self._attempts += 1
            except (BoundaryError, WaypointError) as e:
                self.task.cleanup_()
                self.task.restore_state(self._initial_task_state)
                self._attempts += 1
                if self._attempts >= max_attempts:
                    raise e

        # Let objects come to rest
        [self.pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True
        return descriptions

    def reset(self) -> None:
        """Resets the joint angles. """
        self.robot.gripper.release()
        self.robot_2.gripper.release()

        arm, gripper = self._initial_robot_state
        arm_2, gripper_2 = self._initial_robot_state_2

        self.pyrep.set_configuration_tree(arm)
        self.pyrep.set_configuration_tree(gripper)

        self.pyrep.set_configuration_tree(arm_2)
        self.pyrep.set_configuration_tree(gripper_2)

        self.robot.arm.set_joint_positions(self._start_arm_joint_pos, disable_dynamics=True)
        self.robot.arm.set_joint_target_velocities(
            [0] * len(self.robot.arm.joints))
        self.robot.gripper.set_joint_positions(
            self._starting_gripper_joint_pos, disable_dynamics=True)
        self.robot.gripper.set_joint_target_velocities(
            [0] * len(self.robot.gripper.joints))
        

        if self.robot_2:
            self.robot_2.arm.set_joint_positions(self._start_arm_joint_pos_2, disable_dynamics=True)
            self.robot_2.arm.set_joint_target_velocities(
                [0] * len(self.robot_2.arm.joints))
            self.robot_2.gripper.set_joint_positions(
                self._starting_gripper_joint_pos_2, disable_dynamics=True)
            self.robot_2.gripper.set_joint_target_velocities(
                [0] * len(self.robot_2.gripper.joints))

        if self.task is not None and self._has_init_task:
            self.task.cleanup_()
            self.task.restore_state(self._initial_task_state)
        self.task.set_initial_objects_in_scene()

    def get_observation(self) -> Observation:
        tip = self.robot.arm.get_tip()
        tip_2 = self.robot_2.arm.get_tip()

        joint_forces = None
        if self._obs_config.joint_forces:
            fs = self.robot.arm.get_joint_forces()
            vels = self.robot.arm.get_joint_target_velocities()
            joint_forces = self._obs_config.joint_forces_noise.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))
        
        joint_forces_2 = None
        if self._obs_config.joint_forces_2 and self.robot_2:
            fs = self.robot_2.arm.get_joint_forces()
            vels = self.robot_2.arm.get_joint_target_velocities()
            joint_forces_2 = self._obs_config.joint_forces_noise_2.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        ee_forces_flat = None
        if self._obs_config.gripper_touch_forces:
            ee_forces = self.robot.gripper.get_touch_sensor_forces()
            ee_forces_flat = []
            for eef in ee_forces:
                ee_forces_flat.extend(eef)
            ee_forces_flat = np.array(ee_forces_flat)
        ee_forces_flat_2 = None
        if self._obs_config.gripper_touch_forces_2:
            ee_forces = self.robot_2.gripper.get_touch_sensor_forces()
            ee_forces_flat_2 = []
            for eef in ee_forces:
                ee_forces_flat_2.extend(eef)
            ee_forces_flat_2 = np.array(ee_forces_flat_2)



        lsc_ob = self._obs_config.left_shoulder_camera
        rsc_ob = self._obs_config.right_shoulder_camera
        oc_ob = self._obs_config.overhead_camera
        oc_ob_0 = self._obs_config.overhead_camera_0
        wc_ob = self._obs_config.wrist_camera
        fc_ob = self._obs_config.front_camera
        fc_ob_0 = self._obs_config.front_camera_0

        lsc_mask_fn, rsc_mask_fn, oc_mask_fn, oc_mask_fn_0, wc_mask_fn, fc_mask_fn, fc_mask_fn_0 = [
            (rgb_handles_to_mask if c.masks_as_one_channel else lambda x: x
            ) for c in [lsc_ob, rsc_ob, oc_ob, oc_ob_0, wc_ob, fc_ob, fc_ob_0]]

        def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                        get_pcd: bool, rgb_noise: NoiseModel,
                        depth_noise: NoiseModel, depth_in_meters: bool):
            rgb = depth = pcd = None
            if sensor is not None and (get_rgb or get_depth):

                # Debugging

                print(f"[DEBUG] {sensor.get_name()}: Handle = {sensor._handle}")
                try:
                    # Extra safeguard in case PyRep fails silently
                    # with self._camera_lock:
                    res = sensor.get_resolution()
                    sensor.handle_explicitly()
                except Exception as e:
                    print(f"[ERROR] Failed to handle sensor {sensor.get_name()}: {e}")
                    raise
                
                # sensor.handle_explicitly()
                if get_rgb:
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
                if get_depth or get_pcd:
                    depth = sensor.capture_depth(depth_in_meters)
                    if depth_noise is not None:
                        depth = depth_noise.apply(depth)
                if get_pcd:
                    depth_m = depth
                    if not depth_in_meters:
                        near = sensor.get_near_clipping_plane()
                        far = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                    pcd = sensor.pointcloud_from_depth(depth_m)
                    if not get_depth:
                        depth = None
            return rgb, depth, pcd

        def get_mask(sensor: VisionSensor, mask_fn):
            mask = None
            if sensor is not None:
                sensor.handle_explicitly()
                mask = mask_fn(sensor.capture_rgb())
            return mask

        left_shoulder_rgb, left_shoulder_depth, left_shoulder_pcd = get_rgb_depth(
            self._cam_over_shoulder_left, lsc_ob.rgb, lsc_ob.depth, lsc_ob.point_cloud,
            lsc_ob.rgb_noise, lsc_ob.depth_noise, lsc_ob.depth_in_meters)
        right_shoulder_rgb, right_shoulder_depth, right_shoulder_pcd = get_rgb_depth(
            self._cam_over_shoulder_right, rsc_ob.rgb, rsc_ob.depth, rsc_ob.point_cloud,
            rsc_ob.rgb_noise, rsc_ob.depth_noise, rsc_ob.depth_in_meters)
        overhead_rgb, overhead_depth, overhead_pcd = get_rgb_depth(
            self._cam_overhead, oc_ob.rgb, oc_ob.depth, oc_ob.point_cloud,
            oc_ob.rgb_noise, oc_ob.depth_noise, oc_ob.depth_in_meters)
        
        overhead_rgb_0, overhead_depth_0, overhead_pcd_0 = get_rgb_depth(
            self._cam_overhead_0, oc_ob_0.rgb, oc_ob_0.depth, oc_ob_0.point_cloud,
            oc_ob_0.rgb_noise, oc_ob_0.depth_noise, oc_ob_0.depth_in_meters)
        
        wrist_rgb, wrist_depth, wrist_pcd = get_rgb_depth(
            self._cam_wrist, wc_ob.rgb, wc_ob.depth, wc_ob.point_cloud,
            wc_ob.rgb_noise, wc_ob.depth_noise, wc_ob.depth_in_meters)
        front_rgb, front_depth, front_pcd = get_rgb_depth(
            self._cam_front, fc_ob.rgb, fc_ob.depth, fc_ob.point_cloud,
            fc_ob.rgb_noise, fc_ob.depth_noise, fc_ob.depth_in_meters)
        
        front_rgb_0, front_depth_0, front_pcd_0 = get_rgb_depth(
            self._cam_front_0, fc_ob_0.rgb, fc_ob_0.depth, fc_ob_0.point_cloud,
            fc_ob_0.rgb_noise, fc_ob_0.depth_noise, fc_ob_0.depth_in_meters)

        left_shoulder_mask = get_mask(self._cam_over_shoulder_left_mask,
                                    lsc_mask_fn) if lsc_ob.mask else None
        right_shoulder_mask = get_mask(self._cam_over_shoulder_right_mask,
                                    rsc_mask_fn) if rsc_ob.mask else None
        overhead_mask = get_mask(self._cam_overhead_mask,
                                oc_mask_fn) if oc_ob.mask else None
        overhead_mask_0 = get_mask(self._cam_overhead_mask_0,
                                oc_mask_fn_0) if oc_ob_0.mask else None
        wrist_mask = get_mask(self._cam_wrist_mask,
                            wc_mask_fn) if wc_ob.mask else None
        front_mask = get_mask(self._cam_front_mask,
                            fc_mask_fn) if fc_ob.mask else None
        front_mask_0 = get_mask(self._cam_front_mask_0,
                            fc_mask_fn_0) if fc_ob_0.mask else None

        obs = Observation(
            left_shoulder_rgb=left_shoulder_rgb,
            left_shoulder_depth=left_shoulder_depth,
            left_shoulder_point_cloud=left_shoulder_pcd,
            right_shoulder_rgb=right_shoulder_rgb,
            right_shoulder_depth=right_shoulder_depth,
            right_shoulder_point_cloud=right_shoulder_pcd,
            overhead_rgb=overhead_rgb,
            overhead_depth=overhead_depth,
            overhead_point_cloud=overhead_pcd,
            overhead_rgb_0=overhead_rgb_0,
            overhead_depth_0=overhead_depth_0,
            overhead_point_cloud_0=overhead_pcd_0,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            wrist_point_cloud=wrist_pcd,
            front_rgb=front_rgb,
            front_depth=front_depth,
            front_point_cloud=front_pcd,
            front_rgb_0=front_rgb_0,
            front_depth_0=front_depth_0,
            front_point_cloud_0=front_pcd_0,
            left_shoulder_mask=left_shoulder_mask,
            right_shoulder_mask=right_shoulder_mask,
            overhead_mask=overhead_mask,
            overhead_mask_0=overhead_mask_0,
            wrist_mask=wrist_mask,
            front_mask=front_mask,
            front_mask_0=front_mask_0,
            joint_velocities=(
                self._obs_config.joint_velocities_noise.apply(
                    np.array(self.robot.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities else None),
            joint_positions=(
                self._obs_config.joint_positions_noise.apply(
                    np.array(self.robot.arm.get_joint_positions()))
                if self._obs_config.joint_positions else None),
            joint_forces=(joint_forces
                        if self._obs_config.joint_forces else None),
            gripper_open=(
                (1.0 if self.robot.gripper.get_open_amount()[0] > 0.9 else 0.0)
                if self._obs_config.gripper_open else None),
            gripper_pose=(
                np.array(tip.get_pose())
                if self._obs_config.gripper_pose else None),
            gripper_matrix=(
                tip.get_matrix()
                if self._obs_config.gripper_matrix else None),
            gripper_touch_forces=(
                ee_forces_flat
                if self._obs_config.gripper_touch_forces else None),
            gripper_joint_positions=(
                np.array(self.robot.gripper.get_joint_positions())
                if self._obs_config.gripper_joint_positions else None),

            joint_velocities_2=(
                self._obs_config.joint_velocities_noise_2.apply(
                    np.array(self.robot_2.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities_2 else None),
            joint_positions_2=(
                self._obs_config.joint_positions_noise_2.apply(
                    np.array(self.robot_2.arm.get_joint_positions()))
                if self._obs_config.joint_positions_2 else None),
            joint_forces_2=(joint_forces_2
                        if self._obs_config.joint_forces_2 else None),
            gripper_open_2=(
                (1.0 if self.robot_2.gripper.get_open_amount()[0] > 0.9 else 0.0)
                if self._obs_config.gripper_open_2 else None),
            gripper_pose_2=(
                np.array(tip_2.get_pose())
                if self._obs_config.gripper_pose_2 else None),
            gripper_matrix_2=(
                tip_2.get_matrix()
                if self._obs_config.gripper_matrix_2 else None),
            gripper_touch_forces_2=(
                ee_forces_flat_2
                if self._obs_config.gripper_touch_forces_2 else None),
            gripper_joint_positions_2=(
                np.array(self.robot_2.gripper.get_joint_positions())
                if self._obs_config.gripper_joint_positions_2 else None),

            task_low_dim_state=(
                self.task.get_low_dim_state() if
                self._obs_config.task_low_dim_state else None),
            misc=self._get_misc())
        obs = self.task.decorate_observation(obs)
        return obs

    def step(self):
        self.pyrep.step()
        self.task.step()
        self.take_snap()
        if self._step_callback is not None:
            self._step_callback()





    def register_step_callback(self, func):
        self._step_callback = func

    def get_demo(self, record: bool = True,
                 callable_each_step: Callable[[Observation], None] = None,
                 randomly_place: bool = True) -> Demo:
        """Returns a demo (list of observations)"""

        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints, waypoints_2 = self.task.get_waypoints()
        
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)
        
        if len(waypoints_2) == 0:
            raise NoWaypointsError(
                'No waypoints were found for robot_2.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            self._joint_position_action = None
            gripper_open = 1.0 if self.robot.gripper.get_open_amount()[0] > 0.9 else 0.0
            demo.append(self.get_observation())
        while True:
            success = False
            for i, point in enumerate(waypoints):
                point.start_of_path()
                if point.skip:
                    continue
                grasped_objects = self.robot.gripper.get_grasped_objects()
                colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                    object_type=ObjectType.SHAPE) if s not in grasped_objects
                                    and s not in self._robot_shapes and s.is_collidable()
                                    and self.robot.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                    self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    gripper = self.robot.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                gripper_open = 1.0
                                done = gripper.actuate(gripper_open, 0.04)
                                self.step()
                                self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                gripper_open = 0.0
                                done = gripper.actuate(gripper_open, 0.04)
                                self.step()
                                self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            gripper_open = num
                            done = gripper.actuate(gripper_open, 0.04)
                            self.step()
                            self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.step()
                self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        processed_demo = Demo(demo)
        processed_demo.num_reset_attempts = self._attempts + 1
        return processed_demo

    def get_observation_config(self) -> ObservationConfig:
        return self._obs_config

    def check_target_in_workspace(self, target_pos: np.ndarray) -> bool:
        x, y, z = target_pos
        return (self._workspace_maxx > x > self._workspace_minx and
                self._workspace_maxy > y > self._workspace_miny and
                self._workspace_maxz > z > self._workspace_minz)

    def _demo_record_step(self, demo_list, record, func):
        if record:
            demo_list.append(self.get_observation())
        if func is not None:
            func(self.get_observation())

    def _set_camera_properties(self) -> None:
        def _set_rgb_props(rgb_cam: VisionSensor,
                           rgb: bool, depth: bool, conf: CameraConfig):
            if not (rgb or depth or conf.point_cloud):
                rgb_cam.remove()
            else:
                rgb_cam.set_explicit_handling(1)
                rgb_cam.set_resolution(conf.image_size)
                rgb_cam.set_render_mode(conf.render_mode)

        def _set_mask_props(mask_cam: VisionSensor, mask: bool,
                            conf: CameraConfig):
                if not mask:
                    mask_cam.remove()
                else:
                    mask_cam.set_explicit_handling(1)
                    mask_cam.set_resolution(conf.image_size)
        _set_rgb_props(
            self._cam_over_shoulder_left,
            self._obs_config.left_shoulder_camera.rgb,
            self._obs_config.left_shoulder_camera.depth,
            self._obs_config.left_shoulder_camera)
        _set_rgb_props(
            self._cam_over_shoulder_right,
            self._obs_config.right_shoulder_camera.rgb,
            self._obs_config.right_shoulder_camera.depth,
            self._obs_config.right_shoulder_camera)
        _set_rgb_props(
            self._cam_overhead,
            self._obs_config.overhead_camera.rgb,
            self._obs_config.overhead_camera.depth,
            self._obs_config.overhead_camera)
        # _set_rgb_props(
        #     self._cam_overhead_0,
        #     self._obs_config.overhead_camera_0.rgb,
        #     self._obs_config.overhead_camera_0.depth,
        #     self._obs_config.overhead_camera_0)
        self._cam_overhead_0.set_explicit_handling(1)
        _set_rgb_props(
            self._cam_wrist, self._obs_config.wrist_camera.rgb,
            self._obs_config.wrist_camera.depth,
            self._obs_config.wrist_camera)
        _set_rgb_props(
            self._cam_front, self._obs_config.front_camera.rgb,
            self._obs_config.front_camera.depth,
            self._obs_config.front_camera)
        # _set_rgb_props(
        #     self._cam_front_0, self._obs_config.front_camera_0.rgb,
        #     self._obs_config.front_camera_0.depth,
        #     self._obs_config.front_camera_0)
        self._cam_front_0.set_explicit_handling(1)
        _set_mask_props(
            self._cam_over_shoulder_left_mask,
            self._obs_config.left_shoulder_camera.mask,
            self._obs_config.left_shoulder_camera)
        _set_mask_props(
            self._cam_over_shoulder_right_mask,
            self._obs_config.right_shoulder_camera.mask,
            self._obs_config.right_shoulder_camera)
        _set_mask_props(
            self._cam_overhead_mask,
            self._obs_config.overhead_camera.mask,
            self._obs_config.overhead_camera)
        # _set_mask_props(
        #     self._cam_overhead_mask_0,
        #     self._obs_config.overhead_camera_0.mask,
        #     self._obs_config.overhead_camera_0)
        self._cam_overhead_mask_0.set_explicit_handling(1)
        _set_mask_props(
            self._cam_wrist_mask, self._obs_config.wrist_camera.mask,
            self._obs_config.wrist_camera)
        _set_mask_props(
            self._cam_front_mask, self._obs_config.front_camera.mask,
            self._obs_config.front_camera)
        # _set_mask_props(
        #     self._cam_front_mask_0, self._obs_config.front_camera_0.mask,
        #     self._obs_config.front_camera_0)

        self._cam_front_mask_0.set_explicit_handling(1)

    def _place_task(self) -> None:
        self._workspace_boundary.clear()
        # Find a place in the robot workspace for task

        # Instead of sampling a new position, just reset to initial orientation and position
        self.task.boundary_root().set_position(self._workspace.get_position())
        self.task.boundary_root().set_orientation(self._initial_task_pose)
        
        #########################################################
        ### code for changing object rotation and orientation####
        #########################################################

        # self.task.boundary_root().set_orientation(
        #     self._initial_task_pose)
        # min_rot, max_rot = self.task.base_rotation_bounds()
        # self._workspace_boundary.sample(
        #     self.task.boundary_root(),
        #     min_rotation=min_rot, max_rotation=max_rot)

    def _get_misc(self):
        def _get_cam_data(cam: VisionSensor, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    '%s_extrinsics' % name: cam.get_matrix(),
                    '%s_intrinsics' % name: cam.get_intrinsic_matrix(),
                    '%s_near' % name: cam.get_near_clipping_plane(),
                    '%s_far' % name: cam.get_far_clipping_plane(),
                }
            return d
        misc = _get_cam_data(self._cam_over_shoulder_left, 'left_shoulder_camera')
        misc.update(_get_cam_data(self._cam_over_shoulder_right, 'right_shoulder_camera'))
        misc.update(_get_cam_data(self._cam_overhead, 'overhead_camera'))
        misc.update(_get_cam_data(self._cam_overhead_0, 'overhead_camera_0'))
        misc.update(_get_cam_data(self._cam_front, 'front_camera'))
        misc.update(_get_cam_data(self._cam_front_0, 'front_camera_0'))
        misc.update(_get_cam_data(self._cam_wrist, 'wrist_camera'))
        misc.update({"variation_index": self._variation_index})
        if self._joint_position_action is not None:
            # Store the actual requested joint positions during demo collection
            misc.update({"joint_position_action": self._joint_position_action})
        joint_poses = [j.get_pose() for j in self.robot.arm.joints]
        misc.update({'joint_poses': joint_poses})
        return misc
