from abc import abstractmethod

import numpy as np

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.scene import Scene


def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


class GripperActionMode(object):

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray, gripper_name: str):
        pass

    def action_step(self, scene: Scene, action: np.ndarray):
        pass

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        pass

    def action_post_step(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    @abstractmethod
    def action_bounds(self):
        pass


class Discrete(GripperActionMode):
    """Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open

    def _actuate(self, action, scene: Scene, gripper_name: str):
        done = False
        while not done:
            with scene._scene_lock:
                if gripper_name == "gripper":
                    done = scene.robot.gripper.actuate(action, velocity=0.2)
                elif gripper_name == "gripper1":
                    done = scene.robot_2.gripper.actuate(action, velocity=0.2)
                scene.pyrep.step()
                scene.task.step()

    def action(self, scene: Scene, action: np.ndarray, gripper_name: str):
        assert_action_shape(action, self.action_shape(scene.robot))

        with scene._scene_lock:
            robot = scene.robot if gripper_name == "gripper" else scene.robot_2

            if 0.0 > action[0] or action[0] > 1.0:
                raise InvalidActionError('Gripper action expected to be within 0 and 1.')

            open_condition = all(x > 0.9 for x in robot.gripper.get_open_amount())
            current_ee = 1.0 if open_condition else 0.0
            desired_action = float(action[0] > 0.5)

        if current_ee != desired_action:
            # If not detaching before open, actuate immediately
            if not self._detach_before_open:
                self._actuate(desired_action, scene, gripper_name)

            if desired_action == 0.0 and self._attach_grasped_objects:
                # Gripper closing, try to grasp objects
                with scene._scene_lock:
                    for g_obj in scene.task.get_graspable_objects():
                        robot.gripper.grasp(g_obj)
            else:
                # Gripper opening, release grasp
                with scene._scene_lock:
                    robot.gripper.release()

            if self._detach_before_open:
                self._actuate(desired_action, scene, gripper_name)

            if desired_action == 1.0:
                # Allow time for dropped objects to settle
                for _ in range(10):
                    with scene._scene_lock:
                        scene.pyrep.step()
                        scene.task.step()

    def action_shape(self, scene: Scene) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds."""
        return np.array([0]), np.array([1])



class GripperJointPosition(GripperActionMode):
    """Control the target joint positions absolute or delta) of the gripper.

    The action mode opoerates in absolute mode or delta mode, where delta
    mode takes the current joint positions and adds the new joint positions
    to get a set of target joint positions. The robot uses a simple control
    loop to execute until the desired poses have been reached.
    It os the users responsibility to ensure that the action lies within
    a usuable range.

    Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True,
                 absolute_mode: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open
        self._absolute_mode = absolute_mode
        self._control_mode_set = False

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene, action)
        self.action_post_step(scene, action)

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        if not self._control_mode_set:
            scene.robot.gripper.set_control_loop_enabled(True)
            self._control_mode_set = True
        assert_action_shape(action, self.action_shape(scene.robot))
        action = action.repeat(2)  # use same action for both joints
        a = action if self._absolute_mode else np.array(
            scene.robot.gripper.get_joint_positions())
        scene.robot.gripper.set_joint_target_positions(a)

    def action_step(self, scene: Scene, action: np.ndarray):
        scene.step()

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.gripper.set_joint_target_positions(
            scene.robot.gripper.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds.

        Returns: Returns the min and max of the action.
        """
        return np.array([0]), np.array([0.04])
