from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
import random


class TakeLidOffSaucepan(Task):

    def init_task(self) -> None:
        self.lid = Shape('saucepan_lid_grasp_point')
        self.red_cube = Shape('block0')

        self.objects = [self.red_cube]



        self.success_detector = ProximitySensor('success')
        self.register_graspable_objects([self.red_cube, self.lid])

        self.boundary_0 = Shape('stack_blocks_boundary0')
        self.boundary_1 = Shape('stack_blocks_boundary1')

        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.lid),
            DetectedCondition(self.lid, self.success_detector)
        ])
        self.register_success_conditions([cond_set])

    def init_episode(self, index: int) -> List[str]:

        self.red_cube.set_color([0.89, 0.65, 0.29])
        b0 = SpawnBoundary([self.boundary_0])
        b1 = SpawnBoundary([self.boundary_1])

        chosen_boundary = random.choice([b0, b1])
        b0.sample(self.red_cube)

    

 

        return ['take lid off the saucepan',
                'using the handle, lift the lid off of the pan',
                'remove the lid from the pan',
                'grip the saucepan\'s lid and remove it from the pan',
                'leave the pan open',
                'uncover the saucepan']

    def variation_count(self) -> int:
        return 1

    def reward(self) -> float:
        grasp_lid_reward = -np.linalg.norm(
            self.lid.get_position() - self.robot.arm.get_tip().get_position())
        lift_lid_reward = -np.linalg.norm(
            self.lid.get_position() - self.success_detector.get_position())
        return grasp_lid_reward + lift_lid_reward

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
        # return [0, 0, 0], [0, 0, 0]
