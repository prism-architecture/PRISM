from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from collections import defaultdict
import random


from rlbench.backend.conditions import Condition

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False


class BimanualPickPlate(Task):

    def init_task(self) -> None:
        self.container_1 = Shape('small_container_1')
        self.container_2 = Shape('small_container_2')
        self.pepper_1 = Shape('pepper_1')
        self.pepper_2 = Shape('pepper_2')

        self.register_success_conditions([LiftedCondition(self.pepper_1, 1.0)])
        self.register_graspable_objects([self.pepper_1, self.pepper_2])
        self.waypoint_mapping = defaultdict(lambda: 'right')
        self.waypoint_mapping.update({'waypoint0': 'left', 'waypoint2': 'left', 'waypoint6': 'left'})

    def init_episode(self, index: int) -> List[str]:
        # Optionally set color for container_2
        self.container_2.set_color([0.0, 0.0, 1.0])  # Blue

        # Randomly swap container positions
        if random.random() < 0.5:
            pos_1 = self.container_1.get_position()
            pos_2 = self.container_2.get_position()
            self.container_1.set_position(pos_2)
            self.container_2.set_position(pos_1)

        # Randomly swap pepper positions
        if random.random() < 0.5:
            pos_1 = self.pepper_1.get_position()
            pos_2 = self.pepper_2.get_position()
            self.pepper_1.set_position(pos_2)
            self.pepper_2.set_position(pos_1)

        return ['pick up the plate']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 4.], [0, 0, np.pi / 4.]
