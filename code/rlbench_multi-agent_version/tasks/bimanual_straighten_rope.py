from typing import List
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from collections import defaultdict

class BimanualStraightenRope(Task):

    def init_task(self) -> None:
        self.head = Shape('head')
        self.tail = Shape('tail')
        self.register_graspable_objects([self.head, self.tail])
        self.register_success_conditions(
            [DetectedCondition(Shape('head'), ProximitySensor('success_head')),
             DetectedCondition(Shape('tail'), ProximitySensor('success_tail'))])

        self.waypoint_mapping = defaultdict(lambda: 'right')
        for i in range(3):
            self.waypoint_mapping[f'waypoint{i}'] = 'left'

    def init_episode(self, index: int) -> List[str]:
        return ['straighten rope',
                'pull the rope straight',
                'grasping each end of the rope in turn, leave the rope straight'
                ' on the table',
                'pull each end of the rope until is is straight',
                'tighten the rope',
                'pull the rope tight']

    def variation_count(self) -> int:
        return 1
