from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task


class BimanualPutItemInDrawer(Task):

    def init_task(self) -> None:
        # self._options = ['bottom', 'middle', 'top']
        # self._anchors = [Dummy('waypoint_anchor_%s' % opt)
        #                  for opt in self._options]
        # self._joints = [Joint('drawer_joint_%s' % opt)
        #                 for opt in self._options]
        # self._waypoint1 = Dummy('waypoint1')
        self._item = Shape('item')
        self.register_graspable_objects([self._item])

        # self.waypoint_mapping = defaultdict(lambda: 'right')
        # self.waypoint_mapping.update({'waypoint0': 'left', 'waypoint1': 'left', 'waypoint2': 'left'})


    def init_episode(self, index) -> List[str]:
        # self._variation_index = index
        # option = self._options[index]
        # anchor = self._anchors[index]
        # self._waypoint1.set_position(anchor.get_position())
        # success_sensor = ProximitySensor('success_' + option)
        # self.register_success_conditions(
        #     [DetectedCondition(self._item, success_sensor)])
        return ['put the item in the drawer']

    def variation_count(self) -> int:
        return 3 # len(self._options)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, 0], [0, 0, 0]
    

    # def variation_count(self) -> int:
    #     return 1
    
    # def is_static_workspace(self):
    #     return True
