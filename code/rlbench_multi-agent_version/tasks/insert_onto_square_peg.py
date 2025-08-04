from typing import List
import numpy as np
from pyrep.objects import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, ConditionSet
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.const import colors


class InsertOntoSquarePeg(Task):

    def init_task(self) -> None:
        self._square_ring_1 = Shape('square_ring0')
        self._square_ring_2 = Shape('square_ring')
        self._success_centre = Dummy('success_centre')
        success_detectors = [ProximitySensor(
            'success_detector%d' % i) for i in range(4)]
        self.register_graspable_objects([self._square_ring_2, self._square_ring_1])
        success_condition = ConditionSet([DetectedCondition(
            self._square_ring_2, sd) for sd in success_detectors])
        self.register_success_conditions([success_condition])

    def init_episode(self, index: int) -> List[str]:
        # color_name, color_rgb = colors[index]
        spokes = [Shape('pillar0'), Shape('pillar1'), Shape('pillar2')]
        # chosen_pillar = np.random.choice(spokes)
        # chosen_pillar.set_color(color_rgb)
        # _, _, z = self._success_centre.get_position()
        # x, y, _ = chosen_pillar.get_position()
        # self._success_centre.set_position([x, y, z])

        # color_choices = np.random.choice(
        #     list(range(index)) + list(range(index + 1, len(colors))),
        #     size=2, replace=False)
        # spokes.remove(chosen_pillar)
        # for spoke, i in zip(spokes, color_choices):
        #     name, rgb = colors[i]
        #     spoke.set_color(rgb)
        # b = SpawnBoundary([Shape('boundary0')])
        # b.sample(self._square_ring)

        spokes[0].set_color([1.0, 0.0, 0.0]) # red 
        spokes[1].set_color([0.0, 0.0, 0.0]) # black
        spokes[2].set_color([0.0, 1,0, 0.0]) # green

        self._square_ring_1.set_color([0.0, 0.0, 1.0]) # blue
        self._square_ring_2.set_color([1.0, 1.0, 0.0]) # yellow 


        return ['put the ring on the red spoke',
                'slide the ring onto the black colored spoke',
                'place the ring onto the green spoke']

    def variation_count(self) -> int:
        return len(colors)
