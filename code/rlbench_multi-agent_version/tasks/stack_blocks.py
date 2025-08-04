from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedSeveralCondition
from rlbench.backend.conditions import NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors
import random

MAX_STACKED_BLOCKS = 1
DISTRACTORS = 0


class StackBlocks(Task):

    def init_task(self) -> None:
        self.blocks_stacked = 0
        self.target_blocks = [Shape('stack_blocks_target%d' % i)
                              for i in range(4)]
        self.target = Shape('stack_blocks_target_plane')
        # self.distractors = [
        #     Shape('stack_blocks_distractor%d' % i)
        #     for i in range(DISTRACTORS)]


      
        self.boundary_0 = Shape('stack_blocks_boundary0')
        self.boundary_1 = Shape('stack_blocks_boundary1')

        # self.boundaries = [Shape('stack_blocks_boundary%d' % i)
        #                    for i in range(4)]

        # self.register_graspable_objects(self.target_blocks + self.distractors)
        self.register_graspable_objects(self.target_blocks)

        # self.register_waypoint_ability_start(0, self._move_above_next_target)
        # self.register_waypoint_ability_start(3, self._move_above_drop_zone)
        # self.register_waypoint_ability_start(5, self._is_last)
        # self.register_waypoints_should_repeat(self._repeat)

    def init_episode(self, index: int) -> List[str]:
        # For each color, we want to have 2, 3 or 4 blocks stacked
        # color_index = int(index / MAX_STACKED_BLOCKS)
        # self.blocks_to_stack = 2 + index % MAX_STACKED_BLOCKS
        self.blocks_to_stack = 2
        # color_name, color_rgb = colors[color_index]


        # for b in self.target_blocks:
        #     b.set_color(color_rgb)

        self.target_blocks[0].set_color([1, 0, 0]) # red
        self.target_blocks[1].set_color([0, 0, 1]) # blue
        self.target_blocks[2].set_color([1, 1, 0]) # yellow
        self.target_blocks[3].set_color([0, 0, 0]) # black
        # self.target.set_color([0, 0, 0]) # black



        success_detector = ProximitySensor(
            'stack_blocks_success')
        self.register_success_conditions([DetectedSeveralCondition(
            self.target_blocks, success_detector, self.blocks_to_stack),
            NothingGrasped(self.robot.gripper)
        ])

        self.blocks_stacked = 0


        # color_choices = np.random.choice(
        #     list(range(color_index)) + list(
        #         range(color_index + 1, len(colors))),
        #     size=2, replace=False)
        
        # for i, ob in enumerate(self.distractors):
        #     # name, rgb = colors[color_choices[int(i / 2)]]
        #     # ob.set_color(rgb)
        #     ob.set_color([0, 1, 0]) # green



        # b = SpawnBoundary(self.boundaries)
        b0 = SpawnBoundary([self.boundary_0])
        b1 = SpawnBoundary([self.boundary_1])
        # for block in self.target_blocks + self.distractors:
        # for block in self.target_blocks:
        #     b.sample(block, min_distance=0.1)


        for block in self.target_blocks[0:2]:
            b0.sample(block, min_distance=0.1)
        
        for block in self.target_blocks[2:4]:
            b1.sample(block, min_distance=0.1)




        return ['stack two blocks']
    


    def variation_count(self) -> int:
        # return len(colors) * MAX_STACKED_BLOCKS
        return 1

    def _move_above_next_target(self, _):
        if self.blocks_stacked >= self.blocks_to_stack:
            raise RuntimeError('Should not be here.')
        w2 = Dummy('waypoint1')
        x, y, z = self.target_blocks[self.blocks_stacked].get_position()
        _, _, oz = self.target_blocks[self.blocks_stacked].get_orientation()
        ox, oy, _ = w2.get_orientation()
        w2.set_position([x, y, z])
        w2.set_orientation([ox, oy, -oz])

    def _move_above_drop_zone(self, waypoint):
        target = Shape('stack_blocks_target_plane')
        x, y, z = target.get_position()
        waypoint.get_waypoint_object().set_position(
            [x, y, z + 0.08 + 0.06 * self.blocks_stacked])

    def _is_last(self, waypoint):
        last = self.blocks_stacked == self.blocks_to_stack - 1
        waypoint.skip = last

    def _repeat(self):
        self.blocks_stacked += 1
        return self.blocks_stacked < self.blocks_to_stack