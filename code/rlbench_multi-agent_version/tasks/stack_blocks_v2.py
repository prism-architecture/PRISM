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

MAX_STACKED_BLOCKS = 2
# DISTRACTORS = 4


class StackBlocksV2(Task):

    def init_task(self) -> None:
        self.blocks_stacked = 0
        self.target_block = Shape('block')
        self.handle_left = Shape('handle_left')
        self.handle_right = Shape('handle_right')
        self.tray = Shape('tray')
        
   

        self.register_graspable_objects([self.target_block, self.handle_left, self.handle_right])

       
    def init_episode(self, index: int) -> List[str]:
      
        # For each color, we want to have 2, 3 or 4 blocks stacked
        # color_index = int(index / MAX_STACKED_BLOCKS)
        # self.blocks_to_stack = 2 + index % MAX_STACKED_BLOCKS
        
        # color_name, color_rgb = colors[color_index]
        # for b in self.target_blocks:
        #     b.set_color(color_rgb)


        
        self.target_block.set_color((255, 0, 0))


        self.tray.set_color((105, 105, 105))
        self.handle_left.set_color((105, 105, 105))
        self.handle_right.set_color((105, 105, 105))


        
        

        return ['stack red block']

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
    