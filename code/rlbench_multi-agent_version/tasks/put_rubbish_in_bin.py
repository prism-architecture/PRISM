from typing import List
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition


class PutRubbishInBin(Task):

    def init_task(self):
        success_sensor = ProximitySensor('success')
        self.rubbish1 = Shape('rubbish1')
        self.rubbish2 = Shape('rubbish2')
        self.bin = Shape('bin')
      
        self.register_graspable_objects([self.rubbish1, self.rubbish2 ])
        self.register_success_conditions(
            [DetectedCondition(self.rubbish1,success_sensor), DetectedCondition(self.rubbish2,success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        # tomato1 = Shape('tomato1')
        # tomato2 = Shape('tomato2')
       
        # x2, y2, z2 = self.rubbish1.get_position()

        print(f'bin position: {self.bin.get_position()}')
        print(f'rubbish1 position: {self.rubbish1.get_position()}')
        print(f'rubbish2 position: {self.rubbish2.get_position()}')
       
        self.rubbish1.set_position([0.2, -0.4, 0.76])
        self.rubbish2.set_position([0.2, 0.05, 0.76])
        self.bin.set_position([0.25870043, 0.03055028, 0.84578019])
        #     tomato2.set_position([x2, y2, z1])
        # elif pos == 2:
        #     self.rubbish.set_position([x3, y3, z2])
        #     tomato1.set_position([x2, y2, z3])

        return ['put rubbish in bin',
                'drop the rubbish into the bin',
                'pick up the rubbish and leave it in the trash can',
                'throw away the trash, leaving any other objects alone',
                'chuck way any rubbish on the table rubbish']

    def variation_count(self) -> int:
        return 1
