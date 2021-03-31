from typing import Iterable, Sequence, Union

import numpy as np
import tensorflow as tf  # type: ignore
from driver.car.car import Car, Control
from driver.world import CarWorld


class FixedPlanCar(Car):
    """It's a car that follows a fixed set of controls"""

    def __init__(
        self,
        env: CarWorld,
        init_state: Union[np.ndarray, tf.Tensor, Iterable],
        plan: Sequence[Union[np.ndarray, tf.Tensor, Iterable]],
        default_control: Control = None,
        color: str = "gray",
        opacity=1.0,
        **kwargs
    ):
        super().__init__(env, init_state, color, opacity, **kwargs)
        self.control_already_determined_for_current_step = True
        self.plan = plan
        self.control = default_control
        self.default_control = default_control
        self.t = 0

    def step(self, dt):
        super().step(dt)
        self.t += 1
        if self.t < len(self.plan):
            self.set_next_control(self.plan[self.t])
        else:
            self.set_next_control(self.default_control)

    def _get_next_control(self):
        return self.control

    def reset(self):
        super().reset()
        self.t = 0
        self.set_next_control(self.plan[self.t])

    @tf.function
    def reward_fn(self, world_state, self_control):
        return 0
