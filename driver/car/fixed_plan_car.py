from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Sequence, Union

import numpy as np
import tensorflow as tf  # type: ignore
from driver.car.car import Car, Control
from driver.simulation_utils import to_numpy

if TYPE_CHECKING:
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


class LegacyPlanCar(FixedPlanCar):
    def __init__(
        self,
        env: CarWorld,
        init_state: Union[np.ndarray, tf.Tensor, Iterable, None] = None,
        default_control: Control = None,
        opacity: float = 1.0,
        **kwargs
    ):
        if init_state is None:
            init_state = np.array([env.lane_width, 0.0, np.pi / 2.0, 0.41], dtype=np.float32)
        plan = self.make_legacy_plan(to_numpy(init_state))
        super().__init__(
            env,
            init_state,
            plan,
            default_control=default_control,
            color="orange",
            opacity=opacity,
            legacy_state=True,
            **kwargs
        )

    def set_init_state(self, init_state) -> None:
        self.plan = self.make_legacy_plan(to_numpy(init_state))
        self.reset()

    @staticmethod
    def make_legacy_plan(initial_state: np.ndarray, horizon: int = 50) -> List[np.ndarray]:
        assert horizon % 5 == 0
        phase_length = horizon // 5
        plan = [np.array([0, initial_state[3]], dtype=np.float32)] * phase_length
        plan.extend([np.array([1.0, initial_state[3]], dtype=np.float32)] * phase_length)
        plan.extend([np.array([-1.0, initial_state[3]], dtype=np.float32)] * phase_length)
        plan.extend(
            [np.array([0.0, initial_state[3] * 1.3], dtype=np.float32)] * (2 * phase_length)
        )
        return plan
