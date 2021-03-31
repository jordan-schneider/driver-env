"""Module containing a car that maintains a fixed speed."""

from typing import Iterable, Union

import numpy as np
import tensorflow as tf  # type: ignore
from driver.car.fixed_control_car import FixedControlCar
from driver.world import CarWorld


class FixedVelocityCar(FixedControlCar):
    """
    A car that just goes forward at a fixed velocity, specified in its initial
    state.
    """

    def __init__(
        self,
        env: CarWorld,
        init_state: Union[np.ndarray, tf.Tensor, Iterable],
        color: str = "gray",
        opacity=1.0,
        **kwargs
    ):
        # with friction set to zero, we don't need to use any controls
        controls = tf.constant(np.array([0.0, 0.0]), dtype=tf.float32)
        super().__init__(env, init_state, controls, color, opacity, friction=0.0, **kwargs)
