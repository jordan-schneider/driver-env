"""Module containing a linear reward function car class."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Union

import numpy as np
import tensorflow as tf  # type: ignore
from driver.car.car import Car
from driver.math_utils import safe_normalize

if TYPE_CHECKING:
    from driver.world import CarWorld


class LinearRewardCar(Car):
    """
    A car where the reward function is a linear function of some features.
    """

    def __init__(
        self,
        env: CarWorld,
        init_state: Union[np.ndarray, tf.Tensor, Iterable],
        weights: Union[np.ndarray, tf.Tensor, Iterable],
        color: str = "gray",
        opacity: float = 1.0,
        friction: float = 0.2,
        **kwargs,
    ):
        """
        Args:
            env: the carWorld associated with this car.
            init_state: a vector of the form (x, y, vel, angle), representing
                    the x-y coordinates, velocity, and heading of the car.
            color: the color of this car. Used for visualization.
            opacity: the opacity of the car. Used in visualization.
            friction: the friction of this car, used in the dynamics function.
        """
        super().__init__(env, init_state, color=color, opacity=opacity, friction=friction, **kwargs)

        self.weights_tf = tf.Variable(safe_normalize(weights), dtype=tf.float32)

    @tf.function
    def features(
        self, state: Iterable[Union[tf.Tensor, tf.Variable]], control: Union[tf.Tensor, tf.Variable]
    ) -> tf.Tensor:
        raise NotImplementedError

    @property
    def weights(self):
        return self.weights_tf.numpy()

    @weights.setter
    def weights(self, weights):
        self.weights_tf.assign(safe_normalize(weights))

    @tf.function
    def reward_fn(self, state, control, weights=None):
        feats = self.features(state, control)

        if weights is None:
            weights = self.weights_tf

        assert (
            feats.shape == weights.shape
        ), f"feats.shape={feats.shape}, weights.shape={weights.shape}"

        return tf.reduce_sum(weights * feats, axis=-1)
