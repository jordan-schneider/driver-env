from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence, Union

import numpy as np
import tensorflow as tf  # type: ignore
from driver.car.linear_reward_car import LinearRewardCar

if TYPE_CHECKING:
    from driver.world import CarWorld


class LegacyRewardCar(LinearRewardCar):
    def __init__(
        self,
        env: CarWorld,
        init_state: Union[np.ndarray, tf.Tensor, Iterable],
        weights: Union[np.ndarray, tf.Tensor, Iterable],
        color: str,
        opacity: float = 1.0,
        **kwargs
    ):
        super().__init__(
            env=env,
            init_state=init_state,
            weights=weights,
            color=color,
            opacity=opacity,
            legacy_state=True,
            **kwargs
        )

    @staticmethod
    @tf.function
    def features(
        state: Sequence[Union[tf.Tensor, tf.Variable]],
        control: Union[tf.Tensor, tf.Variable],
    ) -> tf.Tensor:
        my_state = state[0]
        other_state = state[1]

        min_dist_to_lane = tf.minimum(
            tf.minimum((my_state[0] - 0.17) ** 2, (my_state[0]) ** 2), (my_state[0] + 0.17) ** 2
        )
        staying_in_lane = tf.exp(-30 * min_dist_to_lane) / 0.15343634

        assert staying_in_lane.shape == (), staying_in_lane.shape

        # keeping speed (lower is better)
        keeping_speed = (my_state[3] - 1) ** 2 / 0.42202643
        assert keeping_speed.shape == (), keeping_speed.shape

        # heading (higher is better)
        heading = tf.sin(my_state[2]) / 0.06112367
        assert heading.shape == (), heading.shape

        # collision avoidance (lower is better)
        collision_avoidance = (
            tf.exp(
                -(
                    7.0 * (my_state[0] - other_state[0]) ** 2
                    + 3.0 * (my_state[1] - other_state[1]) ** 2
                )
            )
            / 0.15258019
        )
        assert collision_avoidance.shape == (), collision_avoidance.shape

        return tf.stack([staying_in_lane, keeping_speed, heading, collision_avoidance])
