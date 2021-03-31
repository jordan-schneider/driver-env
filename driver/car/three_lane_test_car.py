from typing import Iterable, Sequence, Union

import numpy as np
import tensorflow as tf  # type: ignore
from driver.car import LinearRewardCar, PlannerCar
from driver.math_utils import smooth_bump, smooth_threshold
from driver.world import CarWorld


class ThreeLaneTestCar(LinearRewardCar, PlannerCar):
    def __init__(
        self,
        env: CarWorld,
        init_state,
        horizon: int,
        weights: Union[tf.Tensor, tf.Variable, np.ndarray],
        target_speed=1.0,
        color="orange",
        friction=0.2,
        opacity=1.0,
        planner_args=None,
        debug=False,
        num_lanes=3,
        **kwargs
    ):
        super().__init__(
            env,
            init_state,
            horizon=horizon,
            weights=weights,
            color=color,
            friction=friction,
            opacity=opacity,
            planner_args=planner_args,
            debug=debug,
            **kwargs
        )
        self.target_speed = np.float32(target_speed)
        self.num_lanes = num_lanes

    @tf.function
    def features(
        self, state: Sequence[Union[tf.Tensor, tf.Variable]], control: Union[tf.Tensor, tf.Variable]
    ) -> tf.Tensor:
        """
        Features this car cares about are:
            - its forward velocity
            - its squared distance to each of three lanes
            - the minimum squared distance to any of the three lanes
            - a Gaussian collision detector
            - smooth threshold feature for fencdes

        Args:
            state: the state of the world.
            control: the controls of this car in that state.

        Returns:
            tf.Tensor: the four features this car cares about.
        """

        def bounded_squared_dist(target, bound, x):
            return tf.minimum((x - target) ** 2, bound)

        feats = []
        lane_dists = []

        car_state = state[self.index]
        velocity = car_state[2] * tf.sin(car_state[3])
        feats.append(bounded_squared_dist(self.target_speed, 4 * self.target_speed ** 2, velocity))

        for i, lane in enumerate(self.env.lanes):
            lane_i_dist = lane.dist2median(car_state) * 10
            lane_dists.append(lane_i_dist)
            feats.append(lane_i_dist)
        feats.append(tf.reduce_min(lane_dists, axis=0))

        collision_feats = []
        for i, other_car in enumerate(self.env.cars):
            other_state = state[i]
            x_bump = smooth_bump(other_state[0] - 0.08, other_state[0] + 0.08)
            y_bump = smooth_bump(other_state[1] - 0.15, other_state[1] + 0.15)
            collision_feat = x_bump(car_state[0]) * y_bump(car_state[1])
            if i != self.index:
                collision_feats.append(collision_feat)

        feats.append(tf.reduce_max(collision_feats, axis=0))

        fences = (
            smooth_threshold(0.05 * self.num_lanes, width=0.05)(car_state[0])
            + smooth_threshold(0.05 * self.num_lanes, width=0.05)(-car_state[0])
        ) * abs(car_state[0])
        feats.append(fences)
        return tf.stack(feats, axis=-1)
