"""Module containing the base class for planner cars."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf  # type: ignore
from driver.car.car import Car
from driver.car.legacy_reward_car import LegacyRewardCar
from driver.car.linear_reward_car import LinearRewardCar
from driver.planner.naive_planner import NaivePlanner

if TYPE_CHECKING:
    from driver.world import CarWorld


class PlannerCar(Car):
    """
    A car that performs some variant of model predictive control, maximizing the
     sum of rewards.

    TODO(chanlaw): this should probably also be an abstract class using ABCMeta.
    """

    def __init__(
        self,
        env: CarWorld,
        init_state: Union[np.ndarray, tf.Tensor, Iterable],
        horizon: int,
        color: str = "orange",
        opacity: float = 1.0,
        friction: float = 0.2,
        planner_args: dict = None,
        check_plans: bool = False,
        **kwargs
    ):
        """
        Args:
            env: the carWorld associated with this car.
            init_state: a vector of the form (x, y, vel, angle), representing
                        the x-y coordinates, velocity, and heading of the car.
            horizon: the planning horizon for this car.
            color: the color of this car. Used for visualization.
            opacity: the opacity of the car. Used in visualization.
            friction: the friction of this car, used in the dynamics function.
            planner_args: the arguments to the planner (if any).
            check_plans: checks if the other cars are fixed_control_cars, and if so feeds in their controls to optimization
        """
        super().__init__(env, init_state, color=color, opacity=opacity, friction=friction, **kwargs)
        self.horizon = horizon
        self.planner: Optional[NaivePlanner] = None
        self.plan: List[tf.Variable] = []
        self.planner_args = planner_args if planner_args is not None else {}
        self.check_plans = check_plans

    def initialize_planner(self, planner_args):
        from driver.planner.naive_planner import NaivePlanner

        self.planner = NaivePlanner(self.env, self, self.horizon, **planner_args)

    def _get_next_control(self, return_loss: bool = False):
        if self.planner is None:
            self.initialize_planner(self.planner_args)
            assert self.planner is not None

        if self.check_plans:
            other_plans = []
            for i, other_car in enumerate(self.env.cars):
                if i == self.index:
                    other_plan = [tf.constant([0.0], dtype=tf.float32)] * self.horizon
                else:
                    other_plan = []
                    for j in range(self.horizon):
                        if hasattr(other_car, "plan") and other_car.plan is not None:
                            if j < len(other_car.plan):
                                other_plan.append(other_car.plan[j])
                            else:
                                if (
                                    hasattr(other_car, "default_control")
                                    and other_car.default_control is not None
                                ):
                                    other_plan.append(other_car.default_control)
                                else:
                                    other_plan.append(tf.constant([0.0, 0.0], dtype=tf.float32))
                        else:
                            other_plan.append(tf.constant([0.0, 0.0], dtype=tf.float32))
                other_plan = tf.stack(other_plan, axis=0)
                other_plans.append(other_plan)
            self.plan, loss = self.planner.generate_plan(
                other_controls=other_plans, return_loss=True
            )
        else:
            self.plan, loss = self.planner.generate_plan(return_loss=True)

        if return_loss:
            return tf.identity(self.plan[0]), loss

        return tf.identity(self.plan[0])

    # Note: don't uncommment this or you will mess with multiple inheritance!
    # @tf.function
    # def reward_fn(self, state, control):
    #     raise NotImplementedError


class LinearPlannerCar(LinearRewardCar, PlannerCar):
    pass


class LegacyPlannerCar(LegacyRewardCar, PlannerCar):
    def __init__(
        self,
        env: CarWorld,
        init_state: Union[np.ndarray, tf.Tensor, Iterable],
        weights: Union[np.ndarray, tf.Tensor, Iterable],
        **kwargs
    ):
        super().__init__(env, init_state, weights, check_plans=True, **kwargs)
