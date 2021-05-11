"""Planner that assumes all other cars travel at constant velocity."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import tensorflow as tf  # type: ignore
from driver.car.car import Car
from driver.planner.car_planner import CarPlanner
from driver.simulation_utils import next_car_state

if TYPE_CHECKING:
    from driver.world import CarWorld


class NaivePlanner(CarPlanner):
    """
    MPC-based CarPlanner that assumes all the other cars are FixedVelocityCars.
    """

    def __init__(
        self,
        world: CarWorld,
        car: Car,
        horizon: int,
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(learning_rate=0.1),
        n_iter: int = 100,
        leaf_evaluation=None,
        extra_inits=False,
        init_controls: Optional[List[List[List[float]]]] = None,
        log_best_init: bool = False,
    ):
        super().__init__(world, car)
        self.leaf_evaluation = leaf_evaluation
        self.reward_func = self.initialize_mpc_reward()
        self.horizon = horizon
        self.planned_controls = [tf.Variable([0.0, 0.0]) for _ in range(horizon)]
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.extra_inits = extra_inits
        self.init_controls = init_controls
        self.log_best_init = log_best_init

    def initialize_mpc_reward(self):
        @tf.function
        def mpc_reward(
            init_state: Union[tf.Tensor, tf.Variable],
            controls: Union[tf.Tensor, tf.Variable],
            other_controls: Optional[List[Union[tf.Tensor, tf.Variable]]] = None,
            weights: Union[None, tf.Tensor, tf.Variable, np.ndarray] = None,
        ):
            world_state = init_state
            dt = self.world.dt
            r = 0

            for ctrl_idx in range(self.horizon):
                control = controls[ctrl_idx]
                new_state = []
                for i, car in enumerate(self.world.cars):
                    x = world_state[i]
                    if i == self.car.index:
                        new_x = car.dynamics_fn(x, control, dt)
                    else:
                        # we assume 0 friction
                        if other_controls is not None:
                            v, angle = x[2], x[3]
                            acc, ang_vel = (
                                other_controls[i][ctrl_idx][0],
                                other_controls[i][ctrl_idx][1],
                            )
                            update = tf.stack(
                                [
                                    tf.cos(angle) * (v * dt + 0.5 * acc * dt ** 2),
                                    tf.sin(angle) * (v * dt + 0.5 * acc * dt ** 2),
                                    acc * dt,
                                    ang_vel * dt,
                                ]
                            )
                        else:
                            v, angle = x[2], x[3]
                            update = tf.stack(
                                [tf.cos(angle) * v * dt, tf.sin(angle) * v * dt, 0.0, 0.0]
                            )
                        new_x = x + update
                    new_state.append(new_x)
                world_state = tf.stack(new_state, axis=0)
                if ctrl_idx == self.horizon - 1 and self.leaf_evaluation is not None:
                    r += self.leaf_evaluation(world_state, control)
                else:
                    if weights is not None:
                        r += self.car.reward_fn(world_state, control, weights)
                    else:
                        r += self.car.reward_fn(world_state, control)
            return r

        return mpc_reward

    def generate_plan(
        self,
        init_state: Union[None, tf.Tensor, tf.Variable, np.ndarray] = None,
        weights: Union[None, tf.Tensor, tf.Variable, np.ndarray] = None,
        other_controls: Optional[List] = None,
        use_lbfgs=False,
    ) -> List[tf.Variable]:

        """
        Generates a sequence of controls of length self.horizon by performing
        gradient ascent on the predicted reward of the resulting trajectory.

        Args:
            init_state: The initial state to plan from. If none, we use the
                        current state of the world associated with the car.
            weights: The weights of the reward function (if any).
                (Note: weights should only be not None if the reward function
                of the car associated with this planner takes as input a weight
                vector.)
            other_controls: List of sequences of controls belonging to the other cars.
            use_lbfgs: if true, use L-BFGS for optimization; otherwise, SGD is used.

        Returns:

        """

        init_controls = self.init_controls if self.init_controls is not None else []
        init_controls.append([[0.0, 0.0]] * len(self.planned_controls))
        init_controls.append([[0, -5 * 0.13]] * len(self.planned_controls))
        init_controls.append([[0, 5 * 0.13]] * len(self.planned_controls))

        if self.extra_inits:
            init_controls.append(
                [
                    [self.car.friction * self.car.state[2] ** 2, 0.0]
                    for _ in range(len(self.planned_controls))
                ]
            )
            init_controls.append(
                [
                    [self.car.friction * self.car.state[2] ** 2, -5 * 0.13]
                    for _ in range(len(self.planned_controls))
                ]
            )
            init_controls.append(
                [
                    [self.car.friction * self.car.state[2] ** 2, 5 * 0.13]
                    for _ in range(len(self.planned_controls))
                ]
            )  # use control bounds from old code base for left/right initialization

        if init_state is None:
            init_state = self.world.state

        def loss():
            return -self.reward_func(
                init_state, self.planned_controls, other_controls=other_controls, weights=weights
            )

        if use_lbfgs:

            def flat_controls_to_loss(flat_controls):
                return -self.reward_func(
                    init_state, tf.reshape(flat_controls, (self.horizon, 2)), weights
                )

            import tensorflow_probability as tfp  # type: ignore

            @tf.function
            def loss_and_grad(flat_controls):
                v, g = tfp.math.value_and_gradient(flat_controls_to_loss, flat_controls[0])
                return tf.convert_to_tensor([v]), tf.convert_to_tensor([g])

        best_loss = float("inf")
        best_init = -1
        for i, init_control in enumerate(init_controls):
            logging.debug(f"init_control={init_control}")
            for control, val in zip(self.planned_controls, init_control):
                control.assign(val)

            if use_lbfgs:
                opt = tfp.optimizer.lbfgs_minimize(
                    loss_and_grad,
                    initial_position=[tf.reshape(self.planned_controls, self.horizon * 2)],
                    max_iterations=200,
                    tolerance=1e-12,
                )
                current_loss = opt.objective_value.numpy()[0]
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_plan = list(tf.reshape(opt.position, (self.horizon, 2)).numpy())

            else:
                logging.debug(f"n traj opt iterations={self.n_iter}")
                for _ in range(self.n_iter):
                    self.optimizer.minimize(loss, self.planned_controls)

                current_loss = loss().numpy()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_plan = [c.numpy() for c in self.planned_controls]
                    best_init = i

        if self.log_best_init:
            logging.info(f"Best traj found from init={best_init}")

        for control, val in zip(self.planned_controls, best_plan):
            control.assign(val)

        return self.planned_controls
