import logging
from typing import Any, Tuple

import gym  # type: ignore
import numpy as np
from gym.spaces import Box  # type: ignore

from . import car, dynamics, lane
from .world import World


class GymDriver(gym.Env):
    def __init__(
        self,
        reward: np.ndarray,
        horizon: int,
        random_start: bool = False,
    ) -> None:
        self.reward_weights = reward

        self.world = World()
        self.lane_width = 0.17
        center_lane = lane.StraightLane([0.0, -1.0], [0.0, 1.0], self.lane_width)
        self.world.lanes += [center_lane, center_lane.shifted(1), center_lane.shifted(-1)]
        self.world.fences += [center_lane.shifted(2), center_lane.shifted(-2)]
        self.dyn = dynamics.CarDynamics(0.1)
        # each car's state = [x, y, angle, acceleration]
        self.robot = car.Car(self.dyn, np.array([0.0, -0.3, np.pi / 2.0, 0.4]), color="orange")
        self.human = car.Car(
            self.dyn, np.array([self.lane_width, 0.0, np.pi / 2.0, 0.41]), color="white"
        )
        self.world.cars.append(self.robot)
        self.world.cars.append(self.human)
        self.initial_state = [self.robot.state, self.human.state]
        self.random_start = random_start
        self.random_start_space = Box(
            low=np.array([-3 * self.lane_width, -0.6, 0, -1]),
            high=np.array([3 * self.lane_width, 0.3, 2 * np.pi, 2]),
        )

        self.observation_space = Box(low=-1 * np.ones(shape=(2, 4)), high=np.ones(shape=(2, 4)))
        self.action_space = Box(low=-1 * np.ones(shape=(2,)), high=np.ones((2,)))

        self.horizon = horizon
        self.timestep = 0

    def state(self) -> np.ndarray:
        return np.stack([self.robot.state, self.human.state])

    def _get_human_action(self, t: int, max_t: int):
        if t < max_t // 5:
            return [0, self.initial_state[1][3]]
        elif t < 2 * max_t // 5:
            return [1.0, self.initial_state[1][3]]
        elif t < 3 * max_t // 5:
            return [-1.0, self.initial_state[1][3]]
        elif t < 4 * max_t // 5:
            return [0, self.initial_state[1][3] * 1.3]
        else:
            return [0, self.initial_state[1][3] * 1.3]

    @staticmethod
    def get_features(state: np.ndarray) -> np.ndarray:
        # staying in lane (higher is better)
        min_dist_to_lane = min(
            (state[0, 0] - 0.17) ** 2, (state[0, 0]) ** 2, (state[0, 0] + 0.17) ** 2
        )
        staying_in_lane: float = np.exp(-30 * min_dist_to_lane) / 0.15343634

        # keeping speed (lower is better)
        keeping_speed: float = (state[0, 3] - 1) ** 2 / 0.42202643

        # heading (higher is better)
        heading: float = np.sin(state[0, 2]) / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance: float = (
            np.exp(
                -(7.0 * (state[0, 0] - state[1, 0]) ** 2 + 3.0 * (state[0, 1] - state[1, 1]) ** 2)
            )
            / 0.15258019
        )

        return np.array([staying_in_lane, keeping_speed, heading, collision_avoidance], dtype=float)

    @staticmethod
    def get_feature_batch(states: np.ndarray) -> np.ndarray:
        # staying in lane (higher is better)
        min_dist_to_lane = np.min(
            ((states[:, 0, 0] - 0.17) ** 2, (states[:, 0, 0]) ** 2, (states[:, 0, 0] + 0.17) ** 2),
            axis=0,
        )
        staying_in_lane: float = np.exp(-30 * min_dist_to_lane) / 0.15343634

        # keeping speed (lower is better)
        keeping_speed: float = (states[:, 0, 3] - 1) ** 2 / 0.42202643

        # heading (higher is better)
        heading: float = np.sin(states[:, 0, 2]) / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance: float = (
            np.exp(
                -(
                    7.0 * (states[:, 0, 0] - states[:, 1, 0]) ** 2
                    + 3.0 * (states[:, 0, 1] - states[:, 1, 1]) ** 2
                )
            )
            / 0.15258019
        )

        return np.array(
            [staying_in_lane, keeping_speed, heading, collision_avoidance], dtype=float
        ).transpose()

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        self.robot.action = action
        self.human.action = self._get_human_action(self.timestep, self.horizon)

        self.robot.move()
        self.human.move()

        state = self.state()
        reward_features = self.get_features(state)
        reward = self.reward_weights @ reward_features
        done = self.timestep >= self.horizon - 1

        self.timestep += 1

        return state, reward, done, {"reward_features": reward_features}

    def reset(self) -> Any:
        self.timestep = 0
        if self.random_start:
            self.robot.state = self.random_start_space.sample()
        else:
            self.robot.state = self.initial_state[0]
        self.human.state = self.initial_state[1]
        return self.state()

    def close(self):
        pass

    def seed(self, seed=None):
        logging.warning("Environment is deterministic")
        return

    def render(self, mode="human"):
        raise NotImplementedError
