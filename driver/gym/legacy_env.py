from typing import Dict, Final, Tuple, Union

import numpy as np
from driver.car.fixed_plan_car import LegacyPlanCar
from driver.car.legacy_reward_car import LegacyRewardCar
from driver.gym.car_env import CarEnv
from driver.world import ThreeLaneCarWorld

import gym  # type: ignore
import gym.spaces  # type: ignore

from ..math_utils import safe_normalize


class LegacyEnv(gym.Env):
    HORIZON: Final[int] = 50
    State = Union[np.ndarray, Tuple[np.ndarray, int]]

    def __init__(self, reward: np.ndarray, random_start: bool = False, time_in_state: bool = False):
        """Attempts to replicate legacy/gym.py as closely as possible while using tf ops.

        Args:
            reward (np.ndarray): Weights of linear reward features
            random_start (bool, optional): Should the environment start at a random state after reset? Defaults to False.
            time_in_state (bool, optional): Should the state include how long is left in the episode? Defaults to False.
        """
        self.reward_weights = reward.astype(np.float32)
        world = ThreeLaneCarWorld()

        self.time_in_state = time_in_state

        car_space = gym.spaces.Box(
            low=np.array([[float("-inf"), float("-inf"), 0, -1]] * 2, dtype=np.float32),
            high=np.array([[float("inf"), float("inf"), 2 * np.pi, 1]] * 2),
        )

        if self.time_in_state:
            self.observation_space = gym.spaces.Tuple(
                (car_space, gym.spaces.Discrete(self.HORIZON + 1))
            )
        else:
            self.observation_space = car_space

        self.random_start = random_start
        self.random_start_space = gym.spaces.Box(
            low=np.array([-3 * world.lane_width, -0.6, 0, -1], dtype=np.float32),
            high=np.array([3 * world.lane_width, 0.3, 2 * np.pi, 2], dtype=np.float32),
        )

        if self.random_start:
            init_state = self.random_start_space.sample().astype(np.float32)
        else:
            init_state = np.array([0.0, -0.3, np.pi / 2.0, 0.4], dtype=np.float32)

        self.main_car = LegacyRewardCar(
            env=world,
            init_state=init_state,
            weights=self.reward_weights,
            color="white",
        )

        other_car = LegacyPlanCar(
            env=world,
            init_state=np.array([world.lane_width, 0.0, np.pi / 2.0, 0.41], dtype=np.float32),
        )
        world.add_cars([self.main_car, other_car])

        # Composition instead of inheritance because the state signature is different
        self.car_env = CarEnv(world)
        self.action_space = self.car_env.action_space

        self.t = 0

    @property
    def reward(self) -> np.ndarray:
        return self.reward_weights

    @reward.setter
    def reward(self, reward: np.ndarray) -> None:
        self.reward_weights = reward
        self.main_car.weights = reward

    @property
    def state(self) -> np.ndarray:
        return self.car_env.state

    @state.setter
    def state(self, state: State) -> None:
        if isinstance(state, tuple):
            car_state, time_remaining = state
            self.t = self.HORIZON - time_remaining
            self.car_env.state = car_state
        else:
            self.car_env.state = state

    def features(self, state: State) -> np.ndarray:
        if isinstance(state, tuple):
            state = state[0]
        elif len(state.shape) == 2 and state.shape[1] == 9:
            # We got a pre-flattened batch of states
            state = state[:, :8].reshape((-1, 2, 4))

        return self.main_car.features(state, None).numpy()

    def step(self, action: np.ndarray) -> Tuple[State, float, bool, Dict]:
        action = action.astype(np.float32)
        car_state, reward, _, info = self.car_env.step(action)
        info["reward_features"] = self.main_car.features(car_state, action).numpy()

        self.t += 1
        done = self.t >= self.HORIZON

        state: Union[np.ndarray, Tuple[np.ndarray, int]] = (
            (car_state, self.HORIZON - self.t) if self.time_in_state else car_state
        )

        return state, reward, done, info

    def reset(self) -> State:
        self.t = 0

        if self.random_start:
            self.main_car.init_state = self.random_start_space.sample()

        car_state = self.car_env.reset()
        state = (car_state, self.HORIZON) if self.time_in_state else car_state

        return state


class RandomLegacyEnv(LegacyEnv):
    """Legacy environment, except the reward weigths are randomized each episode"""

    REWARD_SHAPE: Tuple[int] = (4,)

    def __init__(self, random_start: bool = False, time_in_state: bool = False) -> None:
        super().__init__(
            reward=self.generate_reward(), random_start=random_start, time_in_state=time_in_state
        )

    def generate_reward(self) -> np.ndarray:
        return safe_normalize(np.random.random(self.REWARD_SHAPE))

    def step(self, action: np.ndarray) -> Tuple[LegacyEnv.State, float, bool, Dict]:
        state, reward, done, info = super().step(action)
        info["reward_weights"] = self.reward_weights
        return state, reward, done, info

    def reset(self) -> LegacyEnv.State:
        self.reward_weights = self.generate_reward()
        return super().reset()


if __name__ == "__main__":
    LegacyEnv(reward=np.array([0, 0, 0, 0]))
