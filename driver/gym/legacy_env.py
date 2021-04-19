from typing import Dict, Final, Tuple

import numpy as np
from driver.car.fixed_plan_car import LegacyPlanCar
from driver.car.legacy_reward_car import LegacyRewardCar
from driver.gym.car_env import CarEnv
from driver.world import ThreeLaneCarWorld

import gym  # type: ignore
from gym.spaces.box import Box


class LegacyEnv(CarEnv):
    HORIZON: Final[int] = 50

    def __init__(self, reward: np.ndarray, random_start: bool = False):
        self.reward_weights = reward.astype(np.float32)
        world = ThreeLaneCarWorld()

        self.random_start = random_start

        self.random_start_space = Box(
            low=np.array([-3 * world.lane_width, -0.6, 0, -1]),
            high=np.array([3 * world.lane_width, 0.3, 2 * np.pi, 2]),
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
        super().__init__(world, main_car_index=0)

        self.t = 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = action.astype(np.float32)
        state, reward, _, info = super().step(action)
        info["reward_features"] = self.main_car.features(state, action).numpy()

        self.t += 1
        done = self.t >= 50

        return state, reward, done, info

    def reset(self) -> np.ndarray:
        self.t = 0

        if self.random_start_space:
            self.main_car.init_state = self.random_start_space.sample()

        return super().reset()


if __name__ == "__main__":
    LegacyEnv(reward=np.array([0, 0, 0, 0]))
