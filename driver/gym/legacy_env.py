from typing import List

import numpy as np
from driver.car import FixedPlanCar, LegacyRewardCar
from driver.car.car import State
from driver.gym.car_env import CarEnv
from driver.world import ThreeLaneCarWorld

import gym  # type: ignore


def make_legacy_plan(initial_state: np.ndarray, horizon: int = 50) -> List[np.ndarray]:
    assert horizon % 5 == 0
    phase_length = horizon // 5
    plan = [np.array([0, initial_state[3]])] * phase_length
    plan.extend([np.array([1.0, initial_state[3]])] * phase_length)
    plan.extend([np.array([-1.0, initial_state[3]])] * phase_length)
    plan.extend([np.array([-1.0, initial_state[3] * 1.3])] * (2 * phase_length))
    return plan


class LegacyEnv(CarEnv):
    def __init__(self, reward: np.ndarray):
        world = ThreeLaneCarWorld()
        main_car = LegacyRewardCar(
            env=world,
            init_state=np.array([0.0, -0.3, np.pi / 2.0, 0.4]),
            weights=reward,
            color="white",
        )

        other_init_state = np.array([world.lane_width, 0.0, np.pi / 2.0, 0.41])
        other_plan = make_legacy_plan(other_init_state)
        other_car = FixedPlanCar(
            env=world,
            init_state=other_init_state,
            plan=other_plan,
            color="orange",
            legacy_state=True,
        )
        world.add_cars([main_car, other_car])
        super().__init__(world, main_car_index=0)


if __name__ == "__main__":
    LegacyEnv(reward=np.array([0, 0, 0, 0]))
