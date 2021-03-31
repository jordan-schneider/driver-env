import numpy as np
from driver.car.fixed_plan_car import LegacyPlanCar
from driver.car.legacy_reward_car import LegacyRewardCar
from driver.gym.car_env import CarEnv
from driver.world import ThreeLaneCarWorld

import gym  # type: ignore


class LegacyEnv(CarEnv):
    def __init__(self, reward: np.ndarray):
        world = ThreeLaneCarWorld()
        main_car = LegacyRewardCar(
            env=world,
            init_state=np.array([0.0, -0.3, np.pi / 2.0, 0.4]),
            weights=reward,
            color="white",
        )

        other_car = LegacyPlanCar(
            env=world, init_state=np.array([world.lane_width, 0.0, np.pi / 2.0, 0.41])
        )
        world.add_cars([main_car, other_car])
        super().__init__(world, main_car_index=0)


if __name__ == "__main__":
    LegacyEnv(reward=np.array([0, 0, 0, 0]))
