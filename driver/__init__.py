import gym  # type: ignore

gym.envs.register(id="driver-v1", entry_point="driver.gym_driver:GymDriver")
from .models import Driver
