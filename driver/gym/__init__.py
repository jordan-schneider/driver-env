from driver.gym.car_env import CarEnv
from driver.gym.legacy_env import LegacyEnv

import gym  # type: ignore

gym.envs.register(id="CarWorld-v1", entry_point="driver.gym.car_env:CarEnv")
gym.envs.register(id="LegacyDriver-v1", entry_point="driver.gym.legacy_env:LegacyEnv")
gym.envs.register(id="RandomLegacyDriver-v1", entry_point="driver.gym.legacy_env:RandomLegacyEnv")
