import gym  # type: ignore
from driver.gym_env.car_env import CarEnv
from driver.gym_env.legacy_env import LegacyEnv

gym.envs.register(id="CarWorld-v1", entry_point="driver.gym_env.car_env:CarEnv")
gym.envs.register(id="LegacyDriver-v1", entry_point="driver.gym_env.legacy_env:LegacyEnv")
gym.envs.register(
    id="RandomLegacyDriver-v1", entry_point="driver.gym_env.legacy_env:RandomLegacyEnv"
)
