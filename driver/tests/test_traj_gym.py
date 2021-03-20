""" Tests for consistency between trajectory based env (models.py) and the gym implementation """

import driver
import gym  # type: ignore
import numpy as np
from driver.models import Driver
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats


@settings(deadline=None)
@given(
    actions=arrays(dtype=float, shape=(10,), elements=floats(min_value=-1.0, max_value=1.0)),
    reward=arrays(dtype=float, shape=(4,), elements=floats(min_value=-1.0, max_value=1.0)),
)
def test_envs(actions: np.ndarray, reward: np.ndarray):
    traj_env = Driver()
    traj_env.set_ctrl(actions)
    traj_features = np.array(traj_env.get_features())
    traj_return = reward @ traj_features

    gym_env = gym.make("driver-v1", reward=reward, horizon=50)
    gym_actions = actions.reshape(5, 2).repeat(10, axis=0)
    rewards = np.empty(
        50,
    )
    gym_features = np.empty((50, 4))
    for i, action in enumerate(gym_actions):
        raw_state, reward, _, _ = gym_env.step(action)
        rewards[i] = reward
        gym_features[i] = gym_env.get_features(raw_state)

    gym_return = np.mean(rewards)

    assert np.abs(traj_return - gym_return) < 0.0001
    assert np.all(np.abs(traj_features - np.mean(gym_features, axis=0)) < 0.0001)
