import warnings

import driver.gym
import gym  # type: ignore
import numpy as np
from driver.legacy.gym_driver import GymDriver
from driver.math_utils import safe_normalize
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from numpy.testing import assert_allclose, assert_array_equal

warnings.filterwarnings("ignore", "tensorflow")


@settings(deadline=None)
@given(
    reward=arrays(
        dtype=float, shape=(4,), elements=floats(min_value=-1, max_value=1, allow_nan=False)
    ),
    action=arrays(
        dtype=float, shape=(2,), elements=floats(min_value=-2, max_value=2, allow_nan=False)
    ),
)
def test_consistency(reward: np.ndarray, action: np.ndarray):
    reward = safe_normalize(reward)
    new_env = gym.make("LegacyDriver-v1", reward=reward)
    old_env = GymDriver(reward=reward, horizon=50)

    new_state = new_env.reset()
    old_state = old_env.reset()
    assert new_state.shape == old_state.shape
    assert_allclose(new_state, old_state)

    new_state, new_reward, new_done, new_info = new_env.step(action)
    old_state, old_reward, old_done, old_info = old_env.step(action)
    assert_allclose(new_state, old_state, atol=0.001)
    assert abs(new_reward - old_reward) < 0.001
    assert new_done == old_done
    assert_allclose(new_info["reward_features"], old_info["reward_features"], atol=0.001)

    new_state, new_reward, new_done, new_info = new_env.step(action)
    old_state, old_reward, old_done, old_info = old_env.step(action)
    assert_allclose(new_state, old_state, atol=0.001)
    assert abs(new_reward - old_reward) < 0.001
    assert new_done == old_done
    assert_allclose(new_info["reward_features"], old_info["reward_features"], atol=0.001)


@settings(deadline=None)
@given(
    reward=arrays(
        dtype=float, shape=(4,), elements=floats(min_value=-1, max_value=1, allow_nan=False)
    ),
    action=arrays(
        dtype=float, shape=(2,), elements=floats(min_value=-2, max_value=2, allow_nan=False)
    ),
)
def test_determinism(reward: np.ndarray, action: np.ndarray):
    reward = safe_normalize(reward)
    env = gym.make("LegacyDriver-v1", reward=reward)

    first_init_state = env.reset()
    first_state, first_reward, first_done, first_info = env.step(action)

    second_init_state = env.reset()
    assert_array_equal(first_init_state, second_init_state)

    second_state, second_reward, second_done, second_info = env.step(action)
    assert_array_equal(first_state, second_state)
    assert first_reward == second_reward
    assert first_done == second_done
    assert_array_equal(first_info["reward_features"], second_info["reward_features"])
