import driver.gym
import gym  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
from driver.gym.legacy_env import LegacyEnv
from driver.legacy.gym_driver import GymDriver
from driver.math_utils import safe_normalize
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from numpy.testing import assert_allclose, assert_array_equal

ATOL = 1.0


@given(
    state=arrays(
        dtype=np.float32,
        shape=(2, 4),
        elements=floats(min_value=-10, max_value=10, allow_nan=False, width=32),
    )
)
def test_features(state: np.ndarray):
    new_env = LegacyEnv(np.zeros(4))

    new_features = new_env.main_car.features(tf.constant(state), None).numpy()
    old_features = GymDriver.get_features(state)

    assert_allclose(new_features, old_features, atol=ATOL)


@settings(deadline=None)
@given(
    reward=arrays(
        dtype=np.float32,
        shape=(4,),
        elements=floats(min_value=-1, max_value=1, allow_nan=False, width=32),
    ),
    action=arrays(
        dtype=np.float32,
        shape=(2,),
        elements=floats(min_value=-2, max_value=2, allow_nan=False, width=32),
    ),
)
def test_consistency(reward: np.ndarray, action: np.ndarray):
    reward = safe_normalize(reward)
    new_env = LegacyEnv(reward=reward)
    old_env = GymDriver(reward=reward, horizon=50)

    new_state = new_env.reset()
    old_state = old_env.reset()
    assert new_state.shape == old_state.shape
    assert_allclose(new_state, old_state)

    new_state, new_reward, new_done, new_info = new_env.step(action)
    old_state, old_reward, old_done, old_info = old_env.step(action)
    assert_allclose(new_state, old_state, atol=ATOL)
    assert abs(new_reward - old_reward) < ATOL
    assert new_done == old_done
    assert_allclose(new_info["reward_features"], old_info["reward_features"], atol=ATOL)

    new_state, new_reward, new_done, new_info = new_env.step(action)
    old_state, old_reward, old_done, old_info = old_env.step(action)
    assert_allclose(new_state, old_state, atol=ATOL)
    assert abs(new_reward - old_reward) < ATOL
    assert new_done == old_done
    assert_allclose(new_info["reward_features"], old_info["reward_features"], atol=ATOL)


@settings(deadline=None)
@given(
    reward=arrays(
        dtype=np.float32,
        shape=(4,),
        elements=floats(min_value=-1, max_value=1, allow_nan=False, width=32),
    ),
    action=arrays(
        dtype=np.float32,
        shape=(2,),
        elements=floats(min_value=-2, max_value=2, allow_nan=False, width=32),
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
