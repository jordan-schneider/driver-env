from time import perf_counter

import numpy as np

from driver.gym_env.legacy_env import LegacyEnv
from driver.legacy.models import Driver


def make_actions(n_trajs: int) -> np.ndarray:
    inputs = np.random.uniform(
        low=-1,
        high=1,
        size=(n_trajs, 50, 2),
    )
    return inputs


def main() -> None:
    reward_weights = np.ones(4)
    sim = Driver()
    env = LegacyEnv(reward_weights)

    plans = make_actions(1000)

    returns = []
    start = perf_counter()
    for plan in plans:
        sim.feed(plan)
        features = sim.get_features()
        returns.append(reward_weights @ features)
    stop = perf_counter()
    print(f"Legacy env took {(stop - start) / len(plans)} seconds on average")
    # Driver env is a lot faster for rollouts

    returns = []
    start = perf_counter()
    for plan in plans:
        env.reset()
        plan_return = 0.0
        for action in plan:
            _, reward, _, _ = env.step(action)
            plan_return += reward
        returns.append(plan_return)
    stop = perf_counter()
    print(f"tf env took {(stop - start) / len(plans)} seconds on average")


if __name__ == "__main__":
    main()
