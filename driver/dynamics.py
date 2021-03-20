from typing import Callable, Optional

import numpy as np
import theano.tensor as tt  # type: ignore


class Dynamics:
    def __init__(self, state_dim: int, action_dim: int, f: Callable, dt: Optional[float] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = dt
        if dt is None:
            self.f = f
        else:
            self.f = lambda state, action: state + dt * f(state, action)

    def __call__(self, state, action):
        return self.f(state, action)


class CarDynamics(Dynamics):
    def __init__(self, dt: float = 0.1, friction: float = 1.0):
        def f(state: np.ndarray, action: np.ndarray) -> tt.Tensor:
            return tt.stacklists(
                [
                    state[3] * tt.cos(state[2]),
                    state[3] * tt.sin(state[2]),
                    state[3] * action[0],
                    action[1] - state[3] * friction,
                ]
            )

        super(CarDynamics, self).__init__(state_dim=4, action_dim=2, f=f, dt=dt)
