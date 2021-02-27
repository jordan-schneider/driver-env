from typing import Callable, List

import numpy as np
import theano as th  # type: ignore
from theano.tensor import TensorVariable  # type: ignore

import driver.utils_driving as utils
from driver.dynamics import Dynamics


class Trajectory(object):
    def __init__(self, horizon: int, dyn: Dynamics):
        self.dyn = dyn
        self.horizon = horizon
        self.starting_state: TensorVariable = utils.vector(dyn.state_dim)
        self.action: List[TensorVariable] = [
            utils.vector(dyn.action_dim) for _ in range(self.horizon)
        ]
        self.state: List[TensorVariable] = []
        current_state = self.starting_state
        for t in range(horizon):
            current_state = dyn(current_state, self.action[t])
            self.state.append(current_state)
        self.next_state: Callable[[], TensorVariable] = th.function([], self.state[0])

    def tick(self) -> None:
        self.starting_state.set_value(self.next_state())
        for t in range(self.horizon - 1):
            self.action[t].set_value(self.action[t + 1].get_value())
        self.action[self.horizon - 1].set_value(np.zeros(self.dyn.action_dim))
