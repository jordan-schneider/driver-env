from typing import List, Tuple

import numpy as np

from driver.dynamics import Dynamics
from driver.trajectory import Trajectory


class Car:
    def __init__(
        self, dyn: Dynamics, starting_state: np.ndarray, color: str = "yellow", horizon: int = 5
    ):
        self.starting_data = {"starting_state": starting_state}
        self.bounds: List[Tuple[float, float]] = [(-1.0, 1.0), (-1.0, 1.0)]
        self.horizon = horizon
        self.dyn = dyn
        self.traj = Trajectory(horizon, dyn)
        self.traj.starting_state.set_value(starting_state)
        self.linear = Trajectory(horizon, dyn)
        self.linear.starting_state.set_value(starting_state)
        self.color = color
        self.default_action = np.zeros(self.dyn.action_dim)

    def reset(self) -> None:
        self.traj.starting_state.set_value(self.starting_data["starting_state"])
        self.linear.starting_state.set_value(self.starting_data["starting_state"])
        for t in range(self.horizon):
            self.traj.action[t].set_value(np.zeros(self.dyn.action_dim))
            self.linear.action[t].set_value(self.default_action)

    def move(self) -> None:
        self.traj.tick()
        self.linear.starting_state.set_value(self.traj.starting_state.get_value())

    @property
    def state(self) -> np.ndarray:
        return self.traj.starting_state.get_value()

    @state.setter
    def state(self, value) -> None:
        self.traj.starting_state.set_value(value)

    @property
    def action(self) -> np.ndarray:
        return self.traj.action[0].get_value()

    @action.setter
    def action(self, value) -> None:
        self.traj.action[0].set_value(value)
