from typing import Any, List, Optional, Tuple, Union

import numpy as np

from driver import car, dynamics, lane
from driver.world import World


class Simulation(object):
    input_size: int

    def __init__(self, name, total_time: int = 1000, recording_time: Tuple[int, int] = (0, 1000)):
        self.name = name.lower()
        self.total_time = total_time
        self.recording_time = (max(0, recording_time[0]), min(total_time, recording_time[1]))
        self.frame_delay_ms = 0

    def reset(self) -> None:
        self.trajectory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.alreadyRun = False
        self.ctrl_array = np.zeros((self.total_time, self.input_size))

    @property
    def ctrl(self) -> np.ndarray:
        return self.ctrl_array

    @ctrl.setter
    def ctrl(self, value: np.ndarray) -> None:
        self.reset()
        self.ctrl_array = value.copy()
        self.run(reset=False)

    def get_features(self) -> Any:
        raise NotImplementedError

    def feed(self, value: np.ndarray) -> None:
        raise NotImplementedError

    def run(self, reset: bool) -> None:
        raise NotImplementedError


class DrivingSimulation(Simulation):
    def __init__(self, name, total_time: int = 50, recording_time: Tuple[int, int] = (0, 50)):
        super(DrivingSimulation, self).__init__(
            name, total_time=total_time, recording_time=recording_time
        )
        self.world = World()
        clane = lane.StraightLane([0.0, -1.0], [0.0, 1.0], 0.17)
        self.world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
        self.world.fences += [clane.shifted(2), clane.shifted(-2)]
        self.dyn = dynamics.CarDynamics(0.1)
        self.robot = car.Car(self.dyn, [0.0, -0.3, np.pi / 2.0, 0.4], color="orange")
        self.human = car.Car(self.dyn, [0.17, 0.0, np.pi / 2.0, 0.41], color="white")
        self.world.cars.append(self.robot)
        self.world.cars.append(self.human)
        self.initial_state = (self.robot.state, self.human.state)
        self.input_size = 2
        self.reset()
        # I can't type this visualizer without importing it, and I can't import it on a headless
        # server
        self.viewer: Optional[Any] = None

    def initialize_positions(self) -> None:
        self.robot_history_state: List[np.ndarray] = []
        self.robot_history_action: List[np.ndarray] = []
        self.human_history_state: List[np.ndarray] = []
        self.human_history_action: List[np.ndarray] = []
        self.robot.state = self.initial_state[0]
        self.human.state = self.initial_state[1]

    def reset(self) -> None:
        super(DrivingSimulation, self).reset()
        self.initialize_positions()

    def run(self, reset: bool = False) -> None:
        if reset:
            self.reset()
        else:
            self.initialize_positions()
        for i in range(self.total_time):
            self.robot.action = self.ctrl_array[i]
            if i < self.total_time // 5:
                self.human.action = np.array([0, self.initial_state[1][3]])
            elif i < 2 * self.total_time // 5:
                self.human.action = np.array([1.0, self.initial_state[1][3]])
            elif i < 3 * self.total_time // 5:
                self.human.action = np.array([-1.0, self.initial_state[1][3]])
            elif i < 4 * self.total_time // 5:
                self.human.action = np.array([0, self.initial_state[1][3] * 1.3])
            else:
                self.human.action = np.array([0, self.initial_state[1][3] * 1.3])
            self.robot_history_state.append(self.robot.state)
            self.robot_history_action.append(self.robot.action)
            self.human_history_state.append(self.human.state)
            self.human_history_action.append(self.human.action)
            self.robot.move()
            self.human.move()
            self.trajectory.append((self.robot.state, self.human.state))
        self.alreadyRun = True

    # I keep all_info variable for the compatibility with mujoco wrapper
    def get_trajectory(self, all_info: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not self.alreadyRun:
            self.run()
        return self.trajectory.copy()

    def get_recording(self, all_info: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        traj = self.get_trajectory(all_info=all_info)
        return traj[self.recording_time[0] : self.recording_time[1]]

    def watch(self, repeat_count: int = 1) -> None:
        self.robot.state = self.initial_state[0]
        self.human.state = self.initial_state[1]
        if self.viewer is None:
            from driver import visualize

            self.viewer = visualize.Visualizer(0.1, magnify=1.2)
            self.viewer.main_car = self.robot
            self.viewer.use_world(self.world)
            self.viewer.paused = True
        for _ in range(repeat_count):
            self.viewer.run_modified(
                history_x=[self.robot_history_state, self.human_history_state],
                history_u=[self.robot_history_action, self.human_history_action],
            )
        self.viewer.window.close()
        self.viewer = None
