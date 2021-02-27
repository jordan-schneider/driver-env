from typing import List, Tuple

import numpy as np

from driver.simulator import DrivingSimulation


class Driver(DrivingSimulation):
    def __init__(self, total_time: int = 50, recording_time: Tuple[int, int] = (0, 50)):
        super(Driver, self).__init__(
            name="driver", total_time=total_time, recording_time=recording_time
        )
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds: List[Tuple[int, int]] = [(-1, 1)] * self.ctrl_size
        self.state_bounds: List[Tuple[int, int]] = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self) -> List[float]:
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # staying in lane (higher is better)
        staying_in_lane = (
            np.mean(
                np.exp(
                    -30
                    * np.min(
                        [
                            np.square(recording[:, 0, 0] - 0.17),
                            np.square(recording[:, 0, 0]),
                            np.square(recording[:, 0, 0] + 0.17),
                        ],
                        axis=0,
                    )
                )
            )
            / 0.15343634
        )

        # keeping speed (lower is better)
        keeping_speed = np.mean(np.square(recording[:, 0, 3] - 1)) / 0.42202643

        # heading (higher is better)
        heading = np.mean(np.sin(recording[:, 0, 2])) / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance = (
            np.mean(
                np.exp(
                    -(
                        7 * np.square(recording[:, 0, 0] - recording[:, 1, 0])
                        + 3 * np.square(recording[:, 0, 1] - recording[:, 1, 1])
                    )
                )
            )
            / 0.15258019
        )

        return [staying_in_lane, keeping_speed, heading, collision_avoidance]

    @property
    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.robot.state, self.human.state)

    @state.setter
    def state(self, value: np.ndarray) -> None:
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value: np.ndarray) -> None:
        arr = [[0] * self.input_size] * self.total_time
        interval_count = len(value) // self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i * interval_time : (i + 1) * interval_time] = [value[j], value[j + 1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value: np.ndarray) -> None:
        # I don't know why this alias is here but I'm keeping it.
        self.set_ctrl(value)
