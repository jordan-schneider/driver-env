from typing import List, Optional, Tuple

import numpy as np

from .simulator import DrivingSimulation


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

    def get_features(self, actions: Optional[np.ndarray] = None) -> List[float]:
        if actions is not None:
            self.set_ctrl(actions)
        recording = np.array(self.get_recording(all_info=False))

        # staying in lane (higher is better)
        staying_in_lane: float = (
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
            ).item()
            / 0.15343634
        )
        # keeping speed (lower is better)
        keeping_speed: float = np.mean(np.square(recording[:, 0, 3] - 1)).item() / 0.42202643

        # heading (higher is better)
        heading: float = np.mean(np.sin(recording[:, 0, 2])).item() / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance: float = (
            np.mean(
                np.exp(
                    -(
                        7 * np.square(recording[:, 0, 0] - recording[:, 1, 0])
                        + 3 * np.square(recording[:, 0, 1] - recording[:, 1, 1])
                    )
                )
            ).item()
            / 0.15258019
        )

        return [staying_in_lane, keeping_speed, heading, collision_avoidance]

    @property
    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.robot.state, self.human.state)

    def set_ctrl(self, value: np.ndarray) -> None:
        if len(value.shape) == 1:
            value = value.reshape((-1, 2))

        repeats = int(np.ceil(self.total_time / len(value)))
        self.ctrl = np.repeat(value, repeats, axis=0)[: self.total_time]

    def feed(self, value: np.ndarray) -> None:
        # I don't know why this alias is here but I'm keeping it.
        self.set_ctrl(value)
