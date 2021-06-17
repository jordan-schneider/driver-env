import pickle as pkl
from pathlib import Path

import cv2
import numpy as np

from world import TwoTrajectoryWorld

# from array2gif import write_gif

DATA_DIR = Path("/home/joschnei/value-alignment-verification/data/human/active/")


def main():
    render_question()


def render_question() -> None:
    inputs = np.load(DATA_DIR / "elicitation/inputs.npy")
    indices = pkl.load(open(DATA_DIR / "point_reward_test/indices.skip_noise.pkl", "rb",))
    prefereces = np.load(DATA_DIR / "elicitation/preferences.npy")

    index = indices[(2.0, 0.05, 100)][0]
    plans = inputs[index]
    preference = prefereces[index]
    if preference == 1.0:
        good_plan, bad_plan = plans
    else:
        bad_plan, good_plan = plans

    good_plan = np.repeat(good_plan.reshape(5, 2), 10, axis=0)
    bad_plan = np.repeat(bad_plan.reshape(5, 2), 10, axis=0)

    render_plans("question.mp4", good_plan, bad_plan)


def render_plans(
    outpath: Path, good_plan: np.ndarray, bad_plan: np.ndarray, fps: int = 5,
):
    assert good_plan.shape == (50, 2), f"good_plan shape={good_plan.shape} is not (50, 2)"
    assert bad_plan.shape == (50, 2), f"bad_plan shape={bad_plan.shape} is not (50, 2)"

    world = TwoTrajectoryWorld(dt=0.1, good_plan=good_plan, bad_plan=bad_plan)
    world.reset()
    frames = []
    for _ in range(49):
        frames.append(world.render(mode="rgb_array"))
        world.step()
    frames.append(world.render(mode="rgb_array"))

    size = 600, 600
    out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


if __name__ == "__main__":
    main()
