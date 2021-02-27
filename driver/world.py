import math
import shelve
from typing import List

import numpy as np
import theano as th  # type: ignore
import theano.tensor as tt  # type: ignore

from driver import car, dynamics, feature, lane
from driver.car import Car
from driver.lane import Lane

th.config.optimizer_verbose = False
th.config.allow_gc = False
th.config.optimizer = "fast_compile"


class Object:
    def __init__(self, name: str, state):
        self.name = name
        self.state = np.asarray(state)


class World:
    def __init__(self):
        self.cars: List[Car] = []
        self.lanes: List[Lane] = []
        self.fences: List[Lane] = []

    def simple_reward(self, trajs=None, lanes=None, fences=None, speed=1.0, speed_import=1.0):
        if lanes is None:
            lanes = self.lanes
        if fences is None:
            fences = self.fences
        if trajs is None:
            trajs = [c.linear for c in self.cars]
        elif isinstance(trajs, car.Car):
            trajs = [c.linear for c in self.cars if c != trajs]
        r = 0.1 * feature.control()
        theta = [1.0, -50.0, 10.0, 10.0, -60.0]  # Simple model
        for lane in lanes:
            r = r + theta[0] * lane.gaussian()
        for fence in fences:
            r = r + theta[1] * fence.gaussian()
        if speed is not None:
            r = r + speed_import * theta[3] * feature.speed(speed)
        for traj in trajs:
            r = r + theta[4] * traj.gaussian()
        return r


def playground():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0.0, -1.0], [0.0, 1.0], 0.17)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    c = car.Car(dyn, [0.1, -0.25, math.pi / 3.0, 0.0], color="red", horizon=100)
    world.cars.append(c)
    return world


def world_features(num=0):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0.0, -1.0], [0.0, 1.0], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.Car(dyn, [0.0, 0.1, math.pi / 2.0 + math.pi / 5, 0.0], color="yellow"))
    world.cars.append(car.Car(dyn, [-0.13, 0.2, math.pi / 2.0 - math.pi / 5, 0.0], color="yellow"))
    world.cars.append(car.Car(dyn, [0.13, -0.2, math.pi / 2.0, 0.0], color="yellow"))
    return world


if __name__ == "__main__":
    from driver import visualize

    world = playground()
    vis = visualize.Visualizer(0.1, magnify=1.2)
    vis.main_car = None
    vis.use_world(world)
    vis.paused = True

    @feature.feature
    def zero(t, x, u):
        return 0.0

    r = zero
    vis.set_heat(r)
    vis.run()
