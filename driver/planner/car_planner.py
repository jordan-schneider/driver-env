"""Base class for planners."""
from __future__ import annotations

from typing import TYPE_CHECKING

from driver.car.car import Car

if TYPE_CHECKING:
    from driver.world import CarWorld


class CarPlanner:
    """
    Parent class for all the trajectory finders for a car.
    """

    def __init__(self, world: CarWorld, car: Car):
        self.world = world
        self.car = car

    def generate_plan(self):
        raise NotImplementedError


class CoordinateAscentPlanner(CarPlanner):
    """
    CarPlanner that performs coordinate ascent to find an approximate Nash
    equilibrium trajectory.
    """
