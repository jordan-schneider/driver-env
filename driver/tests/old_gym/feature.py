from __future__ import annotations

from typing import Any, Callable

import theano.tensor as tt  # type: ignore


class Feature:
    def __init__(self, f: Callable):
        self.f = f

    def __call__(self, *args) -> Any:
        return self.f(*args)

    def __add__(self, r) -> Feature:
        return Feature(lambda *args: self(*args) + r(*args))

    def __radd__(self, r) -> Feature:
        return Feature(lambda *args: r(*args) + self(*args))

    def __mul__(self, r) -> Feature:
        return Feature(lambda *args: self(*args) * r)

    def __rmul__(self, r) -> Feature:
        return Feature(lambda *args: r * self(*args))

    def __pos__(self, r) -> Feature:
        return self

    def __neg__(self) -> Feature:
        return Feature(lambda *args: -self(*args))

    def __sub__(self, r) -> Feature:
        return Feature(lambda *args: self(*args) - r(*args))

    def __rsub__(self, r) -> Feature:
        return Feature(lambda *args: r(*args) - self(*args))


def feature(f: Callable) -> Feature:
    return Feature(f)


def speed(s=1.0) -> Feature:
    @feature
    def f(_, x, __):
        return -(x[3] - s) * (x[3] - s)

    return f


def control() -> Feature:
    @feature
    def f(_, __, u):
        return -u[0] ** 2 - u[1] ** 2

    return f


def bounded_control(bounds, width=0.05) -> Feature:
    @feature
    def f(_, __, u):
        for i, (a, b) in enumerate(bounds):
            return -tt.exp((u[i] - b) / width) - tt.exp((a - u[i]) / width)

    return f
