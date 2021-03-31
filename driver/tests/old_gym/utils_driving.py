from typing import Any, Callable, Sequence, TypeVar, cast

import numpy as np
import theano as th  # type: ignore
import theano.tensor as tt  # type: ignore
from theano.tensor import TensorVariable

T = TypeVar("T")


def extract(var: T) -> Callable[[], T]:
    return th.function([], var, mode=th.compile.Mode(linker="py"))()


def shape(var: Any) -> Callable[[], tuple]:
    return extract(var.shape)


def vector(n: int) -> TensorVariable:
    return th.shared(np.zeros(n))


def matrix(n: int, m: int) -> TensorVariable:
    return tt.shared(np.zeros((n, m)))


def grad(f: Callable, x: Any, constants: list = []) -> TensorVariable:
    ret = th.gradient.grad(f, x, consider_constant=constants, disconnected_inputs="warn")
    if isinstance(ret, list):
        ret = tt.concatenate(ret)
    return ret


def jacobian(f: Sequence[Callable], x: Any, constants: list = []) -> TensorVariable:
    sz = cast(int, shape(f))  # Theano is doing some implicit casting black magic here
    return tt.stacklists([grad(f[i], x) for i in range(sz)])


def hessian(f: Callable, x: Any, constants: list = []) -> TensorVariable:
    return jacobian(grad(f, x, constants=constants), x, constants=constants)
