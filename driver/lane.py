import numpy as np
import theano.tensor as tt  # type: ignore

from driver import feature


class Lane:
    pass


class StraightLane(Lane):
    def __init__(self, p, q, w):
        # TODO(joschnei): Figure out what's going on here and rename everything.
        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w
        self.m = (self.q - self.p) / np.linalg.norm(self.q - self.p)
        self.n = np.asarray([-self.m[1], self.m[0]])

    def shifted(self, m):
        return StraightLane(self.p + self.n * self.w * m, self.q + self.n * self.w * m, self.w)

    def dist2(self, x):
        r = (x[0] - self.p[0]) * self.n[0] + (x[1] - self.p[1]) * self.n[1]
        return r * r

    def gaussian(self, width=0.5):
        @feature.feature
        def f(t, x, u):
            return tt.exp(-0.5 * self.dist2(x) / (width ** 2 * self.w * self.w / 4.0))

        return f
