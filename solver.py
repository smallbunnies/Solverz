from __future__ import annotations

import numpy as np

from eqn import Equations
from var import Vars
from numpy import abs, max, linalg


def inv(mat: np.ndarray):
    return linalg.inv(mat)


def nr_method(eqn: Equations,
              y: Vars,
              tol: float = 1e-8):
    df = eqn.g(y)
    while max(abs(df)) > tol:
        jacobian = eqn.j(y)
        y = y - inv(jacobian) * df
        df = eqn.g(y)
    return y
