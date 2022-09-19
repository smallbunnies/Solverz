from __future__ import annotations

from eqn import Equations
from var import Vars
from numpy import abs, max, linalg


def nr_method(eqn: Equations,
              y: Vars,
              tol: float = 1e-8):
    df = eqn.g(y)
    while max(abs(df)) > tol:
        jacobian = eqn.j(y)
        y = y + linalg.inv(jacobian) * df
        df = eqn.g(y)
    return y
