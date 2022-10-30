from __future__ import annotations

import numpy as np
from numpy import abs, max, min, linalg, sum, sqrt
import sympy as sp

from .equations import AE, DAE
from .variables import Vars, TimeVars
from .var import Var
from .solverz_array import SolverzArray


def inv(mat: np.ndarray):
    return linalg.inv(mat)


def nr_method(eqn: AE,
              y: Vars,
              tol: float = 1e-8):
    df = eqn.g(y)
    while max(abs(df)) > tol:
        y = y - inv(eqn.j(y)) * df
        df = eqn.g(y)
    return y


def continuous_nr(eqn: AE,
                  y: Vars,
                  tol: float = 1e-8):
    def f(y) -> SolverzArray:
        return -inv(eqn.j(y)) * eqn.g(y)

    dt = 1
    atol = 1e-3
    rtol = 1e-3
    fac = 0.99
    fac_max = 1.015
    fac_min = 0.985
    # atol and rtol can not be too small
    ite = 0
    while max(abs(eqn.g(y))) > tol:
        ite = ite + 1
        # TODO: output iteration numbers
        err = 2
        while err > 1:
            k1 = f(y)
            k2 = f(y + 0.5 * dt * k1)
            k3 = f(y + 0.5 * dt * k2)
            k4 = f(y + dt * k3)
            k5 = f(y + 5 / 32 * k1 + 7 / 32 * k2 + 13 / 32 * k3 - 1 / 32 * k4)
            temp_y = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            # step size control
            err_ = (1 / 6 + 1 / 2) * k1 + (1 / 3 - 7 / 3) * k2 + (1 / 3 - 7 / 3) * k3 + (1 / 6 - 13 / 6) * k4 + (
                    0 + 16 / 3) * k5
            y_ = temp_y - err_
            sc = atol + max(np.concatenate([abs(temp_y.array), abs(y_.array)], axis=1), axis=1) * rtol
            err = sqrt(sum((err_.array.reshape(-1, ) / sc) ** 2) / temp_y.total_size)
            if err < 1:
                y = temp_y
            dt = dt * min([fac_max, max([fac_min, fac * (1 / err) ** (1 / 4)])])
    return y


def implicit_trapezoid(dae: DAE,
                       x: TimeVars,
                       y: TimeVars = None,
                       k=100):
    xi1 = x[0]  # x_{i+1}
    xi0 = x[0]  # x_{i}
    yi1 = y[0]  # y_{i+1}
    yi0 = y[0]  # y_{i}

    ae = dae.discretize('x-x0-dt/2*(f(x0,t0)+f(x,t))')

    i = -1
    while i < k:
        i = i + 1
        [xi1, yi1] = nr_method(ae, [xi1, yi1])
        ae.xi0 = xi1  # Update xi0
        ae.yi0 = yi1  # Update yi0
        x[i] = xi1
        y[i] = yi1
