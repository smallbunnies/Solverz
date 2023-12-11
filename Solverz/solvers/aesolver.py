from __future__ import annotations

import numpy as np
from numpy import abs, max, min, sum, sqrt
from cvxopt import matrix

from Solverz.equation.equations import AE
from Solverz.numerical_interface.custom_function import solve
from Solverz.variable.variables import Vars


def nr_method(eqn: AE,
              y: Vars,
              tol: float = 1e-8,
              stats=False):
    df = eqn.g(y)
    ite = 0
    while max(abs(df)) > tol:
        ite = ite + 1
        y = y - solve(eqn.j(y), df)
        df = eqn.g(y)
        if ite >= 100:
            print(f"Cannot converge within 100 iterations. Deviation: {max(abs(df))}!")
            break
    if not stats:
        return y
    else:
        return y, ite


def continuous_nr(eqn: AE,
                  y: Vars,
                  tol: float = 1e-8,
                  dt=1,
                  hmax=1.7):
    def f(y) -> np.ndarray:
        return -solve(eqn.j(y), eqn.g(y))

    Pow = 1 / 5
    # c2, c3, c4, c5 = 1 / 5, 3 / 10, 4 / 5, 8 / 9 for non-autonomous equations
    a11, a21, a31, a41, a51, a61 = 1 / 5, 3 / 40, 44 / 45, 19372 / 6561, 9017 / 3168, 35 / 384
    a22, a32, a42, a52 = 9 / 40, -56 / 15, -25360 / 2187, -355 / 33
    a33, a43, a53, a63 = 32 / 9, 64448 / 6561, 46732 / 5247, 500 / 1113
    a44, a54, a64 = -212 / 729, 49 / 176, 125 / 192
    a55, a65 = -5103 / 18656, -2187 / 6784
    a66 = 11 / 84
    e1, e3, e4, e5, e6, e7 = 71 / 57600, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40

    dt = dt
    atol = 1e-6
    rtol = 1e-1
    threshold = atol / rtol
    hmax = hmax
    hmin = 16 * np.spacing(0)
    # atol and rtol can not be too small
    ite = 0
    df = eqn.g(y)
    while max(abs(df)) > tol:
        ite = ite + 1
        err = 2
        nofailed = True
        while err > rtol:
            k1 = f(y)
            k2 = f(y + dt * a11 * k1)
            k3 = f(y + dt * (a21 * k1 + a22 * k2))
            k4 = f(y + dt * (a31 * k1 + a32 * k2 + a33 * k3))
            k5 = f(y + dt * (a41 * k1 + a42 * k2 + a43 * k3 + a44 * k4))
            k6 = f(y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4 + a55 * k5))
            ynew = y + dt * (a61 * k1 + a63 * k3 + a64 * k4 + a65 * k5 + a66 * k6)  # y6 has nothing to do with k7
            k7 = f(ynew)
            kE = k1 * e1 + k3 * e3 + k4 * e4 + k5 * e5 + k6 * e6 + k7 * e7

            # error control
            # error estimation
            err = dt * np.linalg.norm(
                kE.reshape(-1, ) / np.maximum(np.maximum(abs(y.array), abs(ynew.array)).reshape(-1, ), threshold),
                np.Inf)
            if err > rtol:  # failed step
                if dt <= hmin:
                    raise ValueError(f'IntegrationTolNotMet step size: {dt} hmin: {hmin}')
                if nofailed:  # There haven't been failed attempts
                    nofailed = False
                    dt = np.max([hmin, dt * np.max([0.1, 0.8 * (rtol / err) ** Pow])])
                else:
                    dt = np.max([hmin, 0.5 * dt])
            else:  # Successful step
                break

        y = ynew
        df = eqn.g(y)

        if nofailed:  # Enlarge step size if no failure is met
            temp = 1.25 * (err / rtol) ** Pow
            if temp > 0.2:
                dt = dt / temp
            else:
                dt = 5 * dt

        dt = np.min([dt, hmax])

        if ite > 100:
            break
    return y, ite
