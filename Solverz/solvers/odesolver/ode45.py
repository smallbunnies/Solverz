from __future__ import annotations

from typing import Union, List

import numpy as np
from numpy import abs, linalg

from Solverz.equation.equations import DAE
from Solverz.num_api.num_eqn import nDAE, nAE
from Solverz.solvers.laesolver import lu_decomposition
from Solverz.solvers.nlaesolver import nr_method
from Solverz.solvers.option import Opt
from Solverz.solvers.parser import dae_io_parser
from Solverz.solvers.stats import Stats
from Solverz.variable.variables import TimeVars


def ode45(ode: DAE,
          y: TimeVars,
          tspan: Union[List, np.ndarray],
          dt=None,
          atol=1e-6,
          rtol=1e-3,
          output_k=False):
    """
    Ode45 with dense output
    :return:
    """

    Pow = 1 / 5
    # c2, c3, c4, c5 = 1 / 5, 3 / 10, 4 / 5, 8 / 9 for non-autonomous equations
    a11, a21, a31, a41, a51, a61 = 1 / 5, 3 / 40, 44 / 45, 19372 / 6561, 9017 / 3168, 35 / 384
    a22, a32, a42, a52 = 9 / 40, -56 / 15, -25360 / 2187, -355 / 33
    a33, a43, a53, a63 = 32 / 9, 64448 / 6561, 46732 / 5247, 500 / 1113
    a44, a54, a64 = -212 / 729, 49 / 176, 125 / 192
    a55, a65 = -5103 / 18656, -2187 / 6784
    a66 = 11 / 84
    e1, e3, e4, e5, e6, e7 = 71 / 57600, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40

    tspan = np.array(tspan)
    T = tspan[-1]
    hmax = np.abs(T) / 10
    i = 0
    t = 0
    tout = TimeVar('tout', length=100)
    hmin = 16 * np.spacing(t)
    threshold = atol / rtol
    y0 = y[0]
    if output_k:
        kout = np.zeros((100, ode.f(y0).shape[0], 7))

    # Compute an initial step size using f(y)
    if dt is None:
        dt = hmax
    dt_ = linalg.norm(ode.f(y0) / np.maximum(np.abs(y0), threshold), np.Inf) / (0.8 * rtol ** Pow)
    if dt * dt_ > 1:
        dt = 1 / dt_
    dt = np.maximum(dt, hmin)

    done = False
    while not done:
        hmin = 16 * np.spacing(dt)
        dt = np.min([hmax, np.max([hmin, dt])])

        # Stretch the step if within 10% of T-t.
        if 1.1 * dt >= np.abs(T - t):
            dt = T - t
            done = True

        nofailed = True
        while True:
            k1 = ode.f(y0)
            k2 = ode.f(y0 + dt * a11 * k1)
            k3 = ode.f(y0 + dt * (a21 * k1 + a22 * k2))
            k4 = ode.f(y0 + dt * (a31 * k1 + a32 * k2 + a33 * k3))
            k5 = ode.f(y0 + dt * (a41 * k1 + a42 * k2 + a43 * k3 + a44 * k4))
            k6 = ode.f(y0 + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4 + a55 * k5))
            ynew = y0 + dt * (a61 * k1 + a63 * k3 + a64 * k4 + a65 * k5 + a66 * k6)  # y6 has nothing to do with k7
            k7 = ode.f(ynew)
            kE = k1 * e1 + k3 * e3 + k4 * e4 + k5 * e5 + k6 * e6 + k7 * e7
            tnew = t + dt
            if done:
                tnew = T
            dt = tnew - t  # purify dt
            # error control
            # error estimation
            err = dt * linalg.norm(kE / np.maximum(np.maximum(abs(y0), abs(ynew)).reshape(-1, ), threshold), np.Inf)

            if err > rtol:  # failed step
                if dt <= hmin:
                    raise ValueError(f'IntegrationTolNotMet step size: {dt} hmin: {hmin}')
                if nofailed:  # There haven't been failed attempts
                    nofailed = False
                    dt = np.max([hmin, dt * np.max([0.1, 0.8 * (rtol / err) ** Pow])])
                else:
                    dt = np.max([hmin, 0.5 * dt])
                done = False
            else:  # Successful step
                break
        i = i + 1

        # TODO: 3. Event

        # Output
        y[i] = ynew
        tout[i] = tnew
        if output_k:
            kout[i - 1, :, :] = np.array([k1, k2, k3, k4, k5, k6, k7]).T

        if done:
            break
        if nofailed:  # Enlarge step size if no failure is met
            temp = 1.25 * (err / rtol) ** Pow
            if temp > 0.2:
                dt = dt / temp
            else:
                dt = 5 * dt

        t = tnew
        y0 = ynew

    tout = tout[0:i + 1]
    y = y[0:i + 1]
    if output_k:
        kout = kout[0:i]
        return tout, y, kout
    else:
        return tout, y


def nrtp45(tinterp: np.ndarray, t: float, y: np.ndarray, dt: float, k: np.ndarray) -> np.ndarray:
    """
    Dense output of ODE45 with continuous Runge-Kutta method
    :return:
    """
    # Define the BI matrix
    BI = np.array([
        [1, - 183 / 64, 37 / 12, - 145 / 128],
        [0, 0, 0, 0],
        [0, 1500 / 371, - 1000 / 159, 1000 / 371],
        [0, - 125 / 32, 125 / 12, - 375 / 64],
        [0, 9477 / 3392, - 729 / 106, 25515 / 6784],
        [0, - 11 / 7, 11 / 3, - 55 / 28],
        [0, 3 / 2, - 4, 5 / 2]
    ])
    s = (tinterp - t) / dt
    yinterp = np.repeat(y.reshape(-1, 1), tinterp.shape[0], axis=1) + k @ (dt * BI) @ np.cumprod([s, s, s, s], axis=0)
    return yinterp
