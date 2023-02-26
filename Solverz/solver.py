from __future__ import annotations

import numpy as np
import tqdm
from numpy import abs, max, min, linalg, sum, sqrt
from typing import Union, List

from .algebra import AliasVar, ComputeParam, F, X, Y
from .equations import AE, DAE
from .event import Event
from .var import TimeVar
from .variables import Vars, TimeVars


def inv(mat: np.ndarray):
    return linalg.inv(mat)


def nr_method(eqn: AE,
              y: Vars,
              tol: float = 1e-8):
    df = eqn.g(y)
    while max(abs(df)) > tol:
        y = y - inv(eqn.j(y)) @ df
        df = eqn.g(y)
    return y


def continuous_nr(eqn: AE,
                  y: Vars,
                  tol: float = 1e-8):
    def f(y) -> np.ndarray:
        return -inv(eqn.j(y)) @ eqn.g(y)

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
            err = sqrt(sum((err_ / sc) ** 2) / temp_y.total_size)
            if err < 1:
                y = temp_y
            dt = dt * min([fac_max, max([fac_min, fac * (1 / err) ** (1 / 4)])])
    return y


def implicit_trapezoid(dae: DAE,
                       x: TimeVars,
                       dt,
                       T,
                       event: Event = None):
    X0 = AliasVar(X, '0')
    Y0 = AliasVar(Y, '0')
    t = ComputeParam('t')
    t0 = ComputeParam('t0')
    Dt = ComputeParam('Dt')
    scheme = X - X0 - Dt / 2 * (F(X, Y, t) + F(X0, Y0, t0))
    ae = dae.discretize(scheme)

    i = 0
    t = 0

    xi1 = x[0]  # x_{i+1}
    xi0 = xi1.derive_alias('0')  # x_{i}

    pbar = tqdm.tqdm(total=T)
    while abs(t - T) > dt / 10:
        ae.update_param('Dt', dt)

        if event:
            ae.update_param(event, t)

        ae.update_param(xi0)
        ae.update_param('t', t + dt)
        ae.update_param('t0', t)

        xi1 = nr_method(ae, xi1)
        xi0.array[:] = xi1.array

        x[i + 1] = xi1
        pbar.update(dt)
        t = t + dt
        i = i + 1

    return x


def sirk_ode(ode: DAE,
             x: TimeVars,
             T,
             dt):
    r = 0.572816
    r21 = -2.34199281090742
    r31 = -0.0273333497228473
    r32 = 0.213811780334800
    r41 = -0.259083734468120
    r42 = -0.190595825778459
    r43 = -0.228030947223683
    a21 = 1.145632
    a31 = 0.520920769237066
    a32 = 0.134294177986617
    a41 = 0.520920769237066
    a42 = 0.134294177986617
    a43 = 0
    b1 = 0.324534692057929
    b2 = 0.0490865790926829
    b3 = 0
    b4 = 0.626378728849388

    # a21 = 1
    # a31 = 1
    # a32 = 0
    # r21 = (1 - 6 * r) / 2
    # r31 = (12 * r ** 2 - 6 * r - 1) / (3 * (1 - 2 * r))
    # r32 = (12 * r ** 2 - 12 * r + 5) / (6 * (1 - 2 * r))
    # b1 = 2 / 3
    # b2 = 1 / 6
    # b3 = 1 / 6

    for i in range(int(T / dt)):
        x0 = x[i]
        J0 = ode.j(x0)
        J_inv = linalg.inv(np.eye(x.total_size) - dt * J0 * r) * dt
        k1 = J_inv @ ode.f(x0)
        k2 = J_inv @ (ode.f(x0 + a21 * k1) + J0 @ (r21 * k1))
        k3 = J_inv @ (ode.f(x0 + a31 * k1 + a32 * k2) + J0 @ (r31 * k1 + r32 * k2))
        k4 = J_inv @ (ode.f(x0 + a41 * k1 + a42 * k2 + a43 * k3) + J0 @ (r41 * k1 + r42 * k2 + r43 * k3))
        x[i + 1] = x0 + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4
        # x[i + 1] = x[i] + b1 * k1 + b2 * k2 + b3 * k3

    return x


def sirk_dae(dae: DAE,
             x: TimeVars,
             y: TimeVars,
             T,
             dt,
             event: Event = None):
    r = 0.572816
    r21 = -2.34199281090742
    r31 = -0.0273333497228473
    r32 = 0.213811780334800
    r41 = -0.259083734468120
    r42 = -0.190595825778459
    r43 = -0.228030947223683
    a21 = 1.145632
    a31 = 0.520920769237066
    a32 = 0.134294177986617
    a41 = 0.520920769237066
    a42 = 0.134294177986617
    a43 = 0
    b1 = 0.324534692057929
    b2 = 0.0490865790926829
    b3 = 0
    b4 = 0.626378728849388

    # a21 = 1
    # a31 = 1
    # a32 = 0
    # r21 = (1 - 6 * r) / 2
    # r31 = (12 * r ** 2 - 6 * r - 1) / (3 * (1 - 2 * r))
    # r32 = (12 * r ** 2 - 12 * r + 5) / (6 * (1 - 2 * r))
    # b1 = 2 / 3
    # b2 = 1 / 6
    # b3 = 1 / 6

    dae.assign_eqn_var_address(x[0], y[0])
    n_v = dae.var_size
    n_s = dae.state_num
    block_I = np.zeros((n_v, n_v))
    block_I[np.ix_(range(n_s), range(n_s))] = block_I[np.ix_(range(n_s), range(n_s))] + np.eye(n_s)

    i = 0
    t = 0

    pbar = tqdm.tqdm(total=T)
    while abs(t - T) > dt / 10:
        if event:
            dae.update_param(event, t)
        x0 = x[i]
        y0 = y[i]
        J0 = dae.j(x0, y0)
        J_inv = linalg.inv(block_I - dt * J0 * r) * dt
        k1 = J_inv @ (np.concatenate((dae.f(x0, y0), dae.g(x0, y0)), axis=0))
        k2 = J_inv @ (np.concatenate(
            (dae.f(x0 + a21 * k1[0:n_s], y0 + a21 * k1[n_s:n_v]),
             dae.g(x0 + a21 * k1[0:n_s], y0 + a21 * k1[n_s:n_v])), axis=0)
                      + J0 @ (r21 * k1))
        k3 = J_inv @ (np.concatenate((dae.f(x0 + a31 * k1[0:n_s] + a32 * k2[0:n_s],
                                            y0 + a31 * k1[n_s:n_v] + a32 * k2[n_s:n_v]),
                                      dae.g(x0 + a31 * k1[0:n_s] + a32 * k2[0:n_s],
                                            y0 + a31 * k1[n_s:n_v] + a32 * k2[n_s:n_v])), axis=0) +
                      J0 @ (r31 * k1 + r32 * k2))
        k4 = J_inv @ (
                np.concatenate(
                    (dae.f(x0 + a41 * k1[0:n_s] + a42 * k2[0:n_s] + a43 * k3[0:n_s],
                           y0 + a41 * k1[n_s:n_v] + a42 * k2[n_s:n_v] + a43 * k3[n_s:n_v]),
                     dae.g(x0 + a41 * k1[0:n_s] + a42 * k2[0:n_s] + a43 * k3[0:n_s],
                           y0 + a41 * k1[n_s:n_v] + a42 * k2[n_s:n_v] + a43 * k3[n_s:n_v])), axis=0) +
                J0 @ (r41 * k1 + r42 * k2 + r43 * k3))
        x[i + 1] = x0 + b1 * k1[0:n_s] + b2 * k2[0:n_s] + b3 * k3[0:n_s] + b4 * k4[0:n_s]
        y[i + 1] = y0 + b1 * k1[n_s:n_v] + b2 * k2[n_s:n_v] + b3 * k3[n_s:n_v] + b4 * k4[n_s:n_v]

        pbar.update(dt)
        t = t + dt
        i = i + 1
    return x, y


def euler(ode: DAE,
          x: TimeVars,
          T,
          dt):
    for i in range(int(T / dt)):
        x[i + 1] = x[i] + dt * ode.f(x[i])

    return x


def improved_euler(ode: DAE,
                   x: TimeVars,
                   T,
                   dt):
    for i in range(int(T / dt)):
        x_pre = x[i] + dt * ode.f(x[i])
        x[i + 1] = x[i] + dt / 2 * (ode.f(x[i]) + ode.f(x_pre))

    return x


def ode45(ode: DAE,
          y: TimeVars,
          tspan: Union[List, np.ndarray],
          dt=None,
          atol=1e-6,
          rtol=1e-3,
          event: Event = None,
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
    hmax = np.abs(T)/10
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
            err = dt * linalg.norm(kE / np.maximum(np.maximum(abs(y0), abs(ynew)).reshape(-1,), threshold), np.Inf)

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


def ode15s(ode: DAE,
           y: TimeVars,
           dt,
           T,
           event: Event = None):
    pass
