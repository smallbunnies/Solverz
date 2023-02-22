from __future__ import annotations

import numpy as np
from numpy import abs, max, min, linalg, sum, sqrt
import tqdm

from Solverz.algebra import AliasVar, ComputeParam, F, X, Y
from .equations import AE, DAE
from .solverz_array import SolverzArray
from .variables import Vars, TimeVars
from .event import Event


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
    def f(y) -> SolverzArray:
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
        J0 = ode.j(x[i])
        J_inv = linalg.inv(np.eye(x.total_size) - dt * J0 * r) * dt
        k1 = J_inv @ ode.f(x[i])
        k2 = J_inv @ (ode.f(x[i] + a21 * k1) + J0 @ (r21 * k1))
        k3 = J_inv @ (ode.f(x[i] + a31 * k1 + a32 * k2) + J0 @ (r31 * k1 + r32 * k2))
        k4 = J_inv @ (ode.f(x[i] + a41 * k1 + a42 * k2 + a43 * k3) + J0 @ (r41 * k1 + r42 * k2 + r43 * k3))
        x[i + 1] = x[i] + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4
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
    block_I = np.zeros((dae.var_size, dae.var_size))
    block_I[np.ix_(range(dae.state_num), range(dae.state_num))] = block_I[np.ix_(range(dae.state_num),
                                                                                 range(dae.state_num))] + np.eye(
        dae.state_num)

    i = 0
    t = 0

    pbar = tqdm.tqdm(total=T)
    while abs(t - T) > dt / 10:
        if event:
            dae.update_param(event, t)
        J0 = dae.j(x[i], y[i])
        J_inv = linalg.inv(block_I - dt * J0 * r) * dt
        k1 = J_inv @ (np.concatenate((dae.f(x[i], y[i]), dae.g(x[i], y[i])), axis=0))
        k2 = J_inv @ (np.concatenate(
            (dae.f(x[i] + a21 * k1[0:dae.state_num], y[i] + a21 * k1[dae.state_num:dae.var_size]),
             dae.g(x[i] + a21 * k1[0:dae.state_num], y[i] + a21 * k1[dae.state_num:dae.var_size])), axis=0)
                      + J0 @ (r21 * k1))
        k3 = J_inv @ (np.concatenate((dae.f(x[i] + a31 * k1[0:dae.state_num] + a32 * k2[0:dae.state_num],
                                            y[i] + a31 * k1[dae.state_num:dae.var_size] + a32 * k2[dae.state_num:dae.var_size]),
                                      dae.g(x[i] + a31 * k1[0:dae.state_num] + a32 * k2[0:dae.state_num],
                                            y[i] + a31 * k1[dae.state_num:dae.var_size] + a32 * k2[dae.state_num:dae.var_size])), axis=0) +
                      J0 @ (r31 * k1 + r32 * k2))
        k4 = J_inv @ (
                np.concatenate(
                    (dae.f(x[i] + a41 * k1[0:dae.state_num] + a42 * k2[0:dae.state_num] + a43 * k3[0:dae.state_num],
                           y[i] + a41 * k1[dae.state_num:dae.var_size] + a42 * k2[dae.state_num:dae.var_size] + a43 * k3[dae.state_num:dae.var_size]),
                     dae.g(x[i] + a41 * k1[0:dae.state_num] + a42 * k2[0:dae.state_num] + a43 * k3[0:dae.state_num],
                           y[i] + a41 * k1[dae.state_num:dae.var_size] + a42 * k2[dae.state_num:dae.var_size] + a43 * k3[dae.state_num:dae.var_size])), axis=0) +
                J0 @ (r41 * k1 + r42 * k2 + r43 * k3))
        x[i + 1] = x[i] + b1 * k1[0:dae.state_num] + b2 * k2[0:dae.state_num] + b3 * k3[0:dae.state_num] + b4 * k4[0:dae.state_num]
        y[i + 1] = y[i] + b1 * k1[dae.state_num:dae.var_size] + b2 * k2[dae.state_num:dae.var_size] + b3 * k3[dae.state_num:dae.var_size] + b4 * k4[dae.state_num:dae.var_size]

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


def ode45():
    """
    Ode 45 with dense output
    :return:
    """
    pass
