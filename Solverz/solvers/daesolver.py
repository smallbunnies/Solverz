from __future__ import annotations

from typing import Union, List

import numpy as np
import tqdm
from numpy import abs, linalg
from scipy.sparse import csc_array, linalg as sla

from Solverz.equation.equations import DAE
from Solverz.event import Event
from Solverz.num.num_alg import AliasVar, X, Y, ComputeParam, F, Var
from Solverz.solvers.aesolver import nr_method
from Solverz.variable.variables import TimeVars, Vars, as_Vars, combine_Vars


# from Solverz.sas.sas_num import DTcache, DTvar, DTeqn


def implicit_trapezoid(dae: DAE,
                       x: TimeVars,
                       tspan: Union[List, np.ndarray],
                       dt,
                       event: Event = None,
                       pbar=False):
    X0 = AliasVar(X, '0')
    Y0 = AliasVar(Y, '0')
    t = ComputeParam('t')
    t0 = ComputeParam('t0')
    Dt = ComputeParam('Dt')
    scheme = X - X0 - Dt / 2 * (F(X, Y, t) + F(X0, Y0, t0))
    ae = dae.discretize(scheme)

    tspan = np.array(tspan)
    T_initial = tspan[0]
    T_end = tspan[-1]
    i = 0
    t = T_initial

    xi1 = x[0]  # x_{i+1}
    xi0 = xi1.derive_alias('0')  # x_{i}

    if pbar:
        bar = tqdm.tqdm(total=T_end)

    while abs(t - T_end) > abs(dt) / 10:
        ae.update_param('Dt', dt)

        if event:
            ae.update_param(event, t)

        ae.update_param(xi0)
        ae.update_param('t', t + dt)
        ae.update_param('t0', t)

        xi1 = nr_method(ae, xi1)
        xi0.array[:] = xi1.array

        x[i + 1] = xi1
        if pbar:
            bar.update(dt)
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


def ode15s(ode: DAE,
           y: TimeVars,
           dt,
           T,
           event: Event = None):
    pass


# def DTsolver(eqn: DTeqn, **args):
#     if 'x' not in args:
#         raise ValueError("No TimeVars input")
#     else:
#         x_: TimeVars = args['x']
#         K: int = args['K']
#         x = DTcache(x_, K)
#         xdt = DTvar(x)
#         if 'y' in args:
#             y_: TimeVars = args['y']
#             y = DTcache(y_, K)
#             x, y = eqn.lambdify(x, y)  # attach intermediate variables
#             ydt = DTvar(y)
#         else:
#             y = None
#             x, y = eqn.lambdify(x, y)  # attach intermediate variables
#             if y is not None:
#                 ydt = DTvar(y)
#
#     dt = args['dt']
#     dt_array = np.array([[dt ** i] for i in range(K + 1)])
#     T = args['T']
#     i = 0
#     t = 0
#     done = False
#
#     if y is None:
#         # ode and no intermediate variables
#         while not done:
#             i = i + 1
#             # Stretch the step if within 10% of T-t.
#             if dt >= np.abs(T - t):
#                 dt = T - t
#                 done = True
#
#             # recursive calculation
#             for k in range(0, K):
#                 x[:, k + 1] = eqn.F(k)
#
#             # store the DTs
#             xdt[i] = x
#             # update initial values and the coefficient matrix
#             x[:, 0] = x @ dt_array
#
#             t = t + dt
#
#         return xdt
#
#     else:
#         while not done:
#
#             # Stretch the step if within 10% of T-t.
#             if dt >= np.abs(T - t):
#                 dt = T - t
#                 done = True
#             A = eqn.A()
#             Ainv = inv(A)
#             # recursive calculation
#             for k in range(0, K):
#                 x[k + 1, :] = eqn.F(k)
#                 y[k + 1, :] = Ainv @ eqn.G(k + 1)  # Ainv matrix is cached
#
#             # store the DTs
#             xdt[i] = x[:, :]
#             ydt[i] = y[:, :]
#             # update initial values and the coefficient matrix
#
#             x[0, :] = (x.T @ dt_array).reshape(-1, )
#             y[0, :] = (y.T @ dt_array).reshape(-1, )
#
#             t = t + dt
#             i = i + 1
#         return xdt, ydt


def Rodas(dae: DAE,
          tspan: Union[List, np.ndarray],
          y0: Vars,
          z0: Vars = None,
          opt: Opt = None,
          event: Event = None,
          scheme='rodas'):
    s, pord, gamma, b, bd, alpha, gamma_tilde = Rodas_param(scheme)

    if opt is None:
        opt = Opt()

    if z0 is not None:
        y0 = combine_Vars(y0, z0)

    tspan = np.array(tspan)
    tend = tspan[-1]
    t0 = tspan[0]
    if opt.hmax is None:
        hmax = np.abs(tend-t0)
    nt = 0
    t = 0
    hmin = 16 * np.spacing(t)
    uround = np.spacing(1.0)
    T = np.zeros((10001,))
    T[nt] = t0
    y = TimeVars(y0, 10000)

    # The initial step size
    if opt.hinit is None:
        dt = 1e-6 * (tend - t0)
    else:
        dt = opt.hinit

    dt = np.maximum(dt, hmin)
    dt = np.minimum(dt, hmax)

    dae.assign_eqn_var_address(y0)
    s_num = dae.state_num
    a_num = dae.algebra_num

    M = csc_array((np.ones((s_num,)), (np.arange(s_num), np.arange(s_num))),
                  (dae.eqn_size, dae.vsize))

    done = False
    while not done:
        # step size too small
        # pass

        # Stretch the step if within 10% of T-t.
        if t + dt >= tend:
            dt = tend - t
        else:
            dt = np.minimum(dt, 0.5 * (tend - t))

        J = dae.j(y0)
        if done:
            break
        K = np.zeros((dae.vsize, s))
        rhs = dae.F(y0)

        lu = sla.splu(M - dt * gamma * J)
        K[:, 0] = lu.solve(rhs)

        for j in range(1, s):
            sum_1 = K @ alpha[:, j]
            sum_2 = K @ gamma_tilde[:, j]
            y1 = y0 + dt * sum_1
            rhs = dae.F(y1) + M @ sum_2
            sol = lu.solve(rhs)
            K[:, j] = sol - sum_2

        sum_1 = K @ (dt * b)
        ynew = y0 + sum_1
        sum_2 = K @ (dt * bd)

        SK = (opt.atol + opt.rtol * abs(ynew)).reshape((-1,))
        err = np.max(np.abs((sum_1 - sum_2) / SK))
        err = np.maximum(err, 1.0e-6)
        fac = opt.f_savety / (err ** (1 / pord))
        fac = np.minimum(opt.facmax, np.maximum(opt.fac1, fac))
        dtnew = dt * fac

        if err <= 1.0:

            told = t
            t = t + dt

            if event is not None:  # event
                pass

            if opt.dense_output:  # dense_output
                pass
            else:
                nt = nt + 1
                T[nt] = t
                y[nt] = ynew

            if np.abs(tend - t) < uround:
                done = True
            y0 = ynew

        dt = np.min([hmax, np.max([hmin, dtnew])])

    T = T[0:nt + 1]
    y = y[0:nt + 1]
    return T, y


def Rodas_param(scheme: str = 'rodas'):
    if scheme == 'rodas':
        s = 6
        pord = 4
        alpha = np.zeros((s, s))
        beta = np.zeros((s, s))
        gamma = 0.25
        alpha[1, 0] = 3.860000000000000e-01
        alpha[2, 0:2] = [1.460747075254185e-01, 6.392529247458190e-02]
        alpha[3, 0:3] = [-3.308115036677222e-01, 7.111510251682822e-01, 2.496604784994390e-01]
        alpha[4, 0:4] = [-4.552557186318003e+00, 1.710181363241323e+00, 4.014347332103149e+00, -1.719715090264703e-01]
        alpha[5, 0:5] = [2.428633765466977e+00, -3.827487337647808e-01, -1.855720330929572e+00, 5.598352992273752e-01,
                         2.499999999999995e-01]
        beta[1, 0] = 3.170000000000250e-02
        beta[2, 0:2] = [1.247220225724355e-02, 5.102779774275723e-02]
        beta[3, 0:3] = [1.196037669338736e+00, 1.774947364178279e-01, -1.029732405756564e+00]
        beta[4, 0:4] = [2.428633765466977e+00, -3.827487337647810e-01, -1.855720330929572e+00, 5.598352992273752e-01]
        beta[5, 0:5] = [3.484442712860512e-01, 2.130136219118989e-01, -1.541025326623184e-01, 4.713207793914960e-01,
                        -1.286761399271284e-01]
        b = np.zeros((6,))
        b[0:5] = beta[5, 0:5]
        b[5] = gamma
        bd = np.zeros((6,))
        bd[0:4] = beta[4, 0:4]
        bd[4] = gamma
        # c
        # d
        gamma_tilde = beta - alpha
        gamma_tilde = gamma_tilde / gamma
        alpha = alpha.T
        gamma_tilde = gamma_tilde.T
        return s, pord, gamma, b, bd, alpha, gamma_tilde
    elif scheme == 'rodasp':
        s = 6
        pord = 4
        alpha = np.zeros((s, s))
        beta = np.zeros((s, s))
        gamma = 0.25
        alpha[1, 0] = 0.75
        alpha[2, 0:2] = [0.0861204008141522, 0.123879599185848]
        alpha[3, 0:3] = [0.774934535507324, 0.149265154950868, -0.294199690458192]
        alpha[4, 0:4] = [5.30874668264614, 1.33089214003727, -5.37413781165556, -0.265501011027850]
        alpha[5, 0:5] = [-1.76443764877448, -0.474756557206303, 2.36969184691580, 0.619502359064983, 0.250000000000000]
        beta[1, 0] = 0.0
        beta[2, 0:2] = [-0.0493920000000000, -0.0141120000000000]
        beta[3, 0:3] = [-0.482049469387756, -0.100879555555556, 0.926729024943312]
        beta[4, 0:4] = [-1.76443764877448, -0.474756557206303, 2.36969184691580, 0.619502359064983]
        beta[5, 0:5] = [-0.0803683707891135, -0.0564906135924476, 0.488285630042799, 0.505716211481619,
                        -0.107142857142857]
        b = np.zeros((6,))
        b[0:5] = beta[5, 0:5]
        b[5] = gamma
        bd = np.zeros((6,))
        bd[0:4] = beta[4, 0:4]
        bd[4] = gamma
        # c
        # d
        gamma_tilde = beta - alpha
        gamma_tilde = gamma_tilde / gamma
        alpha = alpha.T
        gamma_tilde = gamma_tilde.T
        return s, pord, gamma, b, bd, alpha, gamma_tilde
    else:
        raise ValueError("Not implemented")


class Opt:
    def __init__(self,
                 atol=1e-6,
                 rtol=1e-3,
                 f_savety=0.9,
                 facmax=6,
                 fac1=0.2,
                 fac2=6,
                 dense_output=False,
                 hinit=None,
                 hmax=None):
        self.atol = atol
        self.rtol = rtol
        self.f_savety = f_savety
        self.facmax = facmax
        self.fac1 = fac1
        self.fac2 = fac2
        self.dense_output = dense_output
        self.hinit = hinit
        self.hmax = hmax


class Stats:
    pass
