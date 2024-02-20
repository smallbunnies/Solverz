from __future__ import annotations

from typing import Union, List

import numpy as np
import tqdm
from numpy import abs, linalg
from scipy.sparse import csc_array, linalg as sla

from Solverz.equation.equations import DAE
from Solverz.solvers.nlaesolver import nr_method
from Solverz.sym_algebra.symbols import Var
from Solverz.variable.variables import TimeVars, Vars, as_Vars, combine_Vars
from Solverz.numerical_interface.num_eqn import nDAE, nAE
from Solverz.solvers.stats import Stats
from Solverz.solvers.option import Opt
from Solverz.solvers.parser import dae_io_parser
from Solverz.solvers.laesolver import lu_decomposition


@dae_io_parser
def implicit_trapezoid(dae: nDAE,
                       tspan: List | np.ndarray,
                       y0: np.ndarray,
                       opt: Opt = None):
    stats = Stats(scheme='Trapezoidal')
    if opt is None:
        opt = Opt(stats=True)
    dt = opt.hinit
    tspan = np.array(tspan)
    T_initial = tspan[0]
    T_end = tspan[-1]
    nt = 0
    tt = T_initial
    t0 = tt

    y = np.zeros((10000, y0.shape[0]))
    y[0, :] = y0
    T = np.zeros((10000,))

    p = dae.p
    while abs(tt - T_end) > abs(dt) / 10:
        My0 = dae.M @ y0
        F0 = dae.F(t0, y0, p)
        ae = nAE(lambda y_, p_: dt / 2 * (dae.F(t0 + dt, y_, p_) + F0) - dae.M @ y_ + My0,
                 lambda y_, p_: -dae.M + dt / 2 * dae.J(t0 + dt, y_, p_),
                 p)

        y1, ite = nr_method(ae, y0, Opt(stats=True))
        stats.ndecomp = stats.ndecomp + ite
        stats.nfeval = stats.nfeval + ite

        tt = tt + dt
        nt = nt + 1
        y[nt] = y1
        T[nt] = tt
        y0 = y1
        t0 = tt

    y = y[0:nt + 1]
    T = T[0:nt + 1]
    stats.nstep = nt

    return T, y, stats


def backward_euler(dae: nDAE,
                   tspan: Union[List, np.ndarray],
                   y0: np.ndarray,
                   dt,
                   opt: Opt = None):
    stats = Stats(scheme='Backward Euler')
    if opt is None:
        opt = Opt()

    tspan = np.array(tspan)
    T_initial = tspan[0]
    T_end = tspan[-1]
    nt = 0
    tt = T_initial
    t0 = tt

    y = np.zeros((10000, y0.shape[0]))
    y[0, :] = y0
    T = np.zeros((10000,))

    p = dae.p
    while abs(tt - T_end) > abs(dt) / 10:
        My0 = dae.M @ y0
        ae = nAE(lambda y_, p_: dae.M @ y_ - My0 - dt * dae.F(t0 + dt, y_, p_),
                 lambda y_, p_: dae.M - dt * dae.J(t0 + dt, y_, p_),
                 p)

        y1, ite = nr_method(ae, y0, stats=True, tol=opt.ite_tol)
        stats.ndecomp = stats.ndecomp + ite
        stats.nfeval = stats.nfeval + ite

        tt = tt + dt
        nt = nt + 1
        y[nt] = y1
        T[nt] = tt
        y0 = y1
        t0 = tt

    y = y[0:nt + 1]
    T = T[0:nt + 1]
    stats.nstep = nt

    return T, y, stats


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
           T):
    pass


@dae_io_parser
def Rodas(dae: nDAE,
          tspan: List | np.ndarray,
          y0: np.ndarray,
          opt: Opt = None):
    if opt is None:
        opt = Opt()
    stats = Stats(opt.scheme)

    rparam = Rodas_param(opt.scheme)
    vsize = y0.shape[0]
    tspan = np.array(tspan)
    tend = tspan[-1]
    t0 = tspan[0]
    if opt.hmax is None:
        opt.hmax = np.abs(tend - t0)
    nt = 0
    t = t0
    hmin = 16 * np.spacing(t0)
    uround = np.spacing(1.0)
    T = np.zeros((10001,))
    T[nt] = t0
    y = np.zeros((10000, vsize))
    y[0, :] = y0

    dense_output = False
    n_tspan = len(tspan)
    told = t0
    if n_tspan > 2:
        dense_output = True
        inext = 1
        tnext = tspan[inext]

    # The initial step size
    if opt.hinit is None:
        dt = 1e-6 * (tend - t0)
    else:
        dt = opt.hinit

    dt = np.maximum(dt, hmin)
    dt = np.minimum(dt, opt.hmax)

    M = dae.M
    p = dae.p
    done = False
    reject = 0
    while not done:
        # step size too small
        # pass

        if np.abs(dt) < uround:
            print(f"Error exit of RODAS at time = {t}: step size too small h = {dt}.\n")
            break

        if reject > 100:
            print("Step rejected over 100 times.\n")
            break

        # Stretch the step if within 10% of T-t.
        if t + dt >= tend:
            dt = tend - t
        else:
            dt = np.minimum(dt, 0.5 * (tend - t))

        if opt.fix_h:
            dt = opt.hinit

        if done:
            break
        K = np.zeros((vsize, rparam.s))

        if reject == 0:
            J = dae.J(t, y0, p)

        dfdt0 = dt * dfdt(dae, t, y0)
        rhs = dae.F(t, y0, p) + rparam.g[0] * dfdt0
        stats.nfeval = stats.nfeval + 1

        lu = lu_decomposition(M - dt * rparam.gamma * J)
        stats.ndecomp = stats.ndecomp + 1
        K[:, 0] = lu.solve(rhs)

        for j in range(1, rparam.s):
            sum_1 = K @ rparam.alpha[:, j]
            sum_2 = K @ rparam.gamma_tilde[:, j]
            y1 = y0 + dt * sum_1

            rhs = dae.F(t + dt * rparam.a[j], y1, p) + M @ sum_2 + rparam.g[j] * dfdt0
            stats.nfeval = stats.nfeval + 1
            sol = lu.solve(rhs)
            K[:, j] = sol - sum_2

        sum_1 = K @ (dt * rparam.b)
        ynew = y0 + sum_1
        if not opt.fix_h:
            sum_2 = K @ (dt * rparam.bd)
            SK = (opt.atol + opt.rtol * abs(ynew)).reshape((-1,))
            err = np.max(np.abs((sum_1 - sum_2) / SK))
            if np.any(np.isinf(ynew)) or np.any(np.isnan(ynew)):
                err = 1.0e6
                print('Warning Rodas: NaN or Inf occurs.')
            err = np.maximum(err, 1.0e-6)
            fac = opt.f_savety / (err ** (1 / rparam.pord))
            fac = np.minimum(opt.facmax, np.maximum(opt.fac1, fac))
            dtnew = dt * fac
        else:
            err = 1.0
            dtnew = dt

        if err <= 1.0:
            reject = 0
            told = t
            t = t + dt
            stats.nstep = stats.nstep + 1

            if dense_output:  # dense_output
                while t >= tnext > told:
                    tau = (tnext - told) / dt
                    ynext = y0 + tau * dt * K @ (rparam.b + (tau - 1) * (rparam.c + tau * (rparam.d + tau * rparam.e)))
                    nt = nt + 1
                    T[nt] = tnext
                    y[nt] = ynext
                    inext = inext + 1
                    if inext <= n_tspan - 1:
                        tnext = tspan[inext]
                    else:
                        tnext = tend + dt
            else:
                nt = nt + 1
                T[nt] = t
                y[nt] = ynew

            if np.abs(tend - t) < uround:
                done = True
            y0 = ynew
            opt.facmax = opt.fac2

        else:
            reject = reject + 1
            stats.nreject = stats.nreject + 1
            opt.facmax = 1
        dt = np.min([opt.hmax, np.max([hmin, dtnew])])

    T = T[0:nt + 1]
    y = y[0:nt + 1]
    return T, y, stats


class Rodas_param:

    def __init__(self,
                 scheme: str = 'rodas'):
        match scheme:
            case 'rodas4':
                self.scheme = 'rodas'
                self.s = 6
                self.pord = 4
                self.alpha = np.zeros((self.s, self.s))
                self.beta = np.zeros((self.s, self.s))
                self.g = np.zeros((self.s, 1))
                self.gamma = 0.25
                self.alpha[1, 0] = 3.860000000000000e-01
                self.alpha[2, 0:2] = [1.460747075254185e-01, 6.392529247458190e-02]
                self.alpha[3, 0:3] = [-3.308115036677222e-01, 7.111510251682822e-01, 2.496604784994390e-01]
                self.alpha[4, 0:4] = [-4.552557186318003e+00, 1.710181363241323e+00, 4.014347332103149e+00,
                                      -1.719715090264703e-01]
                self.alpha[5, 0:5] = [2.428633765466977e+00, -3.827487337647808e-01, -1.855720330929572e+00,
                                      5.598352992273752e-01,
                                      2.499999999999995e-01]
                self.beta[1, 0] = 3.170000000000250e-02
                self.beta[2, 0:2] = [1.247220225724355e-02, 5.102779774275723e-02]
                self.beta[3, 0:3] = [1.196037669338736e+00, 1.774947364178279e-01, -1.029732405756564e+00]
                self.beta[4, 0:4] = [2.428633765466977e+00, -3.827487337647810e-01, -1.855720330929572e+00,
                                     5.598352992273752e-01]
                self.beta[5, 0:5] = [3.484442712860512e-01, 2.130136219118989e-01, -1.541025326623184e-01,
                                     4.713207793914960e-01,
                                     -1.286761399271284e-01]
                self.b = np.zeros((6,))
                self.b[0:5] = self.beta[5, 0:5]
                self.b[5] = self.gamma
                self.bd = np.zeros((6,))
                self.bd[0:4] = self.beta[4, 0:4]
                self.bd[4] = self.gamma
                self.c = np.array([-4.786970949443344e+00, -6.966969867338157e-01, 4.491962205414260e+00,
                                   1.247990161586704e+00, -2.562844308238056e-01, 0])
                self.d = np.array([1.274202171603216e+01, -1.894421984691950e+00, -1.113020959269748e+01,
                                   -1.365987420071593e+00, 1.648597281428871e+00, 0])
                self.e = np.zeros((6,))
                self.gamma_tilde = self.beta - self.alpha
                self.a = np.sum(self.alpha, axis=1)
                self.g = np.sum(self.gamma_tilde, axis=1) + self.gamma
                self.gamma_tilde = self.gamma_tilde / self.gamma
                self.alpha = self.alpha.T
                self.gamma_tilde = self.gamma_tilde.T

            case 'rodasp':
                pass
                # self.s = 6
                # self.pord = 4
                # self.alpha = np.zeros((s, s))
                # beta = np.zeros((s, s))
                # gamma = 0.25
                # alpha[1, 0] = 0.75
                # alpha[2, 0:2] = [0.0861204008141522, 0.123879599185848]
                # alpha[3, 0:3] = [0.774934535507324, 0.149265154950868, -0.294199690458192]
                # alpha[4, 0:4] = [5.30874668264614, 1.33089214003727, -5.37413781165556, -0.265501011027850]
                # alpha[5, 0:5] = [-1.76443764877448, -0.474756557206303, 2.36969184691580, 0.619502359064983,
                #                  0.250000000000000]
                # beta[1, 0] = 0.0
                # beta[2, 0:2] = [-0.0493920000000000, -0.0141120000000000]
                # beta[3, 0:3] = [-0.482049469387756, -0.100879555555556, 0.926729024943312]
                # beta[4, 0:4] = [-1.76443764877448, -0.474756557206303, 2.36969184691580, 0.619502359064983]
                # beta[5, 0:5] = [-0.0803683707891135, -0.0564906135924476, 0.488285630042799, 0.505716211481619,
                #                 -0.107142857142857]
                # b = np.zeros((6,))
                # b[0:5] = beta[5, 0:5]
                # b[5] = gamma
                # bd = np.zeros((6,))
                # bd[0:4] = beta[4, 0:4]
                # bd[4] = gamma
                # # c
                # # d
                # gamma_tilde = beta - alpha
                # gamma_tilde = gamma_tilde / gamma
                # alpha = alpha.T
                # gamma_tilde = gamma_tilde.T
            case 'rodas3d':
                self.s = 4
                self.pord = 3
                self.alpha = np.zeros((self.s, self.s))
                self.beta = np.zeros((self.s, self.s))
                self.g = np.zeros((self.s, 1))
                self.gamma = 0.57281606
                self.alpha[1, 0] = 1.2451051999132263
                self.alpha[2, 0:2] = [1, 0]
                self.alpha[3, 0:3] = [0.32630307266483527, 0.10088086733516474, 0.57281606]
                self.beta[1, 0] = -3.1474142698552949
                self.beta[2, 0:2] = [0.32630307266483527, 0.10088086733516474]
                self.beta[3, 0:3] = [0.69775271462407906, 0.056490613592447572, -0.32705938821652658]
                self.b = np.zeros((self.s,))
                self.b[0:3] = self.beta[3, 0:3]
                self.b[self.s - 1] = self.gamma
                self.bd = np.zeros((self.s,))
                self.bd[0:2] = self.beta[2, 0:2]
                self.bd[self.s - 2] = self.gamma
                # self.c = np.array([-4.786970949443344e+00, -6.966969867338157e-01, 4.491962205414260e+00,
                #                    1.247990161586704e+00, -2.562844308238056e-01, 0])
                # self.d = np.array([1.274202171603216e+01, -1.894421984691950e+00, -1.113020959269748e+01,
                #                    -1.365987420071593e+00, 1.648597281428871e+00, 0])
                # self.e = np.zeros((6,))
                self.gamma_tilde = self.beta - self.alpha
                self.a = np.sum(self.alpha, axis=1)
                self.g = np.sum(self.gamma_tilde, axis=1) + self.gamma
                self.gamma_tilde = self.gamma_tilde / self.gamma
                self.alpha = self.alpha.T
                self.gamma_tilde = self.gamma_tilde.T
            case _:
                raise ValueError("Not implemented")


def dfdt(dae: nDAE, t, y):
    tscale = np.maximum(0.1 * np.abs(t), 1e-8)
    ddt = t + np.sqrt(np.spacing(1)) * tscale - t
    f0 = dae.F(t, y, dae.p)
    f1 = dae.F(t + ddt, y, dae.p)
    return (f1 - f0) / ddt
