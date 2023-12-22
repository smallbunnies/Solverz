from __future__ import annotations
from typing import Union, List

import numpy as np
from numpy import sin, cos
from scipy.sparse import csc_array, coo_array, linalg as sla
from scipy.interpolate import interp1d
import pandas as pd

from Solverz.variable.variables import TimeVars, Vars

df = pd.read_excel('instances/test_m3b9.xlsx',
                   sheet_name=None,
                   engine='openpyxl',
                   header=None
                   )

p = dict()
p['B'] = np.asarray(df['B'])
p['G'] = np.asarray(df['G'])
Gv = np.array([2.80472685e+00, 1.00000000e+04, 1.00000000e+04, 2.80472685e+00,
               2.80472685e+00, 2.80472685e+00])
Gt = np.array([0.000000e+00, 2.000000e-03, 3.000000e-02, 3.200000e-02,
               1.000000e+01, 1.000001e+01])
p['Gvar'] = interp1d(np.array([0.000000e+00, 2.000000e-03, 3.000000e-02, 3.200000e-02,
                               1.000000e+01, 1.000001e+01]),
                     np.array([2.80472685e+00, 1.00000000e+04, 1.00000000e+04, 2.80472685e+00,
                               2.80472685e+00, 2.80472685e+00]))
p['D'] = np.array([10, 10, 10])
p['Tj'] = np.array([47.2800, 12.8000, 6.0200])
p['g'] = np.array([0, 1, 2], dtype=int)
p['ng'] = np.array([3, 4, 5, 6, 7, 8], dtype=int)
p['Xqp'] = np.array([0.0969, 0.8645, 1.2578])
p['Xdp'] = np.array([0.0608, 0.1198, 0.1813])
p['ra'] = np.array([0.0000, 0.0000, 0.0000])
p['Pm'] = np.array([0.7164, 1.6300, 0.8500])
p['Edp'] = np.array([0.0000, 0.0000, 0.0000])
p['Eqp'] = np.array([1.05636632091501, 0.788156757672709, 0.767859471854610])
p['omega_b'] = np.array(376.991118430775)


class m3b9DAE:

    def __init__(self, p):
        self.M = coo_array(
            (np.array([1., 1., 1., 1., 1., 1.]), (np.array([0, 1, 2, 3, 4, 5]), np.array([0, 1, 2, 3, 4, 5]))),
            shape=(30, 30)).tocsc()
        self.eqnsize = 30
        self.vsize = 30
        self.p = p

    def j(self, t, y, p) -> csc_array:

        delta = y[0:3]
        omega = y[3:6]
        Ixg = y[6:9]
        Iyg = y[9:12]
        Ux = y[12:21]
        Uy = y[21:30]

        B = p['B']
        G = p['G']
        G[6, 6] = p['Gvar'](t)
        D = p['D']
        Tj = p['Tj']
        g = p['g']
        ng = p['ng']
        Xqp = p['Xqp']
        Xdp = p['Xdp']
        ra = p['ra']
        omega_b = p['omega_b']

        a_eqn_delta = np.arange(0, 3)
        a_eqn_rotatorspeed = np.arange(3, 6)
        a_eqn_Ed_prime = np.arange(6, 9)
        a_eqn_Eq_prime = np.arange(9, 12)
        a_eqn_Ixg_inject = np.arange(12, 15)
        a_eqn_Iyg_inject = np.arange(15, 18)
        a_eqn_Ixng_inject = np.arange(18, 24)
        a_eqn_Iyng_inject = np.arange(24, 30)
        a_delta = np.arange(0, 3)
        a_omega = np.arange(3, 6)
        a_Ixg = np.arange(6, 9)
        a_Iyg = np.arange(9, 12)
        a_Ux = np.arange(12, 21)
        a_Uy = np.arange(21, 30)

        temp = np.zeros((30, 30))
        temp[a_eqn_delta, a_omega] = omega_b
        temp[a_eqn_rotatorspeed, a_omega] += -D / Tj
        temp[a_eqn_rotatorspeed, a_Ixg] += (-Ux[g] - 2 * ra * Ixg) / Tj
        temp[a_eqn_rotatorspeed, a_Iyg] += (-Uy[g] - 2 * ra * Iyg) / Tj
        temp[a_eqn_rotatorspeed, a_Ux[g]] += -Ixg / Tj
        temp[a_eqn_rotatorspeed, a_Uy[g]] += -Iyg / Tj
        temp[a_eqn_Ed_prime, a_delta] += -(Ux[g] - Xqp * Iyg + ra * Ixg) * cos(delta) - (
                Uy[g] + Xqp * Ixg + ra * Iyg) * sin(delta)
        temp[a_eqn_Ed_prime, a_Ixg] += Xqp * cos(delta) - ra * sin(delta)
        temp[a_eqn_Ed_prime, a_Iyg] += Xqp * sin(delta) + ra * cos(delta)
        temp[a_eqn_Ed_prime, a_Ux[g]] += -sin(delta)
        temp[a_eqn_Ed_prime, a_Uy[g]] += cos(delta)
        temp[a_eqn_Eq_prime, a_delta] += (Ux[g] - Xdp * Iyg + ra * Ixg) * sin(delta) - (
                Uy[g] + Xdp * Ixg + ra * Iyg) * cos(delta)
        temp[a_eqn_Eq_prime, a_Ixg] += -Xdp * sin(delta) - ra * cos(delta)
        temp[a_eqn_Eq_prime, a_Iyg] += Xdp * cos(delta) - ra * sin(delta)
        temp[a_eqn_Eq_prime, a_Ux[g]] += -cos(delta)
        temp[a_eqn_Eq_prime, a_Uy[g]] += -sin(delta)
        temp[a_eqn_Ixg_inject, a_Ixg] += 1
        temp[np.ix_(a_eqn_Ixg_inject, a_Ux)] += -G[g, ::]
        temp[np.ix_(a_eqn_Ixg_inject, a_Uy)] += B[g, ::]
        temp[a_eqn_Iyg_inject, a_Iyg] += 1
        temp[np.ix_(a_eqn_Iyg_inject, a_Ux)] += -B[g, ::]
        temp[np.ix_(a_eqn_Iyg_inject, a_Uy)] += -G[g, ::]
        temp[np.ix_(a_eqn_Ixng_inject, a_Ux)] += G[ng, ::]
        temp[np.ix_(a_eqn_Ixng_inject, a_Uy)] += -B[ng, ::]
        temp[np.ix_(a_eqn_Iyng_inject, a_Ux)] += B[ng, ::]
        temp[np.ix_(a_eqn_Iyng_inject, a_Uy)] += G[ng, ::]

        return csc_array(temp)

    def F(self, t, y, p):
        B = p['B']
        G = p['G']
        G[6, 6] = p['Gvar'](t)
        D = p['D']
        Tj = p['Tj']
        g = p['g']
        ng = p['ng']
        Xqp = p['Xqp']
        Xdp = p['Xdp']
        ra = p['ra']
        Pm = p['Pm']
        Edp = p['Edp']
        Eqp = p['Eqp']
        g = p['g']
        ng = p['ng']
        omega_b = p['omega_b']

        temp = np.zeros(30, )
        delta = y[0:3]
        omega = y[3:6]
        Ixg = y[6:9]
        Iyg = y[9:12]
        Ux = y[12:21]
        Uy = y[21:30]

        temp[0:3] = omega_b * (omega - 1)
        temp[3:6] = (-Ux[g] * Ixg - Uy[g] * Iyg - D * (omega - 1) + Pm - 0 * (Ixg ** 2 + Iyg ** 2)) / Tj  # rotor speed
        temp[6:9] = Edp - (Ux[g] - Xqp * Iyg + ra * Ixg) * sin(delta) + (Uy[g] + Xqp * Ixg + ra * Iyg) * cos(delta)
        temp[9:12] = Eqp - (Ux[g] - Xdp * Iyg + ra * Ixg) * cos(delta) - (Uy[g] + Xdp * Ixg + 0 * Iyg) * sin(delta)
        temp[12:15] = Ixg - (-B[g, ::] @ Uy + G[g, ::] @ Ux)
        temp[15:18] = Iyg - (B[g, ::] @ Ux + G[g, ::] @ Uy)
        temp[18:24] = -B[ng, ::] @ Uy + G[ng, ::] @ Ux
        temp[24:30] = B[ng, ::] @ Ux + G[ng, ::] @ Uy

        return temp


def Rodas(dae: m3b9DAE,
          tspan: Union[List, np.ndarray],
          y0,
          p,
          opt: Opt = None):
    if opt is None:
        opt = Opt()
    stats = Stats(opt.scheme)

    rparam = Rodas_param(opt.scheme)

    tspan = np.array(tspan)
    tend = tspan[-1]
    t0 = tspan[0]
    if opt.hmax is None:
        opt.hmax = np.abs(tend - t0)
    nt = 0
    t = 0
    hmin = 16 * np.spacing(t)
    uround = np.spacing(1.0)
    T = np.zeros((10001,))
    T[nt] = t0
    y = np.zeros((10000, 30))
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

    done = False
    reject = 0
    while not done:
        # step size too small
        # pass

        if np.abs(dt) < uround:
            print(f"Error exit of RODAS at time = {t}: step size too small h = {dt}.\n")

        # Stretch the step if within 10% of T-t.
        if t + dt >= tend:
            dt = tend - t
        else:
            dt = np.minimum(dt, 0.5 * (tend - t))

        if opt.fix_h:
            dt = opt.hinit

        if done:
            break
        K = np.zeros((dae.vsize, rparam.s))

        if reject == 0:
            J = dae.j(t, y0, p)

        dfdt0 = dt * dfdt(dae, t, y0, p)
        rhs = dae.F(t, y0, p) + rparam.g[0] * dfdt0
        stats.nfeval = stats.nfeval + 1

        lu = sla.splu(M - dt * rparam.gamma * J)
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
                y[nt, :] = ynew

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
    y = y[0:nt + 1, :]
    return T, y, stats


class Rodas_param:

    def __init__(self,
                 scheme: str = 'rodas'):
        if scheme == 'rodas':
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

        elif scheme == 'rodasp':
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
                 scheme='rodas',
                 fix_h: bool = False,
                 hinit=None,
                 hmax=None):
        self.atol = atol
        self.rtol = rtol
        self.f_savety = f_savety
        self.facmax = facmax
        self.fac1 = fac1
        self.fac2 = fac2
        self.fix_h = fix_h  # To force the step sizes to be invariant. This is not robust.
        self.hinit = hinit
        self.hmax = hmax
        self.scheme = scheme


def dfdt(dae, t, y, p):
    tscale = np.maximum(0.1 * np.abs(t), 1e-8)
    ddt = t + np.sqrt(np.spacing(1)) * tscale - t
    f0 = dae.F(t, y, p)
    f1 = dae.F(t + ddt, y, p)
    return (f1 - f0) / ddt


class Stats:

    def __init__(self, scheme):
        self.scheme = scheme
        self.nstep = 0
        self.nfeval = 0
        self.ndecomp = 0
        self.nreject = 0

    def __repr__(self):
        return f"Scheme {self.scheme}, nstep: {self.nstep}, nfeval: {self.nfeval}, ndecomp: {self.ndecomp}, nreject: {self.nreject}."


y0 = np.array([6.25815078e-02, 1.06638275e+00, 9.44865049e-01, 1.00000000e+00,
               1.00000000e+00, 1.00000000e+00, 6.88836022e-01, 1.57988988e+00,
               8.17891312e-01, -2.60077645e-01, 1.92406178e-01, 1.73047792e-01,
               1.04000110e+00, 1.01157933e+00, 1.02160344e+00, 1.02502063e+00,
               9.93215118e-01, 1.01056074e+00, 1.02360471e+00, 1.01579907e+00,
               1.03174404e+00, 9.38510394e-07, 1.65293826e-01, 8.33635520e-02,
               -3.96760163e-02, -6.92587531e-02, -6.51191655e-02, 6.65507084e-02,
               1.29050647e-02, 3.54351212e-02])
# f1 = F(0, y0, p)
# j1 = j(0, y0, p)
# c1 = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.11969465e-07,
#                9.76818626e-07, 2.24989306e-06, -3.33066907e-16, -2.88657986e-15,
#                -1.66533454e-15, 2.46142137e-09, 2.57938115e-10, -1.28045130e-10,
#                5.55111512e-16, 8.88178420e-16, -3.55271368e-15, 1.16573418e-15,
#                9.49240686e-15, 4.94049246e-15, 3.77475828e-15, -3.99680289e-15,
#                2.55351296e-15, -3.21964677e-14, 1.68753900e-14, -8.54871729e-15,
#                -1.73389081e-13, 1.87072580e-14, 1.55431223e-14, 1.28563826e-13,
#                -1.71807013e-13, 3.64153152e-14])

T1, y, stats2 = Rodas(m3b9DAE(p), np.linspace(0, 10, 5001).reshape(-1, ), y0, p, opt=Opt(hinit=1e-5))

df = pd.read_excel('instances/test_m3b9.xlsx',
                   sheet_name=None,
                   engine='openpyxl',
                   header=None
                   )
# import matplotlib.pyplot as plt
#
# plt.plot(T1, df['delta'][2], label='ode15s')
# plt.plot(T1, y[:, 2], label='rodas', linestyle=':')
# plt.legend()
# plt.grid()
# plt.show()


def test_m3b9():
    assert np.abs(np.asarray(df['omega']) - y[:, 3:6]).max() <= 1.62e-5
    assert np.abs(np.asarray(df['delta']) - y[:, 0:3]).max() <= 9.3e-4
