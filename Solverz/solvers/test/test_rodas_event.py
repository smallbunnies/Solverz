import numpy as np
from Solverz import Ode, iVar, Opt, Rodas, as_Vars, DAE, made_numerical, TimeVars


# %% event test
def test_bounceball():
    x = iVar('x', [0, 20])
    y0 = as_Vars(x)
    f1 = Ode('f1', x[1], x[0])
    f2 = Ode('f2', -9.8, x[1])
    bball = DAE([f1, f2])
    nbball = made_numerical(bball, y0, sparse=True)

    def events(t, y):
        value = np.array([y[0]])
        isterminal = np.array([1])
        direction = np.array([-1])
        return value, isterminal, direction

    opt = Opt(event=events)
    tout = np.array([0])
    yout = TimeVars(y0, 1)
    teout = np.array([])
    yeout = TimeVars(y0, 0)
    ieout = np.array([], dtype=int)
    tstart = 0
    tend = 30
    for i in range(10):
        sol = Rodas(nbball, np.linspace(tstart, tend, 100), y0, opt)

        tout = np.concatenate([tout, sol.T[1:]])
        yout.append(sol.Y[1:])
        teout = np.concatenate([teout, sol.te])
        yeout.append(sol.ye)
        ieout = np.concatenate([ieout, sol.ie])

        y0['x'][0] = 0
        y0['x'][1] = -0.9 * sol.Y[-1]['x'][1]
        tstart = sol.T[-1]

    np.testing.assert_allclose(teout,
                               np.array([4.081633047365558, 7.755103268847604, 11.061226369910107, 14.03673684129459,
                                         16.714695668619328, 19.124858513456523, 21.294005283692695, 23.246237159791765,
                                         25.003245923211466, 26.584554164220997]),
                               rtol=1e-5,
                               atol=0)


def test_orbit():
    mu = 1 / 82.45
    mustar = 1 - mu
    y = iVar('y', [1.2, 0, 0, -1.04935750983031990726])
    f1 = Ode('f1', y[2], y[0])
    f2 = Ode('f2', y[3], y[1])
    r13 = ((y[0] + mu) ** 2 + y[1] ** 2) ** 1.5
    r23 = ((y[0] - mustar) ** 2 + y[1] ** 2) ** 1.5
    f3 = Ode('f3',
             2 * y[3] + y[0] - mustar * ((y[0] + mu) / r13) - mu * ((y[0] - mustar) / r23),
             y[2])
    f4 = Ode('f4',
             -2 * y[2] + y[1] - mustar * (y[1] / r13) - mu * (y[1] / r23),
             y[3])
    y0 = as_Vars(y)
    orbit = DAE([f1, f2, f3, f4])
    norbit = made_numerical(orbit, y0, sparse=True)

    def events(t, y):
        dDSQdt = 2 * (y[0:2] - np.array([1.2, 0])).dot(y[2:4])
        value = np.array([dDSQdt, dDSQdt])
        isterminal = np.array([1, 0])
        direction = np.array([1, -1])
        return value, isterminal, direction

    sol = Rodas(norbit, [0, 7], y0, Opt(event=events, rtol=1e-5, atol=1e-4))

    np.testing.assert_allclose(sol.te,
                               np.array([3.09609516, 6.19215951]),
                               rtol=1e-5,
                               atol=0)
    np.testing.assert_allclose(sol.ye,
                               np.array([[-1.262468e+00, 1.162918e-05, 4.956687e-06, 1.049574e+00],
                                         [1.200024e+00, 1.455118e-09, 6.495375e-05, -1.049349e+00]]),
                               rtol=1e-5,
                               atol=0)
