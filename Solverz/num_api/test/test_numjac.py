import numpy as np

from Solverz.num_api.numjac import numjac_ae


def f(x):
    a = 1 + np.exp(x[0]) + np.sin(x[1])
    b = x[1] ** 2 + np.cos(x[0])
    return np.array([a, b])


def Janaly(x):
    return np.array([[np.exp(x[0]), np.cos(x[1])],
                     [-np.sin(x[0]), 2 * x[1]]])


def Jnum(x):
    return numjac_ae(lambda xi: f(xi),
                     x,
                     np.ones(len(x)) * 1.e-12)[0]


point = np.array([1.0, 2.0])

np.testing.assert_allclose(Jnum(point).toarray(), Janaly(point), rtol=1e-5)
