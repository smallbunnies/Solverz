import numpy as np

from Solverz import idx, iVar, Para, Sign, as_Vars, nr_method, Eqn, AE, Mat_Mul, made_numerical

k = idx('k', value=[0, 1, 2])
i = idx('i', value=[0, 0, 3])
j = idx('j', value=[1, 2, 2])
m = idx('m', value=[1, 2, 3])

Pi = iVar('Pi', value=[50, 49, 45, 1.3 * 49])  # 50, 40.8160, 49.7827, 1.3 * 40.8160
fin = iVar('fin', value=[34.6406, 10.8650, 23.7756, 0])  # 29.8308, 4.8098, 18.9658
finset = Para('finset', value=[10.8650, 23.7756, 0])
f = iVar('f', value=[10, 5, 25, 25])
c = Para('c', value=[1.0329225961928894, 1.0329267609699861, 1.032931789457253, 1])
A = Para('A', dim=2, value=[[-1, -1, 0, 0], [1, 0, 0, -1], [0, 1, 1, 0], [0, 0, -1, 1]])

eqn1 = Eqn(name='mass flow continuity', eqn=finset - Mat_Mul(A[m, :], f))
eqn2 = Eqn(name='mass flow1',
           eqn=f[k] - c[k] * Sign(Pi[i] - Pi[j]) * (Sign(Pi[i] - Pi[j]) * (Pi[i] ** 2 - Pi[j] ** 2)) ** (1 / 2))
eqn3 = Eqn(name='pressure', eqn=Pi[0] - 50)
eqn4 = Eqn(name='pressure1', eqn=Pi[3] - 1.3 * Pi[1])

gas_flow = AE(eqn=[eqn1, eqn2, eqn3, eqn4])
y0 = as_Vars([f, Pi])

ngas_flow = made_numerical(gas_flow, y0)
sol = nr_method(ngas_flow, y0)


def test_nr_method():
    bench = np.array([29.830799999999996, 4.809800000000001, 18.965799999999998,
                      18.965799999999998, 50.0, 40.816, 49.78269999999999, 53.06080000000001])
    assert np.max(np.abs((sol.y - bench) / bench)) <= 1e-8
