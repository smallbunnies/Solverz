from Solverz.algebra import AliasVar, ComputeParam, F, X, Y
from Solverz.eqn import Ode, Eqn
from Solverz.equations import DAE
from Solverz.param import Param
from Solverz.var import TimeVar
from Solverz.variables import TimeVars
from Solverz.solver import implicit_trapezoid
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X0 = AliasVar(X, '0')
Y0 = AliasVar(Y, '0')
X_1 = AliasVar(X, '_1')
Y_1 = AliasVar(Y, '_1')
t = ComputeParam('t')
t0 = ComputeParam('t0')
dt = ComputeParam('dt')

scheme = X - X0 - dt / 2 * (F(X, Y, t) + F(X0, Y0, t0))
f = Ode(name='f', e_str='-x**3+0.5*y**2', diff_var='x')
g = Eqn(name='g', e_str='x**2+y**2-2')
dae = DAE([f, g])
x = TimeVar('x')
x.v0 = [1]
y = TimeVar('y')
y.v0 = [1]

xy = implicit_trapezoid(dae, TimeVars([x, y], length=201), 0.1, 20)
plt.plot(np.arange(0, 20.1, 0.1), xy.array[0, :].reshape((-1,)))
plt.plot(np.arange(0, 20.1, 0.1), xy.array[1, :].reshape((-1,)))

df = pd.read_excel('../instances/dae_test.xlsx',
                   sheet_name=None,
                   engine='openpyxl',
                   header=None
                   )


def test_discretize():
    c = ComputeParam('c')
    Xk1 = AliasVar(X, 'k1')
    Yk1 = AliasVar(Y, 'k1')
    scheme1 = F(X + 1 / 2 * Xk1, Y + 1 / 3 * Yk1, t + dt * c)
    f1 = Ode(name='F', e_str='(Pm-D*omega)', diff_var='omega')

    param0, eqn0 = f1.discretize(scheme1)
    assert 'omegak1' in param0.keys()
    assert 'Dk1' in param0.keys()
    assert 'Pmk1' in param0.keys()
    assert eqn0.e_str == 'Pm + 0.333333333333333*Pmk1 - (D + 0.333333333333333*Dk1)*(omega + 0.5*omegak1)'

    param1, eqn1 = f1.discretize(scheme1, param={'D': Param('D'),
                                                 'Pm': Param('Pm')})
    assert 'omegak1' in param1.keys()
    assert 'D' in param1.keys()
    assert 'Pm' in param1.keys()
    assert eqn1.e_str == '-D*(omega + 0.5*omegak1) + Pm'

    param2, eqn2 = f1.discretize(scheme1, extra_diff_var=['D'],
                                 param={'Pm': Param('Pm')})
    assert 'omegak1' in param2.keys()
    assert 'Dk1' in param2.keys()
    assert 'Pm' in param2.keys()
    assert eqn2.e_str == 'Pm - (D + 0.5*Dk1)*(omega + 0.5*omegak1)'


def test_dae():
    xy_bench = np.asarray(df['Sheet1'])
    assert max(abs((xy.array[0, :] - xy_bench[:, 0].reshape(1, -1))).reshape(-1, )) <= 0.0006665273280143102
    assert max(abs((xy.array[1, :] - xy_bench[:, 1].reshape(1, -1))).reshape(-1, )) <= 0.000569613549821657
