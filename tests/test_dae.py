import pandas as pd
import numpy as np

from Solverz import Eqn, Ode, DAE, Var, as_Vars, Rodas, Opt

x = Var('x', 1)
y = Var('y', 1)

f = Ode(name='f', f=-x ** 3 + 0.5 * y ** 2, diff_var=x)
g = Eqn(name='g', eqn=x ** 2 + y ** 2 - 2)
dae = DAE([f, g])

df = pd.read_excel('instances/dae_test.xlsx',
                   sheet_name=None,
                   engine='openpyxl'
                   )

T, Y = Rodas(dae,
             [0, 20],
             as_Vars(x),
             as_Vars(y),
             opt=Opt(hinit=0.1))

T1, Y1 = Rodas(dae,
               np.linspace(0, 20, 201),
               as_Vars(x),
               as_Vars(y),
               opt=Opt(hinit=0.1))


# def test_discretize():
#     c = ComputeParam('c')
#     Xk1 = AliasVar(X, 'k1')
#     Yk1 = AliasVar(Y, 'k1')
#     scheme1 = F(X + 1 / 2 * Xk1, Y + 1 / 3 * Yk1, t + dt * c)
#     f1 = Ode(name='F', eqn='(Pm-D*omega)', diff_var='omega')
#
#     param0, eqn0 = f1.discretize(scheme1)
#     assert 'omegak1' in param0.keys()
#     assert 'Dk1' in param0.keys()
#     assert 'Pmk1' in param0.keys()
#     assert eqn0.e_str == 'Pm + 0.333333333333333*Pmk1 - (D + 0.333333333333333*Dk1)*(omega + 0.5*omegak1)'
#
#     param1, eqn1 = f1.discretize(scheme1, param={'D': Param('D'),
#                                                  'Pm': Param('Pm')})
#     assert 'omegak1' in param1.keys()
#     assert 'D' in param1.keys()
#     assert 'Pm' in param1.keys()
#     assert eqn1.e_str == '-D*(omega + 0.5*omegak1) + Pm'
#
#     param2, eqn2 = f1.discretize(scheme1, extra_diff_var=['D'],
#                                  param={'Pm': Param('Pm')})
#     assert 'omegak1' in param2.keys()
#     assert 'Dk1' in param2.keys()
#     assert 'Pm' in param2.keys()
#     assert eqn2.e_str == 'Pm - (D + 0.5*Dk1)*(omega + 0.5*omegak1)'


def test_dae():
    xy_bench = np.asarray(df['rodas'])
    assert np.max(np.abs(xy_bench - Y.array)) < 1e-8
    xy_bench1 = np.asarray(df['rodas_dense'])
    assert np.max(np.abs(xy_bench1 - Y1.array)) < 1e-8
