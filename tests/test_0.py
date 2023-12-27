import numpy as np

from Solverz import Eqn, AE, nr_method, as_Vars, Var, nr_method_numerical

x = Var(name='x', value=2)

e = Eqn(name='e', eqn=x ** 2 - 1)
f = AE(name='F',
       eqn=e)
y = as_Vars(x)
y = nr_method(f, y)

from Solverz.numerical_interface.code_printer import print_F, print_J, Solverzlambdify
from Solverz.numerical_interface.num_eqn import nAE, parse_ae_v

code_g = print_F(f)
g = Solverzlambdify(code_g, 'F_', modules=['numpy'])
code_J = print_J(f)
from scipy.sparse import csc_array

J = Solverzlambdify(code_J, 'J_', modules=[{'csc_array': csc_array}, 'numpy'])

f1 = nAE(1, lambda z, p: g(0, z, p), lambda z, p: J(0, z, p), dict())
y1 = nr_method_numerical(f1, np.array([2]))
y1 = parse_ae_v(y1, f.var_address)


def test_nr_method():
    assert abs(y['x'] - 1) <= 1e-8
    assert abs(y1['x'] - 1) <= 1e-8
