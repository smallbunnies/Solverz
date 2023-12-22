import numpy as np

from Solverz import Eqn, AE, nr_method, as_Vars, Var

x = Var(name='x', value=2)

e = Eqn(name='e', eqn=x ** 2 - 1)
f = AE(name='F',
       eqn=e)
y = as_Vars(x)
y = nr_method(f, y)

from Solverz.numerical_interface.num_eqn import print_g, print_J, Solverzlambdify, nAE
from Solverz.numerical_interface.custom_function import solve

code_g = print_g(f)
g = Solverzlambdify(code_g, 'F_', modules=['numpy'])
code_J = print_J(f)
from scipy.sparse import csc_array

J = Solverzlambdify(code_J, 'J_', modules=[{'csc_array': csc_array}, 'numpy'])


def nr_method1(eqn: nAE,
               y: np.ndarray,
               p,
               tol: float = 1e-8,
               stats=False):
    df = eqn.g(y, p)
    ite = 0
    while max(abs(df)) > tol:
        ite = ite + 1
        y = y - solve(eqn.J(y, p), df)
        df = eqn.g(y, p)
        if ite >= 100:
            print(f"Cannot converge within 100 iterations. Deviation: {max(abs(df))}!")
            break
    if not stats:
        return y
    else:
        return y, ite


from Solverz.numerical_interface.num_eqn import nAE

f1 = nAE(1, lambda z, p: g(0, z, p), lambda z, p: J(0, z, p), f.var_address)
y1 = nr_method1(f1, np.array([2]), 1)
y1 = f1.parse_v(y1)


def test_nr_method():
    assert abs(y['x'] - 1) <= 1e-8
    assert abs(y1['x'] - 1) <= 1e-8
