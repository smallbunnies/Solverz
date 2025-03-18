from sympy import Integer
from Solverz import Ode, DAE, made_numerical
from Solverz.sym_algebra.symbols import iVar
from Solverz.variable.variables import as_Vars

def test_zero_index():
    # to test if index zero, being the sympy.Integer.instance, raises error when creating the singular mass mat
    u = iVar('u', [3])
    zero = Integer(0)
    f = Ode('f', u[zero], u[zero])
    sdae = DAE(f)
    y = as_Vars(u)
    dae = made_numerical(sdae, y)


def test_ode():
    # The RHS is a scalar zero, but the equation is of size 3
    x = iVar('x', [1, 2, 3])
    f = Ode('f', 0, x[0:3])
    sdae = DAE(f)
    y = as_Vars(x)
    sdae.assign_eqn_var_address(y)
    assert sdae.eqn_size == 3
