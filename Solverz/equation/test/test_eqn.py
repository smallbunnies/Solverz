from numbers import Number

import pandas as pd
import numpy as np
import sympy as sp

from Solverz.sym_algebra.symbols import iVar, idx, Para
from Solverz.sym_algebra.functions import Abs
from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.equations import DAE, AE
from Solverz.variable.variables import combine_Vars, as_Vars
from Solverz import AntiWindUp, Param, Var

x = iVar('x', value=[1])
f = Eqn('f', x - 1)
f1 = f.NUM_EQN(1)
assert isinstance(f1, Number)

x = iVar('x', value=[1, 2])
f = Eqn('f', x - 1)
f1 = f.NUM_EQN(np.array([3, 4]))
assert isinstance(f1, np.ndarray)
assert f1.ndim == 1


def test_discarding_zero_deri():
    # discard zero derivative
    u = Var('u', [1, 1, 1])
    e = Var('e', [1, 1, 1])
    umin = Param('umin', [0,0,0])
    umax = Param('umax', [5,5,5])
    F = Ode('Anti', AntiWindUp(u, umin, umax, e), u)
    F.derive_derivative()
    assert 'u' not in F.derivatives

def test_Var_converter():
    # convert Var/Param to iVar/Para objects
    f = Eqn('f', Var('x'))
    assert f.SYMBOLS['x'].__str__() == 'x'
    f = Eqn('f', Param('x'))
    assert f.SYMBOLS['x'].__str__() == 'x'
    g = Ode('g', Var('x'), diff_var=Var('y'))
    assert g.SYMBOLS['x'].__str__() == 'x'
    assert g.diff_var.name == 'y'

