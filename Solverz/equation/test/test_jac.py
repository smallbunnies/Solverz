import pandas as pd
import numpy as np
import sympy as sp

from Solverz.symboli_algebra.symbols import Var, idx, Para
from Solverz.symboli_algebra.functions import Abs
from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.equations import DAE, AE
from Solverz.variable.variables import combine_Vars, as_Vars

x = Var('x', 1)
y = Var('y', 1)

f = Ode(name='f', f=-x ** 3 + 0.5 * y ** 2, diff_var=x)
g = Eqn(name='g', eqn=x ** 2 + y ** 2 - 2)
dae = DAE([f, g])

z = combine_Vars(as_Vars(x), as_Vars(y))
fxy = dae.f_xy(None, z)
gxy = dae.g_xy(None, z)
assert fxy[0][3].shape == (1,)
assert np.all(np.isclose(fxy[0][3], [-3.]))
assert fxy[1][3].shape == (1,)
assert np.all(np.isclose(fxy[1][3], [1]))
assert gxy[0][3].shape == (1,)
assert np.all(np.isclose(gxy[0][3], [2]))
assert gxy[1][3].shape == (1,)
assert np.all(np.isclose(gxy[1][3], [2]))

x = Var('x', [1, 2, 3])
i = idx('i', value=[0, 2])

f = Eqn('f', eqn=x)
ae = AE(f)
gy = ae.g_y(as_Vars(x))
assert isinstance(gy[0][3], np.ndarray)
assert gy[0][3].ndim == 0
assert np.all(np.isclose(gy[0][3], 1))

f = Eqn('f', eqn=x[i])
ae = AE(f)
gy = ae.g_y(as_Vars(x))
assert isinstance(gy[0][3], np.ndarray)
assert gy[0][3].ndim == 0
assert np.all(np.isclose(gy[0][3], 1))

f = Eqn('f', eqn=x[i] ** 2)
ae = AE(f)
gy = ae.g_y(as_Vars(x))
assert isinstance(gy[0][3], np.ndarray)
assert gy[0][3].ndim == 1
assert np.all(np.isclose(gy[0][3], [2., 6.]))


A = Para('A', np.random.rand(3, 3), dim=2)
f = Eqn('f', eqn=A * x)
ae = AE(f)
gy = ae.g_y(as_Vars(x))
assert isinstance(gy[0][3], np.ndarray)
assert gy[0][3].ndim == 2
assert np.all(np.isclose(gy[0][3], A.value))

