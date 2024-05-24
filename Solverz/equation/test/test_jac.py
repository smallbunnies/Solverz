from numbers import Number

import numpy as np

from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.equations import DAE, AE
from Solverz.equation.param import Param
from Solverz.sym_algebra.symbols import iVar, idx, Para
from Solverz.variable.variables import combine_Vars, as_Vars


def test_jac():
    x = iVar('x', 1)
    y = iVar('y', 1)

    f = Ode(name='f', f=-x ** 3 + 0.5 * y ** 2, diff_var=x)
    g = Eqn(name='g', eqn=x ** 2 + y ** 2 - 2)
    dae = DAE([f, g])

    z = combine_Vars(as_Vars(x), as_Vars(y))
    fxy = dae.fy(None, z)
    gxy = dae.gy(None, z)
    assert fxy[0][3].shape == (1,)
    assert np.all(np.isclose(fxy[0][3], [-3.]))
    assert fxy[1][3].shape == (1,)
    assert np.all(np.isclose(fxy[1][3], [1]))
    assert gxy[0][3].shape == (1,)
    assert np.all(np.isclose(gxy[0][3], [2]))
    assert gxy[1][3].shape == (1,)
    assert np.all(np.isclose(gxy[1][3], [2]))

    x = iVar('x', [1, 2, 3])
    i = idx('i', value=[0, 2])

    f = Eqn('f', eqn=x)
    ae = AE(f)
    gy = ae.gy(as_Vars(x))
    assert isinstance(gy[0][3], Number)
    # assert gy[0][3].ndim == 0
    assert np.all(np.isclose(gy[0][3], 1))

    f = Eqn('f', eqn=x[i])
    ae = AE(f)
    gy = ae.gy(as_Vars(x))
    assert isinstance(gy[0][3], Number)
    # assert gy[0][3].ndim == 0
    assert np.all(np.isclose(gy[0][3], 1))

    f = Eqn('f', eqn=x[i] ** 2)
    ae = AE(f)
    gy = ae.gy(as_Vars(x))
    assert isinstance(gy[0][3], np.ndarray)
    assert gy[0][3].ndim == 1
    assert np.all(np.isclose(gy[0][3], [2., 6.]))

    A_v = np.random.rand(3, 3)
    A = Para('A', dim=2)
    f = Eqn('f', eqn=A * x)
    ae = AE(f)
    ae.param_initializer('A', Param('A', value=A_v, dim=2))
    gy = ae.gy(as_Vars(x))
    assert isinstance(gy[0][3], np.ndarray)
    assert gy[0][3].ndim == 2
    np.testing.assert_allclose(gy[0][3], A_v)

