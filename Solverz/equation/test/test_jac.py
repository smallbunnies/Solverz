from numbers import Number

import numpy as np
from numpy.testing import assert_allclose
import re
import pytest

from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.equations import DAE, AE
from Solverz.equation.param import Param
from Solverz.sym_algebra.symbols import iVar, idx, Para
from Solverz.variable.variables import combine_Vars, as_Vars
from Solverz.equation.jac import JacBlock, Ones
from Solverz.sym_algebra.functions import Diag


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


#%% scalar var and scalar derivative
def test_jb_scalar_var_scalar_deri():
    # non-index var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x'),
                  np.array([1]),
                  slice(1, 2),
                  iVar('y'),
                  np.array([1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == slice(1, 2)
    assert jb.DenDeriExpr == iVar('y') * Ones(3)
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([1, 1, 1]))
    assert jb.SpDeriExpr == iVar('y') * Ones(3)

    # indexed var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x')[1],
                  np.array([1]),
                  slice(1, 3),
                  iVar('y'),
                  np.array([1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == slice(2, 3)
    assert jb.DenDeriExpr == iVar('y') * Ones(3)
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([2, 2, 2]))
    assert jb.SpDeriExpr == iVar('y') * Ones(3)

    # sliced var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x')[1:2],
                  np.array([1]),
                  slice(1, 3),
                  iVar('y'),
                  np.array([1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == slice(2, 3)
    assert jb.DenDeriExpr == iVar('y') * Ones(3)
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([2, 2, 2]))
    assert jb.SpDeriExpr == iVar('y') * Ones(3)


# %% scalar var and vector derivative
def test_jb_scalar_var_vector_deri():
    # non-index var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x'),
                  np.array([1]),
                  slice(1, 2),
                  iVar('y'),
                  np.array([1, 1, 1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == slice(1, 2)
    assert jb.DenDeriExpr == iVar('y')
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([1, 1, 1]))
    assert jb.SpDeriExpr == iVar('y')

    # indexed var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x')[1],
                  np.array([1]),
                  slice(1, 3),
                  iVar('y'),
                  np.array([1, 1, 1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == slice(2, 3)
    assert jb.DenDeriExpr == iVar('y')
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([2, 2, 2]))
    assert jb.SpDeriExpr == iVar('y')

    # sliced var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x')[1:2],
                  np.array([1]),
                  slice(1, 3),
                  iVar('y'),
                  np.array([1, 1, 1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == slice(2, 3)
    assert jb.DenDeriExpr == iVar('y')
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([2, 2, 2]))
    assert jb.SpDeriExpr == iVar('y')

    # Derivative size not compatible with equation size
    with pytest.raises(ValueError, match="Vector derivative size 2 != Equation size 3"):
        jb = JacBlock('a',
                      slice(0, 3),
                      iVar('x')[1:2],
                      np.array([1]),
                      slice(1, 3),
                      iVar('y'),
                      np.array([1, 1]))


# %% vector var and scalar derivative
def test_jb_vector_var_scalar_deri():
    # non-index var
    jb = JacBlock('a',
                  slice(1, 10),
                  iVar('x'),
                  np.ones(9),
                  slice(1, 10),
                  iVar('y'),
                  np.array([1]))

    assert jb.DenEqnAddr == slice(1, 10)
    assert jb.DenVarAddr == slice(1, 10)
    assert jb.DenDeriExpr == Diag(iVar('y') * Ones(9))
    assert_allclose(jb.SpEqnAddr, np.arange(1, 10))
    assert_allclose(jb.SpVarAddr, np.arange(1, 10))
    assert jb.SpDeriExpr == iVar('y') * Ones(9)

    # indexed var
    with pytest.raises(TypeError,
                       match=re.escape("Index of vector variable cant be integer!")):
        jb = JacBlock('a',
                      slice(0, 3),
                      iVar('x')[1],
                      np.array([2, 3, 4]),
                      slice(10, 20),
                      iVar('y')[0],
                      np.array([1]))

    # sliced var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x')[1:4],
                  np.array([2, 3, 4]),
                  slice(10, 20),
                  iVar('y')[0],
                  np.array([1]))
    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == slice(11, 14)
    assert jb.DenDeriExpr == Diag(iVar('y')[0] * Ones(3))
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([11, 12, 13]))
    assert jb.SpDeriExpr == iVar('y')[0] * Ones(3)

    # sliced var with incompatible eqn and var size
    with pytest.raises(ValueError,
                       match=re.escape("Vector variable x[1:3] size 2 != Equation size 3 in scalar derivative case.")):
        jb = JacBlock('a',
                      slice(0, 3),
                      iVar('x')[1:3],
                      np.array([2, 3]),
                      slice(1, 4),
                      iVar('y'),
                      np.array([1]))


# %% vector var and vector derivative
def test_jb_vector_var_vector_deri():
    jb = JacBlock('a',
                  slice(1, 10),
                  iVar('x'),
                  np.ones(9),
                  slice(1, 10),
                  iVar('y'),
                  np.ones(9))

    assert jb.DenEqnAddr == slice(1, 10)
    assert jb.DenVarAddr == slice(1, 10)
    assert jb.DenDeriExpr == Diag(iVar('y'))
    assert_allclose(jb.SpEqnAddr, np.arange(1, 10))
    assert_allclose(jb.SpVarAddr, np.arange(1, 10))
    assert jb.SpDeriExpr == iVar('y')

    # incompatible eqn and var size
    with pytest.raises(ValueError,
                       match=re.escape("Vector variable x[1:3] size 2 != Equation size 3 in vector derivative case.")):
        jb = JacBlock('a',
                      slice(0, 3),
                      iVar('x')[1:3],
                      np.array([2, 3]),
                      slice(1, 4),
                      iVar('y'),
                      np.array([1, 1, 1]))

    # incompatible eqn and derivative size
    with pytest.raises(ValueError,
                       match=re.escape("Vector derivative size 4 != Equation size 3")):
        jb = JacBlock('a',
                      slice(0, 3),
                      iVar('x')[1:4],
                      np.array([2, 3, 4]),
                      slice(1, 4),
                      iVar('y'),
                      np.array([1, 1, 1, 1]))


# %% scalar var and matrix derivative
def test_jb_vector_var_matrix_deri():
    with pytest.raises(TypeError,
                       match=re.escape("Matrix derivative of scalar variables not supported!")):
        jb = JacBlock('a',
                      slice(0, 3),
                      iVar('x')[1],
                      np.array([2]),
                      slice(1, 4),
                      iVar('y'),
                      np.zeros((3, 3)))

    # vector var and matrix derivative
    # compatible vector and matrix size
    with pytest.warns(UserWarning) as record:
        jb = JacBlock('a',
                      slice(0, 3),
                      iVar('x'),
                      np.array([2, 3, 4]),
                      slice(1, 4),
                      iVar('y'),
                      np.zeros((3, 3)))

        assert str(record[0].message) == "Sparse parser of matrix type jac block not implemented!"
        assert jb.DenEqnAddr == slice(0, 3)
        assert jb.DenVarAddr == slice(1, 4)

    # incompatible vector and matrix size
    with pytest.raises(ValueError,
                       match=re.escape("Incompatible matrix derivative size (3, 4) and vector variable size (3,).")):
        jb = JacBlock('a',
                      slice(0, 3),
                      iVar('x'),
                      np.array([2, 3, 4]),
                      slice(1, 4),
                      iVar('y'),
                      np.zeros((3, 4)))
