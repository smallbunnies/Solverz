from numbers import Number

import numpy as np
from numpy.testing import assert_allclose
import re
import pytest

from sympy import Integer

from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.equations import DAE, AE
from Solverz.equation.param import Param
from Solverz.sym_algebra.symbols import iVar, idx, Para
from Solverz.variable.variables import combine_Vars, as_Vars
from Solverz.equation.jac import JacBlock, Ones, Jac
from Solverz.sym_algebra.functions import Diag


def test_jac():
    # test jac element number counter
    jac = Jac()
    x = iVar('x')
    y = iVar('y')
    omega = iVar('omega')
    jac.add_block('a',
                  x[0],
                  JacBlock('a',
                           slice(0, 3),
                           x[0],
                           np.array([1]),
                           slice(1, 2),
                           iVar('y'),
                           np.array([1])))
    jac.add_block('b',
                  omega[1:4],
                  JacBlock('b',
                           slice(3, 12),
                           omega[4:13],
                           np.ones(9),
                           slice(3, 100),
                           iVar('y') ** 2,
                           np.ones(9)))
    jac.add_block('b',
                  omega[5:7],
                  JacBlock('b',
                           slice(0, 3),
                           omega[13],
                           np.ones(1),
                           slice(3, 100),
                           Integer(5),
                           5))
    assert jac.JacEleNum == 15
    row, col, data = jac.parse_row_col_data()
    assert row.size == 15
    assert col.size == 15
    assert data.size == 15
    np.testing.assert_allclose(row, np.concatenate([np.arange(0, 12), np.array([0, 1, 2])]))
    np.testing.assert_allclose(col, np.concatenate([np.array([1, 1, 1]), np.arange(7, 16), np.array([16, 16, 16])]))
    np.testing.assert_allclose(data, np.concatenate([np.zeros(12), np.array([5, 5, 5])]))


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
    assert jb.DenVarAddr == 1
    assert jb.DenDeriExpr == iVar('y') * Ones(3)
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([1, 1, 1]))
    assert jb.SpDeriExpr == iVar('y') * Ones(3)
    assert jb.SpEleSize == 3
    assert jb.IsDeriNumber is False

    # indexed var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x')[1],
                  np.array([1]),
                  slice(1, 3),
                  iVar('y'),
                  np.array([1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == 2
    assert jb.DenDeriExpr == iVar('y') * Ones(3)
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([2, 2, 2]))
    assert jb.SpDeriExpr == iVar('y') * Ones(3)
    assert jb.SpEleSize == 3
    assert jb.IsDeriNumber is False

    # sliced var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x')[1:2],
                  np.array([1]),
                  slice(1, 3),
                  iVar('y'),
                  np.array([1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == 2
    assert jb.DenDeriExpr == iVar('y') * Ones(3)
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([2, 2, 2]))
    assert jb.SpDeriExpr == iVar('y') * Ones(3)
    assert jb.SpEleSize == 3
    assert jb.IsDeriNumber is False


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
    assert jb.DenVarAddr == 1
    assert jb.DenDeriExpr == iVar('y')
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([1, 1, 1]))
    assert jb.SpDeriExpr == iVar('y')
    assert jb.SpEleSize == 3
    assert jb.IsDeriNumber is False

    # indexed var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x')[1],
                  np.array([1]),
                  slice(1, 3),
                  iVar('y'),
                  np.array([1, 1, 1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == 2
    assert jb.DenDeriExpr == iVar('y')
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([2, 2, 2]))
    assert jb.SpDeriExpr == iVar('y')
    assert jb.SpEleSize == 3
    assert jb.IsDeriNumber is False

    # sliced var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x')[1:2],
                  np.array([1]),
                  slice(1, 3),
                  iVar('y'),
                  np.array([1, 1, 1]))

    assert jb.DenEqnAddr == slice(0, 3)
    assert jb.DenVarAddr == 2
    assert jb.DenDeriExpr == iVar('y')
    assert_allclose(jb.SpEqnAddr, np.array([0, 1, 2]))
    assert_allclose(jb.SpVarAddr, np.array([2, 2, 2]))
    assert jb.SpDeriExpr == iVar('y')
    assert jb.SpEleSize == 3
    assert jb.IsDeriNumber is False

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
    assert jb.SpEleSize == 9
    assert jb.IsDeriNumber is False

    # number as derivative
    jb = JacBlock('a',
                  slice(1, 10),
                  iVar('x'),
                  np.ones(9),
                  slice(1, 10),
                  Integer(2),
                  2)

    assert jb.DenEqnAddr == slice(1, 10)
    assert jb.DenVarAddr == slice(1, 10)
    assert jb.DenDeriExpr == Diag(2 * Ones(9))
    assert_allclose(jb.SpEqnAddr, np.arange(1, 10))
    assert_allclose(jb.SpVarAddr, np.arange(1, 10))
    assert jb.SpDeriExpr == 2 * Ones(9)
    assert jb.SpEleSize == 9
    assert jb.IsDeriNumber is True

    jb = JacBlock('a',
                  slice(1, 10),
                  iVar('x'),
                  np.ones(9),
                  slice(1, 10),
                  Integer(2),
                  np.array([2]))

    assert jb.DenEqnAddr == slice(1, 10)
    assert jb.DenVarAddr == slice(1, 10)
    assert jb.DenDeriExpr == Diag(2 * Ones(9))
    assert_allclose(jb.SpEqnAddr, np.arange(1, 10))
    assert_allclose(jb.SpVarAddr, np.arange(1, 10))
    assert jb.SpDeriExpr == 2 * Ones(9)
    assert jb.SpEleSize == 9
    assert jb.IsDeriNumber is True

    jb = JacBlock('a',
                  slice(1, 10),
                  iVar('x'),
                  np.ones(9),
                  slice(1, 10),
                  Integer(2),
                  np.array(2))

    assert jb.DenEqnAddr == slice(1, 10)
    assert jb.DenVarAddr == slice(1, 10)
    assert jb.DenDeriExpr == Diag(2 * Ones(9))
    assert_allclose(jb.SpEqnAddr, np.arange(1, 10))
    assert_allclose(jb.SpVarAddr, np.arange(1, 10))
    assert jb.SpDeriExpr == 2 * Ones(9)
    assert jb.SpEleSize == 9
    assert jb.IsDeriNumber is True

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
    assert jb.SpEleSize == 3
    assert jb.IsDeriNumber is False

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

def test_jb_vector_var_zero_deri():

    with pytest.raises(ValueError,
                       match=re.escape("We wont allow 0.0 derivative!")):
        jb = JacBlock('a',
                      slice(0, 3),
                      iVar('x'),
                      np.ones(3),
                      slice(1, 4),
                      0.,
                      np.array([0]))

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
    assert jb.SpEleSize == 9
    assert jb.IsDeriNumber is False

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
