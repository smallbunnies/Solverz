from sympy import pycode
from numbers import Number

import numpy as np
from numpy.testing import assert_allclose
import re
import pytest
from sympy import sin
from sympy.codegen.ast import Assignment, AddAugmentedAssignment

from Solverz import Model, Var
from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.equations import DAE, AE
from Solverz.equation.param import Param
from Solverz.sym_algebra.symbols import iVar, idx, Para
from Solverz.variable.variables import combine_Vars, as_Vars
from Solverz.equation.jac import JacBlock, Ones, Jac
from Solverz.sym_algebra.functions import Diag
from Solverz.code_printer.python.inline.inline_printer import print_J_block, extend, SolList, print_J_blocks, print_J, \
    print_F, made_numerical

# %%
row = iVar('row', internal_use=True)
col = iVar('col', internal_use=True)
data = iVar('data', internal_use=True)
J_ = iVar('J_', internal_use=True)


def test_jb_printer_scalar_var_scalar_deri():
    # non-index var
    jb = JacBlock('a',
                  slice(0, 3),
                  iVar('x'),
                  np.array([1]),
                  slice(1, 2),
                  iVar('y'),
                  np.array([1]))

    symJb = print_J_block(jb, True)
    assert symJb[0] == extend(row, SolList(0, 1, 2))
    assert symJb[1] == extend(col, SolList(1, 1, 1))
    assert symJb[2] == extend(data, iVar('y') * Ones(3))
    symJb = print_J_block(jb, False)
    assert symJb[0] == AddAugmentedAssignment(J_[0:3, 1:2], iVar('y') * Ones(3))


def test_jb_printer_vector_var_vector_deri():
    jb = JacBlock('a',
                  slice(1, 10),
                  iVar('x'),
                  np.ones(9),
                  slice(0, 9),
                  iVar('y'),
                  np.ones(9))
    symJb = print_J_block(jb, True)
    assert symJb[0] == extend(row, SolList(*np.arange(1, 10).tolist()))
    assert symJb[1] == extend(col, SolList(*np.arange(0, 9).tolist()))
    assert symJb[2] == extend(data, iVar('y'))
    symJb = print_J_block(jb, False)
    assert symJb[0] == AddAugmentedAssignment(iVar('J_', internal_use=True)[1:10, 0:9], Diag(iVar('y')))


def test_jbs_printer():
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
    symJbs = print_J_blocks(jac, True)
    assert symJbs[0] == extend(row, SolList(0, 1, 2))
    assert symJbs[1] == extend(col, SolList(1, 1, 1))
    assert symJbs[2] == extend(data, y * Ones(3))
    assert symJbs[3] == extend(row, SolList(3, 4, 5, 6, 7, 8, 9, 10, 11))
    assert symJbs[4] == extend(col, SolList(7, 8, 9, 10, 11, 12, 13, 14, 15))
    assert symJbs[5] == extend(data, y ** 2)
    symJbs = print_J_blocks(jac, False)
    assert symJbs[0] == AddAugmentedAssignment(J_[0:3, 1:2], y * Ones(3))
    assert symJbs[1] == AddAugmentedAssignment(J_[3:12, 7:16], Diag(y ** 2))


expected_J_den = """def J_(t, y_, p_):
    h = y_[0:1]
    v = y_[1:2]
    g = p_["g"]
    J_ = zeros((2, 2))
    J_[0:1,1:2] += ones(1)
    return J_
""".strip()

expected_J_sp = """def J_(t, y_, p_):
    h = y_[0:1]
    v = y_[1:2]
    g = p_["g"]
    row = []
    col = []
    data = []
    row.extend([0])
    col.extend([1])
    data.extend(ones(1))
    return coo_array((data, (row, col)), (2, 2)).tocsc()
""".strip()

expected_F = """def F_(t, y_, p_):
    h = y_[0:1]
    v = y_[1:2]
    g = p_["g"]
    _F_ = zeros((2, ))
    _F_[0:1] = v
    _F_[1:2] = g
    return _F_
""".strip()


def test_print_F_J():
    m = Model()
    m.h = Var('h', 0)
    m.v = Var('v', 20)
    m.g = Param('g', -9.8)
    m.f1 = Ode('f1', f=m.v, diff_var=m.h)
    m.f2 = Ode('f2', f=m.g, diff_var=m.v)
    bball, y0 = m.create_instance()
    assert print_J(bball) == expected_J_den
    assert print_J(bball, True) == expected_J_sp
    assert print_F(bball) == expected_F
    nbball = made_numerical(bball, y0, sparse=True)
    np.testing.assert_allclose(nbball.F(0, y0, nbball.p),
                               np.array([20, -9.8]))
    np.testing.assert_allclose(nbball.J(0, y0, nbball.p).toarray(),
                               np.array([[0., 1.], [0., 0.]]))
    nbball = made_numerical(bball, y0, sparse=False)
    np.testing.assert_allclose(nbball.J(0, y0, nbball.p),
                               np.array([[0., 1.], [0., 0.]]))


def test_made_numerical():
    x = iVar('x', [1, 1])
    f1 = Eqn('f1', 2 * x[0] + x[1])
    f2 = Eqn('f2', x[0] ** 2 + sin(x[1]))

    F = AE([f1, f2])
    y = as_Vars([x])
    nF, code = made_numerical(F, y, sparse=True, output_code=True)
    F0 = nF.F(y, nF.p)
    J0 = nF.J(y, nF.p)
    np.testing.assert_allclose(F0, np.array([2 * 1 + 1, 1 + np.sin(1)]), rtol=1e-8)
    np.testing.assert_allclose(J0.toarray(), np.array([[2, 1], [2, 0.54030231]]), rtol=1e-8)
