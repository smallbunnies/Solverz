from sympy import pycode
from numbers import Number

import numpy as np
from numpy.testing import assert_allclose
import re
import pytest
from sympy.codegen.ast import Assignment, AddAugmentedAssignment

from Solverz import Model, Var, AliasVar, sin
from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.equations import DAE, AE
from Solverz.equation.param import Param
from Solverz.sym_algebra.symbols import iVar, idx, Para
from Solverz.variable.variables import combine_Vars, as_Vars
from Solverz.equation.jac import JacBlock, Jac
from Solverz.equation.hvp import Hvp
from Solverz.sym_algebra.functions import *
from Solverz.code_printer.python.inline.inline_printer import print_J_block, extend, SolList, print_J_blocks, print_J, \
    print_F, print_Hvp, made_numerical, Solverzlambdify
from Solverz.utilities.address import Address
from Solverz.num_api.module_parser import modules


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
    assert symJb[0] == AddAugmentedAssignment(
        J_[0:3, 1], iVar('y') * Ones(3))


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
    assert symJb[0] == AddAugmentedAssignment(
        iVar('J_', internal_use=True)[1:10, 0:9], Diag(iVar('y')))


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
    assert symJbs[0] == AddAugmentedAssignment(J_[0:3, 1], y * Ones(3))
    assert symJbs[1] == AddAugmentedAssignment(J_[3:12, 7:16], Diag(y ** 2))


expected_J_den = """def J_(t, y_, p_):
    h = y_[0:1]
    v = y_[1:2]
    g = p_["g"]
    J_ = np.zeros((2, 2))
    J_[0:1,1] += np.ones(1)
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
    data.extend(np.ones(1))
    return sps.coo_array((data, (row, col)), (2, 2)).tocsc()
""".strip()

expected_F = """def F_(t, y_, p_):
    h = y_[0:1]
    v = y_[1:2]
    g = p_["g"]
    _F_ = np.zeros((2, ))
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
    bball.FormJac(y0)
    assert print_J(bball.__class__.__name__,
                   bball.jac,
                   bball.a,
                   bball.var_address,
                   bball.PARAM,
                   bball.nstep) == expected_J_den
    assert print_J(bball.__class__.__name__,
                   bball.jac,
                   bball.a,
                   bball.var_address,
                   bball.PARAM,
                   bball.nstep,
                   True) == expected_J_sp
    assert print_F(bball.__class__.__name__,
                   bball.EQNs,
                   bball.a,
                   bball.var_address,
                   bball.PARAM,
                   bball.nstep) == expected_F
    nbball = made_numerical(bball, y0, sparse=True)
    np.testing.assert_allclose(nbball.F(0, y0, nbball.p),
                               np.array([20, -9.8]))
    np.testing.assert_allclose(nbball.J(0, y0, nbball.p).toarray(),
                               np.array([[0., 1.], [0., 0.]]))
    nbball = made_numerical(bball, y0, sparse=False)
    np.testing.assert_allclose(nbball.J(0, y0, nbball.p),
                               np.array([[0., 1.], [0., 0.]]))


expected_J_den_fdae = """def J_(t, y_, p_, y_0):
    p = y_[0:3]
    q = y_[3:6]
    p_tag_0 = y_0[0:3]
    q_tag_0 = y_0[3:6]
    J_ = np.zeros((6, 6))
    J_[0:2,1:3] += np.diagflat(np.ones(2))
    J_[2:4,0:2] += np.diagflat(-np.ones(2))
    J_[4:5,2] += -np.ones(1)
    J_[5:6,2] += np.ones(1)
    return J_
""".strip()

expected_J_sp_fdae = """def J_(t, y_, p_, y_0):
    p = y_[0:3]
    q = y_[3:6]
    p_tag_0 = y_0[0:3]
    q_tag_0 = y_0[3:6]
    row = []
    col = []
    data = []
    row.extend([0, 1])
    col.extend([1, 2])
    data.extend(np.ones(2))
    row.extend([2, 3])
    col.extend([0, 1])
    data.extend(-np.ones(2))
    row.extend([4])
    col.extend([2])
    data.extend(-np.ones(1))
    row.extend([5])
    col.extend([2])
    data.extend(np.ones(1))
    return sps.coo_array((data, (row, col)), (6, 6)).tocsc()
""".strip()

expected_F_fdae = """def F_(t, y_, p_, y_0):
    p = y_[0:3]
    q = y_[3:6]
    p_tag_0 = y_0[0:3]
    q_tag_0 = y_0[3:6]
    _F_ = np.zeros((6, ))
    _F_[0:2] = -p_tag_0[0:2] + p[1:3]
    _F_[2:4] = p_tag_0[1:3] - p[0:2]
    _F_[4:5] = p_tag_0[0] - p[2]
    _F_[5:6] = -p_tag_0[0] + p[2]
    return _F_
""".strip()


def test_print_F_J_FDAE():
    m = Model()
    m.p = Var('p', value=[2, 3, 4])
    m.q = Var('q', value=[0, 1, 2])
    m.p0 = AliasVar('p', init=m.p)
    m.q0 = AliasVar('q', init=m.q)

    m.ae1 = Eqn('cha1',
                m.p[1:3] - m.p0[0:2])

    m.ae2 = Eqn('cha2',
                m.p0[1:3] - m.p[0:2])
    m.ae3 = Eqn('cha3',
                m.p0[0] - m.p[2])
    m.ae4 = Eqn('cha4',
                m.p[2] - m.p0[0])
    fdae, y0 = m.create_instance()
    fdae.FormJac(y0)
    assert print_J(fdae.__class__.__name__,
                   fdae.jac,
                   fdae.a,
                   fdae.var_address,
                   fdae.PARAM,
                   fdae.nstep) == expected_J_den_fdae
    assert print_J(fdae.__class__.__name__,
                   fdae.jac,
                   fdae.a,
                   fdae.var_address,
                   fdae.PARAM,
                   fdae.nstep,
                   True) == expected_J_sp_fdae
    assert print_F(fdae.__class__.__name__,
                   fdae.EQNs,
                   fdae.a,
                   fdae.var_address,
                   fdae.PARAM,
                   fdae.nstep) == expected_F_fdae


def test_made_numerical():
    x = iVar('x', [1, 1])
    f1 = Eqn('f1', 2 * x[0] + x[1])
    f2 = Eqn('f2', x[0] ** 2 + sin(x[1]))

    F = AE([f1, f2])
    y = as_Vars([x])
    nF, code = made_numerical(F, y, sparse=True, output_code=True)
    F0 = nF.F(y, nF.p)
    J0 = nF.J(y, nF.p)
    np.testing.assert_allclose(F0, np.array(
        [2 * 1 + 1, 1 + np.sin(1)]), rtol=1e-8)
    np.testing.assert_allclose(J0.toarray(), np.array(
        [[2, 1], [2, 0.54030231]]), rtol=1e-8)


def test_hvp_printer():
    jac = Jac()
    x = iVar("x")
    jac.add_block(
        "a",
        x[0],
        JacBlock(
            "a",
            slice(0, 1),
            x[0],
            np.array([1]),
            slice(0, 2),
            exp(x[0]),
            np.array([2.71828183]),
        ),
    )
    jac.add_block(
        "a",
        x[1],
        JacBlock(
            "a",
            slice(0, 1),
            x[1],
            np.array([1]),
            slice(0, 2),
            cos(x[1]),
            np.array([0.54030231]),
        ),
    )
    jac.add_block(
        "b",
        x[0],
        JacBlock("b",
                 slice(1, 2),
                 x[0],
                 np.ones(1),
                 slice(0, 2),
                 1,
                 np.array([1])),
    )
    jac.add_block(
        "b",
        x[1],
        JacBlock("b", slice(1, 2), x[1], np.ones(1),
                 slice(0, 2), 2 * x[1], np.array([2])),
    )

    h = Hvp(jac)
    eqn_addr = Address()
    eqn_addr.add('a', 1)
    eqn_addr.add('b', 1)
    var_addr = Address()
    var_addr.add('x', 2)

    code = print_Hvp('AE',
                     h,
                     eqn_addr,
                     var_addr,
                     dict())

    HVP = Solverzlambdify(code, 'Hvp_', modules)
    np.testing.assert_allclose(HVP(np.array([1, 2]), dict(), np.array([1, 1])).toarray(),
                               np.array([[2.71828183, -0.90929743],
                                         [0.,  2.]]))
