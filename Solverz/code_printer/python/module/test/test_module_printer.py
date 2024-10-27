import pytest
import numpy as np
import re

from sympy import symbols, pycode, Integer
from sympy.codegen.ast import FunctionCall as SpFuncCall, Assignment

from Solverz.sym_algebra.symbols import iVar, Para
from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.param import Param, TimeSeriesParam
from Solverz.equation.jac import Jac, JacBlock
from Solverz.equation.hvp import Hvp
from Solverz.sym_algebra.functions import exp, cos
from Solverz.utilities.address import Address
from sympy.codegen.ast import FunctionDefinition, Return
from Solverz.code_printer.python.module.module_printer import (print_J, print_inner_J, print_F,
                                                               print_inner_F, print_sub_inner_F,
                                                               print_Hvp, print_inner_Hvp)

expected_AE_Hvp = """def Hvp_(y_, p_, v_):
    omega = y_[0:10]
    delta = y_[10:15]
    x = y_[15:18]
    y = y_[18:25]
    ax = p_["ax"]
    lam = p_["lam"]
    ax = ax_trigger_func(x)
    data_hvp = inner_Hvp(_data_hvp, v_, omega, delta, x, y, ax, lam)
    return sps.coo_array((data_hvp, (row_hvp, col_hvp)), (25, 25)).tocsc()
""".strip()
expected_FDAE_Hvp = """def Hvp_(t, y_, p_, v_, y_0):
    omega = y_[0:10]
    delta = y_[10:15]
    x = y_[15:18]
    y = y_[18:25]
    omega_tag_0 = y_0[0:10]
    delta_tag_0 = y_0[10:15]
    x_tag_0 = y_0[15:18]
    y_tag_0 = y_0[18:25]
    ax = p_["ax"]
    lam = p_["lam"]
    ax = ax_trigger_func(x)
    data_hvp = inner_Hvp(_data_hvp, v_, omega, delta, x, y, omega_tag_0, delta_tag_0, x_tag_0, y_tag_0, ax, lam)
    return sps.coo_array((data_hvp, (row_hvp, col_hvp)), (25, 25)).tocsc()
""".strip()


def test_print_Hvp():
    eqs_type = 'AE'
    VarAddr = Address()
    VarAddr.add('omega', 10)
    VarAddr.add('delta', 5)
    VarAddr.add('x', 3)
    VarAddr.add('y', 7)
    Pdict = dict()
    ax = Param('ax',
               [0, 1],
               triggerable=True,
               trigger_var=['x'],
               trigger_fun=np.sin)
    Pdict['ax'] = ax
    lam = Param('lam', [1, 1, 1])
    Pdict["lam"] = lam

    with pytest.raises(ValueError, match=re.escape("Hvp matrix, with size (20*25), not square")):
        print_Hvp(eqs_type,
                  20,
                  VarAddr,
                  Pdict)

    assert print_Hvp(eqs_type,
                     25,
                     VarAddr,
                     Pdict) == expected_AE_Hvp

    assert print_Hvp("FDAE",
                     25,
                     VarAddr,
                     Pdict,
                     nstep=1) == expected_FDAE_Hvp


expected_inner_Hvp = """def inner_Hvp(_data_hvp, v_, x, c):
    _data_hvp[0:1] = inner_Hvp0(v_, x)
    _data_hvp[1:2] = inner_Hvp1(v_, x)
    _data_hvp[2:3] = inner_Hvp2(c, v_)
    return _data_hvp
""".strip()
expected_inner_Hvp0 = """def inner_Hvp0(v_, x):
    return v_[0]*np.exp(x[0])*np.ones(1)
""".strip()
expected_inner_Hvp1 = """def inner_Hvp1(v_, x):
    return -v_[1]*sin(x[1])*np.ones(1)
""".strip()
expected_inner_Hvp2 = """def inner_Hvp2(c, v_):
    return v_[1]*c*np.ones(1)
""".strip()


def test_print_inner_Hvp():
    jac = Jac()
    x = iVar("x")
    c = Para('c')
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
                 Integer(1),
                 np.array([1])),
    )
    jac.add_block(
        "b",
        x[1],
        JacBlock("b", slice(1, 2), x[1], np.ones(1), slice(0, 2), c * x[1], np.array([2])),
    )

    Pdict = {"c": Param("c", np.array([2.0]))}
    VarAddr = Address()
    VarAddr.add('x', 2)

    h = Hvp(jac)
    code_dict = print_inner_Hvp(VarAddr,
                                Pdict,
                                h)

    assert code_dict['code_inner_Hvp'] == expected_inner_Hvp
    assert code_dict['code_sub_inner_Hvp'][0] == expected_inner_Hvp0
    assert code_dict['code_sub_inner_Hvp'][1] == expected_inner_Hvp1
    assert code_dict['code_sub_inner_Hvp'][2] == expected_inner_Hvp2


expected = """def J_(t, y_, p_):
    omega = y_[0:10]
    delta = y_[10:15]
    x = y_[15:18]
    y = y_[18:25]
    ax = p_["ax"]
    lam = p_["lam"]
    G6 = p_["G6"].get_v_t(t)
    ax = ax_trigger_func(x)
    data = inner_J(_data_, omega, delta, x, y, ax, lam, G6)
    return sps.coo_array((data, (row, col)), (25, 25)).tocsc()
""".strip()
expected1 = """def J_(t, y_, p_, y_0):
    omega = y_[0:10]
    delta = y_[10:15]
    x = y_[15:18]
    y = y_[18:25]
    omega_tag_0 = y_0[0:10]
    delta_tag_0 = y_0[10:15]
    x_tag_0 = y_0[15:18]
    y_tag_0 = y_0[18:25]
    ax = p_["ax"]
    lam = p_["lam"]
    G6 = p_["G6"].get_v_t(t)
    ax = ax_trigger_func(x)
    data = inner_J(_data_, omega, delta, x, y, omega_tag_0, delta_tag_0, x_tag_0, y_tag_0, ax, lam, G6)
    return sps.coo_array((data, (row, col)), (25, 25)).tocsc()
""".strip()


def test_print_J():
    eqs_type = 'DAE'
    VarAddr = Address()
    VarAddr.add('omega', 10)
    VarAddr.add('delta', 5)
    VarAddr.add('x', 3)
    VarAddr.add('y', 7)
    Pdict = dict()
    ax = Param('ax',
               [0, 1],
               triggerable=True,
               trigger_var=['x'],
               trigger_fun=np.sin)
    Pdict['ax'] = ax
    lam = Param('lam', [1, 1, 1])
    Pdict["lam"] = lam
    G6 = TimeSeriesParam('G6',
                         [0, 1, 2, 3],
                         [0, 100, 200, 300])
    Pdict["G6"] = G6

    with pytest.raises(ValueError, match=re.escape("Jac matrix, with size (20*25), not square")):
        print_J(eqs_type,
                20,
                VarAddr,
                Pdict)

    assert print_J(eqs_type,
                   25,
                   VarAddr,
                   Pdict) == expected

    assert print_J("FDAE",
                   25,
                   VarAddr,
                   Pdict,
                   1) == expected1


expected2 = """def inner_J(_data_, omega, x, y, z, ax, lam, G6):
    _data_[0:3] = inner_J0(y)
    _data_[3:12] = inner_J1(y)
    return _data_
""".strip()
expected3 = """def inner_J0(y):
    return y*np.ones(3)
""".strip()
expected4 = """def inner_J1(y):
    return y**2
""".strip()
expected5 = """def inner_J(_data_, omega, x, y, z, omega_tag_0, x_tag_0, y_tag_0, z_tag_0, ax, lam, G6):
    _data_[0:3] = inner_J0(y)
    _data_[3:12] = inner_J1(y)
    return _data_
""".strip()


def test_print_inner_J():
    jac = Jac()
    x = iVar('x')
    y = iVar('y')
    z = iVar('z')
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
                  y,
                  JacBlock('b',
                           slice(3, 12),
                           z,
                           np.ones(9),
                           slice(100, 109),
                           Integer(1),
                           np.ones(1)))

    VarAddr = Address()
    VarAddr.add('omega', 97)
    VarAddr.add('x', 1)
    VarAddr.add('y', 1)
    VarAddr.add('z', 9)
    Pdict = dict()
    ax = Param('ax',
               [0, 1],
               triggerable=True,
               trigger_var=['x'],
               trigger_fun=np.sin)
    Pdict['ax'] = ax
    lam = Param('lam', [1, 1, 1])
    Pdict["lam"] = lam
    G6 = TimeSeriesParam('G6',
                         [0, 1, 2, 3],
                         [0, 100, 200, 300])
    Pdict["G6"] = G6

    code_dict = print_inner_J(VarAddr,
                              Pdict,
                              jac)

    assert code_dict['code_inner_J'] == expected2
    assert code_dict['code_sub_inner_J'][0] == expected3
    assert code_dict['code_sub_inner_J'][1] == expected4

    code_dict = print_inner_J(VarAddr,
                              Pdict,
                              jac,
                              1)

    assert code_dict['code_inner_J'] == expected5


expected6 = """def F_(t, y_, p_):
    omega = y_[0:10]
    delta = y_[10:15]
    x = y_[15:18]
    y = y_[18:25]
    ax = p_["ax"]
    lam = p_["lam"]
    G6 = p_["G6"].get_v_t(t)
    ax = ax_trigger_func(x)
    return inner_F(_F_, omega, delta, x, y, ax, lam, G6)
""".strip()


def test_print_F():
    eqs_type = 'DAE'
    VarAddr = Address()
    VarAddr.add('omega', 10)
    VarAddr.add('delta', 5)
    VarAddr.add('x', 3)
    VarAddr.add('y', 7)
    Pdict = dict()
    ax = Param('ax',
               [0, 1],
               triggerable=True,
               trigger_var=['x'],
               trigger_fun=np.sin)
    Pdict['ax'] = ax
    lam = Param('lam', [1, 1, 1])
    Pdict["lam"] = lam
    G6 = TimeSeriesParam('G6',
                         [0, 1, 2, 3],
                         [0, 100, 200, 300])
    Pdict["G6"] = G6
    assert print_F(eqs_type,
                   VarAddr,
                   Pdict) == expected6


expected7 = """def inner_F(_F_, omega, delta, x, y, ax, lam, G6):
    _F_[0:10] = inner_F0(delta, omega)
    _F_[10:15] = inner_F1(lam, x, y)
    return _F_
""".strip()
expected8 = """def inner_F(_F_, omega, delta, x, y, omega_tag_0, delta_tag_0, x_tag_0, y_tag_0, ax, lam, G6):
    _F_[0:10] = inner_F0(delta, omega)
    _F_[10:15] = inner_F1(lam, x, y)
    return _F_
""".strip()


def test_print_inner_F():
    EQNs = dict()
    omega = iVar('omega')
    delta = iVar('delta')
    x = iVar('x')
    y = iVar('y')
    lam = Para('lam')
    EQNs['a'] = Eqn('a', omega * delta)
    EQNs['b'] = Ode('b', x + y * lam, diff_var=x)
    EqnAddr = Address()
    EqnAddr.add('a', 10)
    EqnAddr.add('b', 5)
    VarAddr = Address()
    VarAddr.add('omega', 10)
    VarAddr.add('delta', 5)
    VarAddr.add('x', 3)
    VarAddr.add('y', 7)
    Pdict = dict()
    ax = Param('ax',
               [0, 1],
               triggerable=True,
               trigger_var=['x'],
               trigger_fun=np.sin)
    Pdict['ax'] = ax
    lam = Param('lam', [1, 1, 1])
    Pdict["lam"] = lam
    G6 = TimeSeriesParam('G6',
                         [0, 1, 2, 3],
                         [0, 100, 200, 300])
    Pdict["G6"] = G6
    assert print_inner_F(EQNs,
                         EqnAddr,
                         VarAddr,
                         Pdict) == expected7

    assert print_inner_F(EQNs,
                         EqnAddr,
                         VarAddr,
                         Pdict,
                         1) == expected8


expected9 = """def inner_F0(delta, omega):
    return delta*omega
""".strip()
expected10 = """def inner_F1(lam, x, y):
    return lam*y + x
""".strip()


def test_print_sub_inner_F():
    EQNs = dict()
    omega = iVar('omega')
    delta = iVar('delta')
    x = iVar('x')
    y = iVar('y')
    lam = Para('lam')
    EQNs['a'] = Eqn('a', omega * delta)
    EQNs['b'] = Ode('b', x + y * lam, diff_var=x)

    assert print_sub_inner_F(EQNs)[0] == expected9
    assert print_sub_inner_F(EQNs)[1] == expected10
