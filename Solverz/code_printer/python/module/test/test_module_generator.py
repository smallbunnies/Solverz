import inspect
import os
import shutil
import sys

import numpy as np
from sympy import symbols, pycode, Integer

from Solverz import as_Vars, Eqn, Ode, AE, sin, made_numerical, Model, Var, Param, TimeSeriesParam, AliasVar, Abs
from Solverz.code_printer.make_module import module_printer
from Solverz.code_printer.python.inline.inline_printer import print_J_block
from Solverz.code_printer.python.utilities import _print_var_parser
from Solverz.sym_algebra.symbols import idx, iVar, Para

expected_dependency = r"""import os
current_module_dir = os.path.dirname(os.path.abspath(__file__))
from Solverz import load
auxiliary = load(f"{current_module_dir}\\param_and_setting.pkl")
from numpy import *
from numpy import abs
from Solverz.num_api.custom_function import *
from scipy.sparse import *
from numba import njit
setting = auxiliary["eqn_param"]
row = setting["row"]
col = setting["col"]
p_ = auxiliary["p"]
y = auxiliary["vars"]
"""

expected_init = r"""from .num_func import F_, J_
from .dependency import setting, p_, y
import time
from Solverz.num_api.num_eqn import nAE
mdl = nAE(F_, J_, p_)
print("Compiling model test_eqn1...")
start = time.perf_counter()
mdl.F(y, p_)
mdl.J(y, p_)
end = time.perf_counter()
print(f'Compiling time elapsed: {end-start}s')
"""


def test_AE_module_printer():
    x = iVar('x', [1, 1])
    f1 = Eqn('f1', 2 * x[0] + x[1])
    f2 = Eqn('f2', x[0] ** 2 + sin(x[1]))

    F = AE([f1, f2])
    y = as_Vars([x])

    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)

    test_folder_path = current_folder + '\\Solverz_testaabbccddeeffgghh'

    pyprinter = module_printer(F,
                               y,
                               'test_eqn1',
                               directory=test_folder_path + '\\a_test_direc',
                               jit=True)
    pyprinter.render()

    pyprinter1 = module_printer(F,
                                y,
                                'test_eqn2',
                                directory=test_folder_path + '\\a_test_direc',
                                jit=False)
    pyprinter1.render()

    sys.path.extend([test_folder_path + '\\a_test_direc'])

    from test_eqn1 import mdl, y
    from test_eqn1.num_func import inner_F, inner_F0, inner_F1, inner_J

    F0 = mdl.F(y, mdl.p)
    J0 = mdl.J(y, mdl.p)
    np.testing.assert_allclose(F0, np.array([2 * 1 + 1, 1 + np.sin(1)]), rtol=1e-8)
    np.testing.assert_allclose(J0.toarray(), np.array([[2, 1], [2, 0.54030231]]), rtol=1e-8)

    assert inspect.getsource(
        inner_F) == '@njit(cache=True)\ndef inner_F(_F_, x):\n    _F_[0:1] = inner_F0(x)\n    _F_[1:2] = inner_F1(x)\n    return _F_\n'
    assert inspect.getsource(inner_F0.func_code) == '@njit(cache=True)\ndef inner_F0(x):\n    return 2*x[0] + x[1]\n'
    assert inspect.getsource(
        inner_F1.func_code) == '@njit(cache=True)\ndef inner_F1(x):\n    return x[0]**2 + sin(x[1])\n'
    assert inspect.getsource(
        inner_J) == '@njit(cache=True)\ndef inner_J(_data_, x):\n    _data_[2:3] = inner_J0(x)\n    _data_[3:4] = inner_J1(x)\n    return _data_\n'

    # read dependency.py
    with open(test_folder_path + '\\a_test_direc\\test_eqn1\\dependency.py', 'r', encoding='utf-8') as file:
        file_content = file.read()

    assert file_content == expected_dependency

    with open(test_folder_path + '\\a_test_direc\\test_eqn1\\__init__.py', 'r', encoding='utf-8') as file:
        file_content = file.read()

    assert file_content == expected_init

    from test_eqn2 import mdl, y
    from test_eqn2.num_func import inner_F, inner_F0, inner_F1, inner_J

    F0 = mdl.F(y, mdl.p)
    J0 = mdl.J(y, mdl.p)
    np.testing.assert_allclose(F0, np.array([2 * 1 + 1, 1 + np.sin(1)]), rtol=1e-8)
    np.testing.assert_allclose(J0.toarray(), np.array([[2, 1], [2, 0.54030231]]), rtol=1e-8)

    assert inspect.getsource(
        inner_F) == 'def inner_F(_F_, x):\n    _F_[0:1] = inner_F0(x)\n    _F_[1:2] = inner_F1(x)\n    return _F_\n'
    assert inspect.getsource(inner_F0) == 'def inner_F0(x):\n    return 2*x[0] + x[1]\n'
    assert inspect.getsource(inner_F1) == 'def inner_F1(x):\n    return x[0]**2 + sin(x[1])\n'
    assert inspect.getsource(
        inner_J) == 'def inner_J(_data_, x):\n    _data_[2:3] = inner_J0(x)\n    _data_[3:4] = inner_J1(x)\n    return _data_\n'

    shutil.rmtree(test_folder_path)


expected_F = """def F_(t, y_, p_, y_0):
    p = y_[0:82]
    q = y_[82:164]
    p_tag_0 = y_0[0:82]
    q_tag_0 = y_0[82:164]
    pb = p_["pb"].get_v_t(t)
    qb = p_["qb"]
    return inner_F(_F_, p, q, p_tag_0, q_tag_0, pb, qb)
"""

expected_inner_F = """@njit(cache=True)
def inner_F(_F_, p, q, p_tag_0, q_tag_0, pb, qb):
    _F_[0:81] = inner_F0(p, p_tag_0, q, q_tag_0)
    _F_[81:162] = inner_F1(p, p_tag_0, q, q_tag_0)
    _F_[162:163] = inner_F2(p, pb)
    _F_[163:164] = inner_F3(q, qb)
    return _F_
"""

expected_J = """def J_(t, y_, p_, y_0):
    p = y_[0:82]
    q = y_[82:164]
    p_tag_0 = y_0[0:82]
    q_tag_0 = y_0[82:164]
    pb = p_["pb"].get_v_t(t)
    qb = p_["qb"]
    data = inner_J(_data_, p, q, p_tag_0, q_tag_0, pb, qb)
    return coo_array((data, (row, col)), (164, 164)).tocsc()
"""

expected_inner_J = """@njit(cache=True)
def inner_J(_data_, p, q, p_tag_0, q_tag_0, pb, qb):
    _data_[0:81] = inner_J0(p, p_tag_0, q, q_tag_0)
    _data_[81:162] = inner_J1(p, p_tag_0, q, q_tag_0)
    _data_[162:243] = inner_J2(p, p_tag_0, q, q_tag_0)
    _data_[243:324] = inner_J3(p, p_tag_0, q, q_tag_0)
    return _data_
"""


def test_FDAE_module_generator():
    L = 51000 * 0.8
    p0 = 6621246.69079594
    q0 = 14

    va = Integer(340)
    D = 0.5901
    S = np.pi * (D / 2) ** 2
    lam = 0.03

    dx = 500
    dt = 1.4706
    M = int(L / dx)
    m1 = Model()
    m1.p = Var('p', value=p0 * np.ones((M + 1,)))
    m1.q = Var('q', value=q0 * np.ones((M + 1,)))
    m1.p0 = AliasVar('p', init=m1.p)
    m1.q0 = AliasVar('q', init=m1.q)

    m1.ae1 = Eqn('cha1',
                 m1.p[1:M + 1] - m1.p0[0:M] + va / S * (m1.q[1:M + 1] - m1.q0[0:M]) +
                 lam * va ** 2 * dx / (4 * D * S ** 2) * (m1.q[1:M + 1] + m1.q0[0:M]) * Abs(
                     m1.q[1:M + 1] + m1.q0[0:M]) / (
                         m1.p[1:M + 1] + m1.p0[0:M]))

    m1.ae2 = Eqn('cha2',
                 m1.p0[1:M + 1] - m1.p[0:M] + va / S * (m1.q[0:M] - m1.q0[1:M + 1]) +
                 lam * va ** 2 * dx / (4 * D * S ** 2) * (m1.q[0:M] + m1.q0[1:M + 1]) * Abs(
                     m1.q[0:M] + m1.q0[1:M + 1]) / (
                         m1.p[0:M] + m1.p0[1:M + 1]))
    T = 5 * 3600
    pb1 = 1e6
    pb0 = 6621246.69079594
    pb_t = [pb0, pb0, pb1, pb1]
    tseries = [0, 1000, 1000 + 10 * dt, T]
    m1.pb = TimeSeriesParam('pb',
                            v_series=pb_t,
                            time_series=tseries)
    m1.qb = Param('qb', q0)
    m1.bd1 = Eqn('bd1', m1.p[0] - m1.pb)
    m1.bd2 = Eqn('bd2', m1.q[M] - m1.qb)
    fdae, y0 = m1.create_instance()

    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)

    test_folder_path = current_folder + '\\Solverz_testaabbccddeeffgghh'

    pyprinter = module_printer(fdae,
                               y0,
                               'test_fdae',
                               directory=test_folder_path,
                               jit=True)
    pyprinter.render()

    sys.path.extend([test_folder_path])

    from test_fdae.num_func import F_, J_, inner_F, inner_J

    assert inspect.getsource(F_) == expected_F
    assert inspect.getsource(J_) == expected_J
    assert inspect.getsource(inner_F) == expected_inner_F
    assert inspect.getsource(inner_J) == expected_inner_J

    shutil.rmtree(test_folder_path)


expected_F1 = """def F_(t, y_, p_):
    h = y_[0:1]
    v = y_[1:2]
    return inner_F(_F_, h, v)
"""

expected_inner_F1 = """@njit(cache=True)
def inner_F(_F_, h, v):
    _F_[0:1] = inner_F0(v)
    _F_[1:2] = inner_F1()
    return _F_
"""

expected_J1 = """def J_(t, y_, p_):
    h = y_[0:1]
    v = y_[1:2]
    data = inner_J(_data_, h, v)
    return coo_array((data, (row, col)), (2, 2)).tocsc()
"""

expected_inner_J1 = """@njit(cache=True)
def inner_J(_data_, h, v):
    return _data_
"""


def test_DAE_module_generator():
    m = Model()
    m.h = Var('h', 0)
    m.v = Var('v', 20)
    m.f1 = Ode('f1', f=m.v, diff_var=m.h)
    m.f2 = Ode('f2', f=-9.8, diff_var=m.v)
    bball, y0 = m.create_instance()

    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)

    test_folder_path = current_folder + '\\Solverz_testaabbccddeeffgghh'

    pyprinter = module_printer(bball,
                               y0,
                               'test_dae',
                               directory=test_folder_path,
                               jit=True)
    pyprinter.render()

    sys.path.extend([test_folder_path])

    from test_dae.num_func import F_, J_, inner_F, inner_J

    assert inspect.getsource(F_) == expected_F1
    assert inspect.getsource(J_) == expected_J1
    assert inspect.getsource(inner_F) == expected_inner_F1
    assert inspect.getsource(inner_J) == expected_inner_J1

    shutil.rmtree(test_folder_path)
