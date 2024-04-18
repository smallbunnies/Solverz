import inspect
import os
import shutil
import sys

import numpy as np
from sympy import symbols, pycode, Integer

from Solverz import as_Vars, Eqn, AE, sin, made_numerical
from Solverz.code_printer.make_module import module_printer
from Solverz.code_printer.py_printer import _parse_jac_eqn_address, _parse_jac_var_address, _parse_jac_data, \
    print_J_block, _print_var_parser
from Solverz.sym_algebra.symbols import idx, Var, Para


def test_address_parser():
    eqn_address = slice(0, 1)
    assert pycode(_parse_jac_eqn_address(eqn_address, 1, False)) == '0'
    assert _parse_jac_eqn_address(eqn_address, 2, False).__repr__() == 'slice(0, 1, None)'
    assert pycode(_parse_jac_eqn_address(eqn_address, 1, True)) == '[0]'
    assert pycode(_parse_jac_eqn_address(eqn_address, 2, True)) == 'arange(0, 1)[value_coo.row]'

    eqn_address = slice(0, 5)
    assert pycode(_parse_jac_eqn_address(eqn_address, 1, False)) == 'arange(0, 5)'
    assert pycode(_parse_jac_eqn_address(eqn_address, 1, True)) == 'arange(0, 5)'
    assert _parse_jac_eqn_address(eqn_address, 2, False).__repr__() == 'slice(0, 5, None)'
    assert _parse_jac_eqn_address(eqn_address, 2, True).__repr__() == 'arange(0, 5)[value_coo.row]'

    v_address = slice(0, 1)
    assert pycode(_parse_jac_var_address(v_address, 1, None, False)) == 'arange(0, 1)'
    assert pycode(_parse_jac_var_address(v_address, 1, None, True)) == 'arange(0, 1)'
    assert _parse_jac_var_address(v_address, 2, None, False).__repr__() == 'slice(0, 1, None)'
    assert pycode(_parse_jac_var_address(v_address, 2, None, True)) == 'arange(0, 1)[value_coo.col]'

    v_address = slice(0, 3)
    # _parse_jac_var_address(v_address, 2, v_idx, True)
    assert _parse_jac_var_address(v_address, 2, None, False).__repr__() == 'slice(0, 3, None)'
    assert _parse_jac_var_address(v_address, 2, None, True).__repr__() == 'arange(0, 3)[value_coo.col]'
    assert pycode(_parse_jac_var_address(v_address, 1, None, True)) == 'arange(0, 3)'
    assert pycode(_parse_jac_var_address(v_address, 1, None, False)) == 'arange(0, 3)'
    assert pycode(_parse_jac_var_address(v_address, 0, 2, True, 3)) == '3*[2]'
    assert pycode(_parse_jac_var_address(v_address, 1, 2, True, 3)) == '3*[2]'

    M = idx('M')
    assert _parse_jac_var_address(v_address, 2, M, True).__repr__() == 'arange(0, 3)[M[value_coo.col]]'
    assert pycode(_parse_jac_var_address(v_address, 2, M, False)) == 'arange(0, 3)[M]'
    assert _parse_jac_var_address(v_address, 1, M, True).__repr__() == 'arange(0, 3)[M]'
    assert pycode(_parse_jac_var_address(v_address, 1, M, False)) == 'arange(0, 3)[M]'

    v_idx = [0, 1, 2]
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, True)) == 'arange(0, 3)'
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, False)) == 'arange(0, 3)'
    assert pycode(_parse_jac_var_address(v_address, 2, v_idx, True)) == 'arange(0, 3)[value_coo.col]'
    assert _parse_jac_var_address(v_address, 2, v_idx, False).__repr__() == 'slice(0, 3, None)'

    v_address = slice(0, 10)
    v_idx = [0, 2, 4, 6]
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, True)) == '[0, 2, 4, 6]'
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, False)) == '[0, 2, 4, 6]'
    assert pycode(_parse_jac_var_address(v_address, 2, v_idx, True)) == 'array([0,2,4,6])[value_coo.col]'
    assert pycode(_parse_jac_var_address(v_address, 2, v_idx, False)) == '[0, 2, 4, 6]'

    v_idx = 2
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, True)) == '[2]'
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, False)) == '2'
    assert _parse_jac_var_address(v_address, 2, v_idx, True).__repr__() == 'array([2])[value_coo.col]'
    assert pycode(_parse_jac_var_address(v_address, 2, v_idx, False)) == '2'

    x, y = symbols('x, y')
    assert pycode(_parse_jac_data(5, 1, x * y)) == 'x*y'
    assert pycode(_parse_jac_data(5, 0, x * y)) == '5*((x*y).tolist())'
    assert pycode(_parse_jac_data(5, 0, 3, rhs_v_type='Number')) == '5*[3]'
    assert pycode(_parse_jac_data(5, 2, 3)) == 'value_coo.data'


def test_var_parser():
    assert pycode(_print_var_parser('x', '0', slice(0, 1))) == 'x_tag_0 = y_0[0:1]'
    assert pycode(_print_var_parser('x', '0', slice(3, 6))) == 'x_tag_0 = y_0[3:6]'


# def test_F_printer():
#     F = _print_F_assignment(slice(20, 21), Var('x'))
#     assert pycode(F) == '_F_[20:21] = x'
#     F = _print_F_assignment(slice(20, 23), Var('x'))
#     assert pycode(F) == '_F_[20:23] = x'


def test_J_block_printer():
    # dense
    Jb = print_J_block(slice(0, 3),
                       slice(3, 6),
                       0,
                       None,
                       Var('omega_b'),
                       False)
    assert pycode(Jb) == '[J_[arange(0, 3),arange(3, 6)] += omega_b]'
    Jb = print_J_block(slice(3, 6),
                       slice(12, 21),
                       1,
                       idx('g'),
                       -Var('Ixg') / Var('Tj'),
                       False)
    assert pycode(Jb) == '[J_[arange(3, 6),arange(12, 21)[g]] += -Ixg/Tj]'
    Jb = print_J_block(slice(12, 15),
                       slice(6, 9),
                       0,
                       None,
                       1,
                       False)
    assert pycode(Jb) == '[J_[arange(12, 15),arange(6, 9)] += 1]'
    Jb = print_J_block(slice(18, 24),
                       slice(12, 21),
                       2,
                       None,
                       Var('G')[idx('ng'), :],
                       False)
    assert pycode(Jb) == '[J_[18:24,12:21] += G[ng,:]]'

    Jb = print_J_block(slice(30, 31),
                       slice(0, 10),
                       2,
                       None,
                       Para('G', dim=2),
                       False)
    assert pycode(Jb) == '[J_[30:31,0:10] += G]'

    # sparse
    Jb = print_J_block(slice(39, 47),
                       slice(15, 26),
                       0,
                       [0, 1, 2, 3, 4, 5, 9, 10],
                       Integer(1),
                       True,
                       rhs_v_dtpe='Number')
    assert pycode(
        Jb) == '[row.extend(arange(39, 47)), col.extend([15, 16, 17, 18, 19, 20, 24, 25]), data.extend(8*[1])]'

    Jb = print_J_block(slice(248, 259),
                       slice(269, 280),
                       0,
                       None,
                       Para('dt') * Para('va') ** 2 / Para('S'),
                       True)
    assert pycode(
        Jb) == '[row.extend(arange(248, 259)), col.extend(arange(269, 280)), data.extend(11*((dt*va**2/S).tolist()))]'

    Jb = print_J_block(slice(281, 292),
                       slice(281, 292),
                       1,
                       None,
                       Para('dx') + Var('p'),
                       True)
    assert pycode(Jb) == '[row.extend(arange(281, 292)), col.extend(arange(281, 292)), data.extend(dx + p)]'

    Jb = print_J_block(slice(0, 3),
                       slice(3, 6),
                       2,
                       None,
                       Var('omega_b'),
                       True)
    assert pycode(
        Jb) == '[value_coo = coo_array(omega_b), row.extend(arange(0, 3)[value_coo.row]), col.extend(arange(3, 6)[value_coo.col]), data.extend(value_coo.data)]'

    Jb = print_J_block(slice(3, 6),
                       slice(12, 21),
                       2,
                       idx('g'),
                       Para('G'),
                       True)
    assert pycode(
        Jb) == '[value_coo = coo_array(G), row.extend(arange(3, 6)[value_coo.row]), col.extend(arange(12, 21)[g[value_coo.col]]), data.extend(value_coo.data)]'

    Jb = print_J_block(slice(39, 47),
                       slice(15, 26),
                       2,
                       [0, 1, 2, 3, 4, 5, 9, 10],
                       Para('G'),
                       True)
    assert pycode(
        Jb) == '[value_coo = coo_array(G), row.extend(arange(39, 47)[value_coo.row]), col.extend(array([15,16,17,18,19,20,24,25])[value_coo.col]), data.extend(value_coo.data)]'

    Jb = print_J_block(slice(39, 47),
                       slice(15, 26),
                       2,
                       slice(0, 5),
                       Para('G'),
                       True)
    assert pycode(
        Jb) == '[value_coo = coo_array(G), row.extend(arange(39, 47)[value_coo.row]), col.extend(arange(15, 20)[value_coo.col]), data.extend(value_coo.data)]'


def test_py_printer():
    x = Var('x', [1, 1])
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


def test_made_numerical():
    x = Var('x', [1, 1])
    f1 = Eqn('f1', 2 * x[0] + x[1])
    f2 = Eqn('f2', x[0] ** 2 + sin(x[1]))

    F = AE([f1, f2])
    y = as_Vars([x])
    nF, code = made_numerical(F, y, sparse=True, output_code=True)
    F0 = nF.F(y, nF.p)
    J0 = nF.J(y, nF.p)
    np.testing.assert_allclose(F0, np.array([2 * 1 + 1, 1 + np.sin(1)]), rtol=1e-8)
    np.testing.assert_allclose(J0.toarray(), np.array([[2, 1], [2, 0.54030231]]), rtol=1e-8)
