import inspect
import os
import shutil
import sys

import numpy as np
from sympy import symbols, pycode, Integer

from Solverz import as_Vars, Eqn, AE, sin, made_numerical
from Solverz.code_printer.make_module import module_printer
from Solverz.code_printer.python.inline.inline_printer import print_J_block
from Solverz.code_printer.python.utilities import _print_var_parser
from Solverz.sym_algebra.symbols import idx, iVar, Para


def test_module_printer():
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


