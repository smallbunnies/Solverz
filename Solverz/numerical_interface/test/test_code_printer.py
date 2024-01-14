from sympy import symbols, pycode, Integer

from Solverz.numerical_interface.code_printer import _parse_jac_eqn_address, _parse_jac_var_address, _parse_jac_data, \
    print_J_block, _print_F_assignment
from Solverz.symboli_algebra.symbols import idx, Var, Para


def test_address_parser():
    eqn_address = slice(0, 1)
    assert pycode(_parse_jac_eqn_address(eqn_address, 1, False)) == '0'
    assert pycode(_parse_jac_eqn_address(eqn_address, 1, True)) == '[0]'

    eqn_address = slice(0, 5)
    assert pycode(_parse_jac_eqn_address(eqn_address, 1, False)) == 'arange(0, 5)'
    assert pycode(_parse_jac_eqn_address(eqn_address, 1, True)) == 'arange(0, 5)'
    assert _parse_jac_eqn_address(eqn_address, 2, False).__repr__() == 'slice(0, 5, None)'
    # assert _parse_jac_eqn_address(eqn_address, 2, True).__repr__() == 'Arange(0, 5)'

    v_address = slice(0, 1)
    assert _parse_jac_var_address(v_address, 1, None, False).__repr__() == '0'
    assert pycode(_parse_jac_var_address(v_address, 1, None, True)) == '[0]'

    v_address = slice(0, 3)
    # _parse_jac_var_address(v_address, 2, v_idx, True)
    assert _parse_jac_var_address(v_address, 2, None, False).__repr__() == 'slice(0, 3, None)'
    assert pycode(_parse_jac_var_address(v_address, 1, None, True)) == 'arange(0, 3)'
    assert pycode(_parse_jac_var_address(v_address, 1, None, False)) == 'arange(0, 3)'

    M = idx('M')
    # print(_parse_jac_var_address(v_address, 2, v_idx, True))
    assert pycode(_parse_jac_var_address(v_address, 2, M, False)) == 'arange(0, 3)[M]'

    v_idx = [0, 1, 2]
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, True)) == 'arange(0, 3)'
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, False)) == 'arange(0, 3)'

    v_idx = [0, 2]
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, True)) == '[0, 2]'
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, False)) == '[0, 2]'

    v_idx = 2
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, True)) == '[2]'
    assert pycode(_parse_jac_var_address(v_address, 1, v_idx, False)) == '2'

    x, y = symbols('x, y')
    assert pycode(_parse_jac_data(5, 1, x * y)) == 'x*y'
    assert pycode(_parse_jac_data(5, 0, x * y)) == '5*((x*y).tolist())'
    assert pycode(_parse_jac_data(5, 0, 3)) == '5*[3]'


def test_F_printer():
    F = _print_F_assignment(slice(20, 21), Var('x'))
    assert pycode(F) == 'F_[20:21] = x'
    F = _print_F_assignment(slice(20, 23), Var('x'))
    assert pycode(F) == 'F_[20:23] = x'


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
                       True)
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
