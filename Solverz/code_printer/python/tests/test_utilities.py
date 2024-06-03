import pytest
import numpy as np

from sympy import symbols, pycode
from sympy.codegen.ast import FunctionCall as SpFuncCall, Assignment

from Solverz.code_printer.python.utilities import FunctionCall, parse_p, parse_trigger_func, \
    print_var, print_param, print_trigger, print_F_J_prototype, print_eqn_assignment, zeros
from Solverz.sym_algebra.symbols import iVar, Para
from Solverz.equation.eqn import Eqn, Ode
from Solverz.equation.param import Param, TimeSeriesParam
from Solverz.utilities.address import Address
from sympy.codegen.ast import FunctionDefinition, Return

expected = """
            The `args` parameter passed to sympy.codegen.ast.FunctionCall should not contain str, which may cause sympy parsing error. 
            For example, the sympy.codegen.ast.FunctionCall parses str E in args to math.e! Please use sympy.Symbol objects instead.
            """


def test_FunctionCall():
    E = symbols('E')
    assert pycode(FunctionCall('a', [E])) == 'a(E)'
    assert pycode(SpFuncCall('a', ['E'])) == 'a(math.e)'

    with pytest.raises(ValueError, match=expected):
        assert pycode(FunctionCall('a', ['E'])) == 'a(E)'


def test_parse_p():
    Pdict = dict()
    Pdict["lam"] = Param('lam', [1, 1, 1])
    Pdict["G6"] = TimeSeriesParam('G6',
                                  [0, 1, 2, 3],
                                  [0, 100, 200, 300])
    p = parse_p(Pdict)
    assert "lam" in p
    np.testing.assert_allclose(p["lam"], np.array([1, 1, 1]))
    assert "G6" in p
    np.testing.assert_allclose(p["G6"].value, 0)
    np.testing.assert_allclose(p["G6"].get_v_t(40), 0.4)


def test_parse_trigger_func():
    ax = Param('ax',
               [0, 1],
               triggerable=True,
               trigger_var=['x'],
               trigger_fun=np.sin)
    lam = Param('lam', [1, 1, 1])
    G6 = TimeSeriesParam('G6',
                         [0, 1, 2, 3],
                         [0, 100, 200, 300])
    func_dict = parse_trigger_func({'ax': ax, 'lam': lam, 'G6': G6})
    assert "ax_trigger_func" in func_dict
    np.testing.assert_allclose(np.sin(1), func_dict["ax_trigger_func"](1))


def test_print_var():
    va = Address()
    va.add('delta', 10)
    va.add('omega', 3)
    va.add('p', 2)
    var_dec, var_list = print_var(va, 0)
    y_ = iVar('y_', internal_use=True)
    delta = iVar('delta')
    omega = iVar('omega')
    p = iVar('p')
    assert var_dec == [Assignment(delta, y_[0:10]),
                       Assignment(omega, y_[10:13]),
                       Assignment(p, y_[13:15])]
    assert var_list == [iVar('delta'),
                        iVar('omega'),
                        iVar('p')]

    var_dec, var_list = print_var(va, 1)
    suffix = '0'
    delta_tag_0 = iVar('delta' + '_tag_' + suffix)
    omega_tag_0 = iVar('omega' + '_tag_' + suffix)
    p_tag_0 = iVar('p' + '_tag_' + suffix)
    y_0 = iVar('y_' + suffix, internal_use=True)
    assert var_dec == [Assignment(delta, y_[0:10]),
                       Assignment(omega, y_[10:13]),
                       Assignment(p, y_[13:15]),
                       Assignment(delta_tag_0, y_0[0:10]),
                       Assignment(omega_tag_0, y_0[10:13]),
                       Assignment(p_tag_0, y_0[13:15])]
    assert var_list == [delta,
                        omega,
                        p,
                        delta_tag_0,
                        omega_tag_0,
                        p_tag_0]


def test_print_param():
    Pdict = dict()
    lam = Param('lam', [1, 1, 1])
    Pdict["lam"] = lam
    G6 = TimeSeriesParam('G6',
                         [0, 1, 2, 3],
                         [0, 100, 200, 300])
    Pdict["G6"] = G6
    Area = Param('Area',
                 [1, 1, 1],
                 is_alias=True)
    Pdict["Area"] = Area
    param_declaration, param_list = print_param(Pdict)
    assert pycode(param_declaration[0]) == 'lam = p_["lam"]'
    assert pycode(param_declaration[1]) == 'G6 = p_["G6"].get_v_t(t)'
    assert param_list[0] == Para('lam')
    assert param_list[1] == Para('G6')


def test_print_trigger():
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
    tri_decla = print_trigger(Pdict)
    assert pycode(tri_decla[0]) == 'ax = ax_trigger_func(x)'


expected1 = """def F_(y_, p_):
    return a
""".strip()
expected2 = """def J_(y_, p_):
    return a
""".strip()
expected3 = """def F_(t, y_, p_):
    return a
""".strip()
expected4 = """def J_(t, y_, p_):
    return a
""".strip()
expected5 = """def J_(t, y_, p_, y_0):
    return a
""".strip()
expected6 = """def F_(t, y_, p_, y_0):
    return a
""".strip()


def test_print_F_J_prototype():
    body = []
    body.extend([Return(iVar('a'))])

    fp = print_F_J_prototype('AE', 'F_', 0)
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    assert pycode(fd) == expected1

    fp = print_F_J_prototype('AE', 'J_', 0)
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    assert pycode(fd) == expected2

    fp = print_F_J_prototype('DAE', 'F_', 0)
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    assert pycode(fd) == expected3

    fp = print_F_J_prototype('DAE', 'J_', 0)
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    assert pycode(fd) == expected4

    with pytest.raises(TypeError, match="Only FDAE supports the case where nstep > 0! Not DAE"):
        fp = print_F_J_prototype('DAE', 'J_', 1)

    with pytest.raises(ValueError, match="Func name A not supported!"):
        fp = print_F_J_prototype('DAE', 'A', 1)

    with pytest.raises(ValueError, match="nstep of FDAE should be greater than 0!"):
        fp = print_F_J_prototype('FDAE', 'J_', 0)

    fp = print_F_J_prototype('FDAE', 'J_', 1)
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    assert pycode(fd) == expected5

    fp = print_F_J_prototype('FDAE', 'F_', 1)
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    assert pycode(fd) == expected6


def test_eqn_assignment():
    _F_ = iVar('_F_', internal_use=True)
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
    eqn_decl = print_eqn_assignment(EQNs, EqnAddr)
    assert eqn_decl[0] == Assignment(_F_, zeros(15))
    assert eqn_decl[1] == Assignment(_F_[0:10], delta*omega)
    assert eqn_decl[2] == Assignment(_F_[10:15], lam*y+x)

    eqn_decl = print_eqn_assignment(EQNs, EqnAddr, True)
    assert eqn_decl[0] == Assignment(_F_[0:10], FunctionCall('inner_F0', [delta, omega]))
    assert eqn_decl[1] == Assignment(_F_[10:15], FunctionCall('inner_F1', [lam, x, y]))
