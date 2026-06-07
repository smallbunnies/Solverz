# tests/test_provenance.py
import numpy as np
from Solverz import Eqn, Ode, Var, Model, Idx, LoopEqn, Sum, Param


def test_equations_have_source_default_none():
    e = Eqn('e', Var('x', 1) - 1)
    o = Ode('o', Var('y', 1), Var('y', 1))
    assert e.source is None
    assert o.source is None  # Ode inherits Eqn.__init__


def test_loopeqn_has_source_default_none():
    m = Model()
    m.x = Var('x', np.ones(3))
    m.b = Param('b', np.zeros(3))
    i = Idx('i', 3)
    m.le = LoopEqn('le', outer_index=i, body=m.x[i] - m.b[i], model=m)
    assert m.le.source is None


from Solverz import stamp_source


def test_stamp_source_tags_all_equations():
    m = Model()
    m.x = Var('x', np.ones(2))
    m.b = Param('b', np.zeros(2))
    m.e0 = Eqn('e0', m.x[0] - m.b[0])
    m.e1 = Eqn('e1', m.x[1] - m.b[1])
    stamp_source(m, component='demo', package='Pkg', version='1.2.3')
    assert m.e0.source == {'component': 'demo', 'package': 'Pkg', 'version': '1.2.3'}
    assert m.e1.source == {'component': 'demo', 'package': 'Pkg', 'version': '1.2.3'}


def test_stamp_source_overwrite_false_preserves_existing():
    m = Model()
    m.x = Var('x', np.ones(1))
    m.e = Eqn('e', m.x[0])
    m.e.source = {'component': 'inner', 'package': 'Sub', 'version': '0.1'}
    stamp_source(m, component='outer', package='Pkg', version='9.9')
    assert m.e.source['component'] == 'inner'  # not clobbered
    stamp_source(m, component='outer', package='Pkg', version='9.9', overwrite=True)
    assert m.e.source['component'] == 'outer'


def test_stamp_source_ignores_non_equations():
    m = Model()
    m.x = Var('x', np.ones(1))     # a Var, not an equation
    m.b = Param('b', np.zeros(1))  # a Param
    m.e = Eqn('e', m.x[0])
    stamp_source(m, component='c', package='P', version='1')
    # Vars/Params untouched; only the Eqn is stamped.
    assert m.e.source is not None
    assert not hasattr(m.x, 'source') or m.x.source is None


from Solverz.code_printer.python.module.module_printer import _with_docstring


def test_with_docstring_inserts_after_def_line():
    src = "def inner_F0(x, y):\n    return x + y\n"
    out = _with_docstring(src, 'P_eqn — Pkg 1.0 / comp')
    lines = out.split('\n')
    assert lines[0] == 'def inner_F0(x, y):'
    assert lines[1].strip().startswith('"""')
    assert 'P_eqn' in lines[1]
    assert lines[2].strip() == 'return x + y'


def test_with_docstring_sanitizes_to_single_line():
    src = "def f():\n    return 1\n"
    out = _with_docstring(src, 'line1\nline2  with   spaces')
    doc_line = out.split('\n')[1]
    assert doc_line.count('"""') == 2          # opening + closing on one line
    assert 'line1 line2 with spaces' in doc_line


def test_with_docstring_noop_when_empty_doc():
    src = "def f():\n    return 1\n"
    assert _with_docstring(src, '') == src
