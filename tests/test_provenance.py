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


import os, sys, tempfile, importlib, uuid


def _render(model, jit=False):
    """create_instance + module_printer render; return the module dir."""
    from Solverz import module_printer
    spf, y0 = model.create_instance()
    d = tempfile.mkdtemp(prefix='sz_prov_')
    name = f'prov_{uuid.uuid4().hex[:8]}'
    module_printer(spf, y0, name, directory=d, jit=jit).render()
    return os.path.join(d, name)


def _read(path, fname):
    with open(os.path.join(path, fname)) as f:
        return f.read()


def _demo_model_with_source():
    m = Model()
    m.x = Var('x', np.ones(2))
    m.b = Param('b', np.array([1.0, 2.0]))
    m.e0 = Eqn('e0', m.x[0] - m.b[0])
    m.e1 = Eqn('e1', m.x[1] - m.b[1])
    stamp_source(m, component='demo', package='Pkg', version='1.2.3')
    return m


def test_f_functions_have_source_docstrings():
    mod = _render(_demo_model_with_source(), jit=False)
    num_func = _read(mod, 'num_func.py')
    # Each inner_F has a docstring naming its equation + source.
    assert '"""e0' in num_func or '"""e1' in num_func
    assert 'Pkg 1.2.3 / demo' in num_func


def test_f_functions_name_only_when_unsourced():
    m = Model()
    m.x = Var('x', np.ones(1))
    m.b = Param('b', np.array([1.0]))
    m.e = Eqn('e', m.x[0] - m.b[0])
    mod = _render(m, jit=False)
    num_func = _read(mod, 'num_func.py')
    assert '"""e"""' in num_func          # name-only docstring
    assert ' / ' not in num_func.split('"""e"""')[0][-200:]  # no source suffix nearby


def _demo_model_nonconstant_jac():
    """A scalar model whose Jacobian is *non-constant*, so the printer
    emits per-derivative ``inner_J{k}`` kernels (a linear model folds
    its constant derivatives into the static ``_data_`` and emits no
    scalar kernel at all)."""
    m = Model()
    m.x = Var('x', np.ones(2))
    m.b = Param('b', np.array([1.0, 2.0]))
    m.e0 = Eqn('e0', m.x[0] ** 2 - m.b[0])   # d/dx0 = 2*x0 -> non-constant
    m.e1 = Eqn('e1', m.x[1] ** 2 - m.b[1])
    stamp_source(m, component='demo', package='Pkg', version='1.2.3')
    return m


def test_j_kernels_have_derivative_docstrings():
    mod = _render(_demo_model_nonconstant_jac(), jit=False)
    num_func = _read(mod, 'num_func.py')
    # Scalar J kernels annotated as d(eqn)/d(var) with source. The diff
    # var of a scalar block is the indexed element, so var.name is the
    # ``x[k]`` form, e.g. ``d(e0)/d(x[0])``.
    assert 'd(e0)/d(x[' in num_func or 'd(e1)/d(x[' in num_func
    assert 'Pkg 1.2.3 / demo' in num_func.split('def inner_J')[-1] \
        or 'Pkg 1.2.3 / demo' in num_func


def test_loop_jac_kernel_has_derivative_docstring():
    """A LoopEqn whose Jacobian flows through the dense ``LoopEqnDiff``
    kernel path (``_sz_loop_jac_kernel_N``) must also carry the
    ``d(eqn)/d(var)`` + source docstring. A bilinear body with no
    matrix Param carrier forces that path."""
    m = Model()
    m.x = Var('x', np.array([1.5, 2.0, 2.5]))
    m.b = Param('b', np.zeros(3))
    i = Idx('i', 3)
    j = Idx('j', 3)
    m.le = LoopEqn('le', outer_index=i, body=Sum(m.x[i] * m.x[j], j) - m.b[i], model=m)
    stamp_source(m, component='demo', package='Pkg', version='1.0')
    num_func = _read(_render(m, jit=False), 'num_func.py')
    assert '_sz_loop_jac_kernel_' in num_func   # the kernel path is exercised
    kernel = num_func.split('def _sz_loop_jac_kernel_')[-1]
    assert 'd(le)/d(x)' in kernel
    assert 'Pkg 1.0 / demo' in kernel
