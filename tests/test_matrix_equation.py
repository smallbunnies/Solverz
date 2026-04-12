import copy
import os
import shutil
import sys
import warnings

import numpy as np
import scipy.sparse.linalg
from Solverz import (MatVecMul, Mat_Mul, Var, Param, made_numerical, Model, Eqn,
                     nr_method, module_printer)


# %%
def test_matrix_equation1():
    """A@x-b=0 using legacy MatVecMul"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m.eqnf = Eqn('eqnf', m.b - MatVecMul(m.A, m.x))

    # %%
    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    # %%
    sol = nr_method(mdl, y0)

    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))


def test_matrix_equation2():
    """-A@x+b=0 using legacy MatVecMul"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m.eqnf = Eqn('eqnf', - m.b + MatVecMul(m.A, m.x))

    # %%
    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    # %%
    sol = nr_method(mdl, y0)

    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))


# --- Mat_Mul tests (new unified interface) ---

def test_mat_mul_inline():
    """A@x-b=0 using Mat_Mul in inline mode"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    m.eqnf = Eqn('eqnf', m.b - Mat_Mul(m.A, m.x))

    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))


def test_mat_mul_negative():
    """-A@x+b=0 using Mat_Mul"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    m.eqnf = Eqn('eqnf', -m.b + Mat_Mul(m.A, m.x))

    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))


def test_mat_mul_nonlinear():
    """A@x + x^2 - b = 0: mutable matrix Jacobian (A + diag(2x))"""
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [4.0, 5.0])
    m.A = Param('A', [[2, 1], [1, 3]], dim=2, sparse=True)
    m.eqn = Eqn('f', Mat_Mul(m.A, m.x) + m.x ** 2 - m.b)

    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y.array, np.array([1.0, 1.0]), atol=1e-5)


def _build_mutable_matrix_model():
    """x * (A@x) - b = 0: Jacobian = diag(A@x) + diag(x)@A (mutable matrix)."""
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.A = Param('A', [[2, 1], [1, 3]], dim=2, sparse=True)
    m.b = Param('b', [4.0, 5.0])
    m.eqn = Eqn('f', m.x * Mat_Mul(m.A, m.x) - m.b)
    return m


def test_mat_mul_mutable_jac(tmp_path):
    """Mutable matrix Jacobian: inline, module (jit=False), module (jit=True)
    must produce identical Jacobians at every Newton step.
    """
    m = _build_mutable_matrix_model()
    spf, y0 = m.create_instance()

    # --- 1. Inline ---
    mdl_inline = made_numerical(spf, y0, sparse=True)

    # --- 2. Module (jit=False) ---
    dir_nojit = str(tmp_path / 'nojit')
    printer = module_printer(spf, y0, 'mut_nojit', directory=dir_nojit, jit=False)
    printer.render()
    sys.path.insert(0, dir_nojit)
    from mut_nojit import mdl as mdl_mod, y as y_mod

    # --- 3. Module (jit=True) ---
    dir_jit = str(tmp_path / 'jit')
    printer_jit = module_printer(spf, y0, 'mut_jit', directory=dir_jit, jit=True)
    printer_jit.render()
    sys.path.insert(0, dir_jit)
    from mut_jit import mdl as mdl_jit, y as y_jit

    # --- Compare Jacobian at each Newton step (drive iteration via inline) ---
    y_test = copy.deepcopy(y0)
    for step in range(6):
        J_i = mdl_inline.J(y_test, mdl_inline.p)
        J_m = mdl_mod.J(y_test, mdl_mod.p)
        J_j = mdl_jit.J(y_test, mdl_jit.p)
        np.testing.assert_allclose(
            J_i.toarray(), J_m.toarray(), atol=1e-12,
            err_msg=f"module(jit=False) J mismatch at step {step}")
        np.testing.assert_allclose(
            J_i.toarray(), J_j.toarray(), atol=1e-12,
            err_msg=f"module(jit=True) J mismatch at step {step}")
        F_val = mdl_inline.F(y_test, mdl_inline.p)
        dy = scipy.sparse.linalg.spsolve(J_i, -F_val)
        y_test.array[:] = y_test.array + dy

    # --- Three modes must converge to the same solution ---
    sol_inline = nr_method(mdl_inline, y0)
    sol_mod = nr_method(mdl_mod, y_mod)
    sol_jit = nr_method(mdl_jit, y_jit)
    np.testing.assert_allclose(sol_inline.y.array, sol_mod.y.array, atol=1e-10)
    np.testing.assert_allclose(sol_inline.y.array, sol_jit.y.array, atol=1e-10)


# ---- Regression tests for review findings ----

def test_multi_diag_accumulation(tmp_path):
    """Finding 4: multiple independent ``Diag(...)`` terms must
    accumulate on the diagonal, not overwrite. The derivative of
    ``x*(A@x) + x*(B@x) - b`` w.r.t. ``x`` is
    ``diag(A@x) + diag(x)@A + diag(B@x) + diag(x)@B`` — two diag
    terms both land on ``(i,i)`` positions and their values must sum.
    """
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.A = Param('A', [[2, 1], [1, 3]], dim=2, sparse=True)
    m.b = Param('b', [4.0, 5.0])
    m.B = Param('B', [[1.5, 0.5], [0.5, 2.0]], dim=2, sparse=True)
    m.eqn = Eqn('f', m.x * Mat_Mul(m.A, m.x) + m.x * Mat_Mul(m.B, m.x) - m.b)

    smdl, y0 = m.create_instance()
    mdl_inline = made_numerical(smdl, y0, sparse=True)

    dir_mod = str(tmp_path / 'multi_diag_mod')
    printer = module_printer(smdl, y0, 'multi_diag_mod',
                             directory=dir_mod, jit=True)
    printer.render()
    sys.path.insert(0, dir_mod)
    import importlib
    if 'multi_diag_mod' in sys.modules:
        del sys.modules['multi_diag_mod']
    mod = importlib.import_module('multi_diag_mod')
    mdl_mod = mod.mdl
    y_mod = mod.y

    # Drive by a non-trivial iterate so every diag term has a distinct,
    # non-zero value at (i,i). The bug manifests as module J missing
    # one of the diagonal contributions.
    rng = np.random.default_rng(20260413)
    y_seed = rng.uniform(0.5, 1.5, size=y0.array.shape[0])
    y_test = copy.deepcopy(y0)
    y_test.array[:] = y_seed
    y_test_mod = copy.deepcopy(y_mod)
    y_test_mod.array[:] = y_seed
    J_i = mdl_inline.J(y_test, mdl_inline.p)
    J_m = mdl_mod.J(y_test_mod, mdl_mod.p)
    np.testing.assert_allclose(
        J_i.toarray(), J_m.toarray(), rtol=1e-10, atol=1e-12,
        err_msg='module J misses accumulation across multiple Diag terms')


def test_reserved_prefix_rejected():
    """Findings 5,6: user symbols starting with reserved internal
    prefixes must be rejected at construction time so the code
    generator can freely emit helper names of the form
    ``_sz_mm_<int>`` and ``_sz_mb_<int>_...`` without shadowing."""
    import pytest
    with pytest.raises(ValueError, match='reserved internal prefix'):
        Var('_sz_mm_0', [1.0])
    with pytest.raises(ValueError, match='reserved internal prefix'):
        Var('_sz_mb_0_u0', [1.0])
    with pytest.raises(ValueError, match='reserved internal prefix'):
        Param('_sz_mm_42', [1.0])


def test_triggerable_param_in_matmul_errors():
    """Finding 2: a triggerable parameter used in the same equation as
    ``Mat_Mul`` must raise ``NotImplementedError`` at ``FormJac`` /
    ``create_instance`` time. Silently freezing the triggered values
    would lead to wrong Newton steps once the trigger fires.
    """
    import pytest
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [4.0, 5.0])
    # A triggerable param in the SAME equation as Mat_Mul.
    m.K = Param('K', [1.0, 1.0], triggerable=True,
                trigger_var=['x'], trigger_fun=lambda x: 2 * x)
    m.A = Param('A', [[2, 1], [1, 3]], dim=2, sparse=True)
    m.eqn = Eqn('f', m.K * m.x + Mat_Mul(m.A, m.x) - m.b)

    smdl, _ = m.create_instance()
    from Solverz.variable.variables import Vars
    from Solverz.utilities.address import Address
    # Reuse the existing y0 from create_instance
    _, y0 = m.create_instance()
    with pytest.raises(NotImplementedError, match='Mat_Mul'):
        smdl.FormJac(y0)


def test_dense_dim2_param_warning():
    """Finding 3: dense ``dim=2`` parameters used in Mat_Mul must
    produce a warning pointing at the performance cost. They should
    still work correctly (via the fallback path)."""
    m = Model()
    m.x = Var('x', [0.5, 0.5])
    m.b = Param('b', [4.0, 5.0])
    m.A = Param('A', np.array([[2.0, 1.0], [1.0, 3.0]]),
                dim=2, sparse=False)
    m.eqn = Eqn('f', m.x * Mat_Mul(m.A, m.x) - m.b)

    smdl, y0 = m.create_instance()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        smdl.FormJac(y0)
        dense_warnings = [x for x in w
                          if issubclass(x.category, UserWarning)
                          and 'dense 2-D' in str(x.message)]
        assert len(dense_warnings) >= 1, \
            'expected a UserWarning about dense dim=2 in Mat_Mul'
