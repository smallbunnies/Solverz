import warnings

import numpy as np
from Solverz import MatVecMul, Mat_Mul, Var, Param, made_numerical, Model, Eqn, nr_method


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
