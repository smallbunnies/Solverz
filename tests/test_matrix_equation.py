import numpy as np
from Solverz import Mat_Mul, Var, Param, made_numerical, Model, Eqn, nr_method


# %%
def test_matrix_equation1():
    """A@x-b=0"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    m.eqnf = Eqn('eqnf', m.b - Mat_Mul(m.A, m.x))

    # %%
    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    # %%
    sol = nr_method(mdl, y0)

    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))


def test_matrix_equation2():
    """-A@x+b=0"""
    m = Model()
    m.x = Var('x', [0, 0])
    m.b = Param('b', [0.5, 1])
    m.A = Param('A', [[1, 3], [-1, 2]], dim=2, sparse=True)
    m.eqnf = Eqn('eqnf', - m.b + Mat_Mul(m.A, m.x))

    # %%
    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    # %%
    sol = nr_method(mdl, y0)

    np.testing.assert_allclose(sol.y.array, np.array([-0.4, 0.3]))
