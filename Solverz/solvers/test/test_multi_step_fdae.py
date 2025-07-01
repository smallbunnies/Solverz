import numpy as np
from Solverz import fdae_solver, Var, AliasVar, Eqn, made_numerical, Model, Opt


# %%
def test_multi_step_fdae():
    m = Model()
    m.x = Var('x', [1, 2])  # x
    m.x0 = AliasVar('x', value=m.x.value, step=1)  # x_0
    m.x1 = AliasVar('x', value=m.x.value, step=2)  # x_{-1}

    m.f1 = Eqn('f1', m.x[0] + m.x0[0] + m.x1[1])
    m.f2 = Eqn('f2', m.x[1] + m.x0[1] - m.x1[0])

    smdl, y0 = m.create_instance()
    mdl = made_numerical(smdl, y0, sparse=True)

    sol = fdae_solver(mdl,
                      [0, 120],
                      y0,
                      Opt(step_size=60),
                      y1=y0)

    np.testing.assert_allclose(sol.Y[1], np.array([-3., -1.]))
