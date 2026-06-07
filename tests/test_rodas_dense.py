"""Regression: Rodas must run with a DENSE iteration matrix.

The row-equilibration step (1/max|row| before the LU) assumed a sparse
iteration matrix and called ``.toarray()`` on the row reduction. With
``made_numerical(..., sparse=False)`` the iteration matrix is a dense
ndarray, so that raised ``AttributeError: 'numpy.ndarray' object has no
attribute 'toarray'``. This test exercises the dense Rodas path.
"""
import numpy as np

from Solverz import Model, Var, Ode, made_numerical, Rodas, Opt


def test_rodas_runs_with_dense_jacobian():
    m = Model()
    m.x = Var('x', [1.0, 2.0])
    m.decay = Ode('decay', f=-m.x, diff_var=m.x)
    sdae, y0 = m.create_instance()

    tspan = np.linspace(0.0, 1.0, 11)
    opt = Opt(rtol=1e-6, atol=1e-8)
    # The dense path is the regression: before the fix this raised
    # AttributeError in the row-equilibration step.
    sol_den = Rodas(made_numerical(sdae, y0, sparse=False), tspan, y0, opt)
    sol_sp = Rodas(made_numerical(sdae, y0, sparse=True), tspan, y0, opt)

    # Dense matches sparse (row scaling leaves the solution unchanged)...
    np.testing.assert_allclose(np.asarray(sol_den.Y['x']),
                               np.asarray(sol_sp.Y['x']), rtol=1e-8, atol=1e-10)
    # ...and both match the analytic decay x(t) = x0 * exp(-t).
    np.testing.assert_allclose(np.asarray(sol_den.Y['x'])[-1],
                               np.array([1.0, 2.0]) * np.exp(-1.0),
                               rtol=1e-4, atol=1e-6)
