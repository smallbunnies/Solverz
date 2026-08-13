import numpy as np
from scipy.sparse import csc_matrix

from Solverz import AdamsBDF, Opt, PE, Rodas


class TinyDAE:
    def __init__(self, M, F, J):
        self.M = M
        self._F = F
        self._J = J
        self.p = {}

    def F(self, t, y, p):
        return self._F(t, y)

    def J(self, t, y, p):
        return self._J(t, y)


def _linear_dae():
    M = csc_matrix(np.diag([1.0, 0.0]))

    def Fcn(t, y):
        return np.array([-y[1], y[1] - y[0] ** 2])

    def Jcn(t, y):
        return csc_matrix(np.array([
            [0.0, -1.0],
            [-2.0 * y[0], 1.0],
        ]))

    return TinyDAE(M, Fcn, Jcn), np.array([1.0, 1.0])


def _robertson_dae():
    M = csc_matrix(np.diag([1.0, 1.0, 0.0]))

    def Fcn(t, y):
        y1, y2, y3 = y
        return np.array([
            -0.04 * y1 + 1.0e4 * y2 * y3,
            0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
            y1 + y2 + y3 - 1.0,
        ])

    def Jcn(t, y):
        y1, y2, y3 = y
        return csc_matrix(np.array([
            [-0.04, 1.0e4 * y3, 1.0e4 * y2],
            [0.04, -1.0e4 * y3 - 6.0e7 * y2, -1.0e4 * y2],
            [1.0, 1.0, 1.0],
        ]))

    return TinyDAE(M, Fcn, Jcn), np.array([1.0, 0.0, 0.0])


def test_pe_partitioned_accuracy_orders():
    dae, y0 = _linear_dae()
    tend = 5.0
    x_ref = 1.0 / (1.0 + tend)
    hs = [0.02, 0.01, 0.005]

    for scheme, expected_order, max_error in [
        ("euler", (0.8, 1.4), 5e-3),
        ("modified_euler", (1.7, 2.4), 1e-5),
    ]:
        errs = []
        for h in hs:
            sol = PE(
                dae,
                [0.0, tend],
                y0,
                Opt(
                    scheme=scheme,
                    step_size=h,
                    ite_tol=1e-12,
                    rtol=1e-6,
                    atol=1e-9,
                ),
            )
            errs.append(abs(sol.Y[-1, 0] - x_ref))

        order = np.log(errs[1] / errs[2]) / np.log(hs[1] / hs[2])
        assert expected_order[0] < order < expected_order[1]
        assert errs[2] < max_error
        assert abs(sol.Y[-1, 1] - sol.Y[-1, 0] ** 2) < 1e-8


def test_pe_event_and_dense_grid():
    dae, y0 = _linear_dae()

    def event(t, y):
        return np.array([y[0] - 0.5]), np.array([1]), np.array([-1])

    sol_event = PE(
        dae,
        [0.0, 5.0],
        y0,
        Opt(scheme="modified_euler", step_size=0.001, event=event, ite_tol=1e-12),
    )
    assert sol_event.te is not None
    np.testing.assert_allclose(sol_event.te[0], 1.0, atol=1e-3)
    np.testing.assert_allclose(sol_event.T[-1], sol_event.te[0], atol=1e-9)

    grid = np.linspace(0.0, 5.0, 51)
    sol_grid = PE(
        dae,
        grid,
        y0,
        Opt(scheme="modified_euler", step_size=0.001, ite_tol=1e-12),
    )
    np.testing.assert_allclose(sol_grid.T, grid, atol=1e-6)
    np.testing.assert_allclose(sol_grid.Y[:, 0], 1.0 / (1.0 + grid), atol=1e-4)


def test_mixed_adams_bdf_linear_dae_and_order_history():
    dae, y0 = _linear_dae()
    sol = AdamsBDF(dae, [0.0, 5.0], y0, Opt(rtol=1e-5, atol=1e-8))
    x_ref = 1.0 / 6.0
    y_ref = x_ref ** 2

    np.testing.assert_allclose(sol.Y[-1, 0], x_ref, atol=1e-3)
    np.testing.assert_allclose(sol.Y[-1, 1], y_ref, atol=1e-3)
    assert sol.stats.order_variation is not None
    assert sol.stats.order_variation.shape == sol.T.shape
    assert set(np.unique(sol.stats.order_variation)).issubset({1.0, 2.0})


def test_mixed_adams_bdf_robertson_matches_rodas():
    dae, y0 = _robertson_dae()
    sol_ref = Rodas(
        dae,
        [0.0, 0.4],
        y0,
        Opt(rtol=1e-9, atol=1e-12, scheme="rodas3"),
    )
    sol = AdamsBDF(dae, [0.0, 0.4], y0, Opt(rtol=1e-5, atol=1e-9))

    err = np.abs(sol.Y[-1] - sol_ref.Y[-1])
    wt = np.maximum(np.abs(sol_ref.Y[-1]), 1e-6)
    assert np.max(err / wt) < 0.05
