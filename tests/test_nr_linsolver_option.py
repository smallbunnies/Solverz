"""``nr_method`` honours ``Opt(linsolver=...)`` like every other solver (issue #158)."""
import numpy as np
import scipy.sparse as sps
import pytest

from Solverz import nr_method, Opt
from Solverz.num_api.num_eqn import nAE
from Solverz.solvers.klu_backend import KLU_AVAILABLE
from Solverz.solvers.laesolver import model_cache, set_linsolver, get_linsolver


def _linear_ae(n, seed):
    rng = np.random.default_rng(seed)
    A = (sps.random(n, n, density=3.0 / n, format='csc', random_state=rng) + sps.eye(n) * 8).tocsc()
    b = rng.standard_normal(n)
    return nAE(lambda y, p: A @ y - b, lambda y, p: A, {}), A, b


@pytest.mark.skipif(not KLU_AVAILABLE, reason="libklu not installed")
def test_opt_linsolver_selects_the_newton_backend():
    prev = get_linsolver()
    try:
        set_linsolver('klu')
        ae, A, b = _linear_ae(300, 0)
        sol = nr_method(ae, np.zeros(300), Opt(ite_tol=1e-10, linsolver='superlu'))
        assert np.allclose(A @ sol.y, b, atol=1e-8)
        assert model_cache(ae).symbolic is None          # KLU did not run on this solve

        set_linsolver('superlu')
        ae, A, b = _linear_ae(300, 1)
        sol = nr_method(ae, np.zeros(300), Opt(ite_tol=1e-10, linsolver='klu'))
        assert np.allclose(A @ sol.y, b, atol=1e-8)
        assert model_cache(ae).symbolic is not None      # KLU ran on request

        ae, A, b = _linear_ae(300, 2)
        sol = nr_method(ae, np.zeros(300), Opt(ite_tol=1e-10))
        assert np.allclose(A @ sol.y, b, atol=1e-8)
        assert model_cache(ae).symbolic is None          # None follows the global selection
    finally:
        set_linsolver(prev)
