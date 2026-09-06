"""Tests for the optional SuiteSparse KLU backend.

Skipped entirely when libklu is not installed, so the default scipy-only
install is unaffected.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import pytest

from Solverz.solvers.klu_backend import KLU_AVAILABLE, klu_decomposition, KLUCache
from Solverz.solvers.laesolver import (
    lu_decomposition, sp_decomposition,
    set_linsolver, get_linsolver, linsolver, resolve_backend,
)

pytestmark = pytest.mark.skipif(not KLU_AVAILABLE, reason="libklu not installed")


def _spd_like(n, seed):
    rng = np.random.default_rng(seed)
    return (sp.random(n, n, density=0.05, format="csc", random_state=rng)
            + sp.eye(n) * 5).tocsc(), rng


def test_klu_matches_superlu():
    A, rng = _spd_like(400, 0)
    b = rng.standard_normal(A.shape[0])
    x_klu = klu_decomposition(A).solve(b)
    x_su = splu(A).solve(b)
    assert np.allclose(x_klu, x_su, atol=1e-10)


def test_symbolic_reuse_same_pattern():
    A, rng = _spd_like(300, 1)
    d1 = klu_decomposition(A)
    A2 = A.copy()
    A2.data *= rng.uniform(0.5, 1.5, size=A.data.shape)  # same pattern, new values
    d2 = klu_decomposition(A2, symbolic=d1.symbolic)
    assert d2.symbolic is d1.symbolic                    # ordering reused, not recomputed
    b = rng.standard_normal(A.shape[0])
    assert np.allclose(d2.solve(b), splu(A2).solve(b), atol=1e-10)


def test_lu_decomposition_dispatch_and_cache():
    A, rng = _spd_like(250, 2)
    b = rng.standard_normal(A.shape[0])
    cache = KLUCache()
    x_klu = lu_decomposition(A, backend="klu", cache=cache).solve(b)
    x_su = lu_decomposition(A, backend="superlu").solve(b)
    assert cache.symbolic is not None
    assert np.allclose(x_klu, x_su, atol=1e-10)
    # second call on a fresh same-pattern matrix reuses the cached symbolic
    A2 = A.copy(); A2.data *= 1.1
    sym0 = cache.symbolic
    lu_decomposition(A2, backend="klu", cache=cache)
    assert cache.symbolic is sym0


def test_complex_falls_back_to_superlu():
    A, rng = _spd_like(120, 3)
    Ac = A.astype(complex)
    Ac.data += 1j * 0.1
    b = rng.standard_normal(A.shape[0]).astype(complex)
    # backend='klu' on a complex matrix must not raise; it routes to scipy
    x = lu_decomposition(Ac, backend="klu").solve(b)
    assert np.allclose(Ac @ x, b, atol=1e-9)


def test_global_selection_and_context_manager():
    prev = get_linsolver()
    try:
        set_linsolver("superlu")
        assert get_linsolver() == "superlu"
        assert resolve_backend(None) == "superlu"
        with linsolver("klu"):                       # scoped override
            assert resolve_backend(None) == "klu"
        assert resolve_backend(None) == "superlu"    # restored after the block
        assert resolve_backend("klu") == "klu"       # explicit arg wins over global

        A, rng = _spd_like(150, 7)
        assert isinstance(lu_decomposition(A), sp_decomposition)   # global superlu
        with linsolver("klu"):
            assert isinstance(lu_decomposition(A), klu_decomposition)
    finally:
        set_linsolver(prev)


# --------------------------------------------------------------------------- #
# maximum-product row matching before the KLU analysis
# --------------------------------------------------------------------------- #
def _empty_diagonal_case(n, seed):
    """A diagonally dominant matrix with its rows scrambled, so the structural
    diagonal is empty, as for a LoopEqn Jacobian."""
    rng = np.random.default_rng(seed)
    B = (sp.random(n, n, density=0.03, format="csc", random_state=rng) + sp.eye(n) * 8).tocsc()
    p = rng.permutation(n)
    return sp.csc_array(B[p]), rng


def test_matching_solves_scrambled_system_like_superlu():
    from Solverz.solvers.klu_backend import klu_matching_enabled
    assert klu_matching_enabled()
    A, rng = _empty_diagonal_case(1500, 7)
    b = rng.standard_normal(A.shape[0])
    dec = klu_decomposition(A)
    assert dec.symbolic.perm is not None
    assert np.count_nonzero(np.abs(sp.csc_array(A[dec.symbolic.perm]).diagonal()) >= 8) == A.shape[0]
    assert np.allclose(dec.solve(b), splu(A).solve(b), atol=1e-10)
    B = rng.standard_normal((A.shape[0], 3))
    assert np.allclose(dec.solve(B), splu(A).solve(B), atol=1e-10)


def test_matching_reused_with_the_symbolic():
    A, rng = _empty_diagonal_case(1200, 8)
    d1 = klu_decomposition(A)
    A2 = A.copy()
    A2.data *= rng.uniform(0.5, 1.5, size=A.data.shape)
    d2 = klu_decomposition(A2, symbolic=d1.symbolic)
    assert d2.symbolic is d1.symbolic
    b = rng.standard_normal(A.shape[0])
    assert np.allclose(d2.solve(b), splu(A2).solve(b), atol=1e-10)
    cache = KLUCache()
    x = lu_decomposition(A2, backend="klu", cache=cache).solve(b)
    assert cache.symbolic.perm is not None
    assert np.allclose(x, splu(A2).solve(b), atol=1e-10)


def test_matching_skipped_below_the_size_threshold():
    from Solverz.solvers import klu_backend as kb
    A, rng = _empty_diagonal_case(300, 11)
    b = rng.standard_normal(A.shape[0])
    assert A.shape[0] < kb.MATCHING_MIN_N
    dec = klu_decomposition(A)                  # default: below the threshold, no matching
    assert dec.symbolic.perm is None
    assert np.allclose(dec.solve(b), splu(A).solve(b), atol=1e-10)
    dec2 = klu_decomposition(A, matching=True)  # explicit request overrides the threshold
    assert dec2.symbolic.perm is not None
    assert np.allclose(dec2.solve(b), splu(A).solve(b), atol=1e-10)


def test_matching_can_be_disabled():
    from Solverz.solvers.klu_backend import set_klu_matching, klu_matching_enabled
    A, rng = _empty_diagonal_case(1200, 9)
    b = rng.standard_normal(A.shape[0])
    set_klu_matching(False)
    try:
        assert not klu_matching_enabled()
        dec = klu_decomposition(A)
        assert dec.symbolic.perm is None
        assert np.allclose(dec.solve(b), splu(A).solve(b), atol=1e-10)
        dec2 = klu_decomposition(A, matching=True)
        assert dec2.symbolic.perm is not None
    finally:
        set_klu_matching(True)
    assert klu_matching_enabled()


def test_matching_reaches_klu_and_removes_off_diagonal_pivots():
    """The permutation must reach KLU: with the dominant entries matched onto
    the diagonal, the numeric factorization never pivots off it, while KLU's
    own transversal of the scrambled matrix leaves it no such guarantee."""
    A, _ = _empty_diagonal_case(2000, 10)
    d_on = klu_decomposition(A, matching=True)
    assert d_on.symbolic.perm is not None
    assert d_on._common.btf == 0
    assert d_on._common.noffdiag == 0
    d_off = klu_decomposition(A, matching=False)
    assert d_off.symbolic.perm is None
    assert d_off._common.btf == 1
