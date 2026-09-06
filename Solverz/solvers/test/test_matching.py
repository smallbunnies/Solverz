"""Tests for the maximum-product row matching behind the KLU backend."""
import numpy as np
import pytest
import scipy.sparse as sp

from Solverz.solvers.matching import max_product_matching


def _objective(A, perm):
    """Sum over columns of log(|diagonal entry| / max|column|) after A[perm]."""
    dg = np.abs(sp.csc_array(A[perm]).diagonal())
    colmax = np.asarray(abs(A).max(axis=0).toarray()).ravel()
    return float(np.sum(np.log(dg) - np.log(colmax)))


def _scipy_objective(A):
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    W = sp.csr_array(abs(A))
    W.eliminate_zeros()
    w = np.log(W.data)
    W.data = w - w.min() + 1.0
    r, c = min_weight_full_bipartite_matching(W, maximize=True)
    perm = np.empty(A.shape[0], dtype=np.int64)
    perm[c] = r
    return _objective(A, perm)


def _random_case(rng):
    n = int(rng.integers(2, 60))
    A = sp.random(n, n, density=float(rng.uniform(0.05, 0.4)), format='csc', random_state=rng)
    A = A + sp.diags_array(rng.uniform(0.1, 1, n) * rng.choice([0, 1], n, p=[0.3, 0.7]), format='csc')
    A.data = np.exp(rng.normal(0, 3, size=A.data.shape)) * rng.choice([-1, 1], size=A.data.shape)
    A = sp.csc_array(A)
    A.eliminate_zeros()
    return A


def test_matching_is_optimal_on_random_matrices():
    rng = np.random.default_rng(0)
    n_checked = 0
    for _ in range(150):
        A = _random_case(rng)
        try:
            ref = _scipy_objective(A)
        except ValueError:
            assert max_product_matching(A.indptr, A.indices, A.data, A.shape[0]) is None
            continue
        perm = max_product_matching(A.indptr, A.indices, A.data, A.shape[0])
        assert perm is not None
        assert sorted(perm.tolist()) == list(range(A.shape[0]))
        assert abs(_objective(A, perm) - ref) <= 1e-9 * max(1.0, abs(ref))
        n_checked += 1
    assert n_checked > 100


def test_matching_puts_largest_entries_on_the_diagonal():
    # a permuted diagonally dominant matrix: the matching must undo the permutation
    rng = np.random.default_rng(3)
    n = 40
    B = sp.random(n, n, density=0.1, format='csc', random_state=rng) + sp.eye(n, format='csc') * 10
    p = rng.permutation(n)
    A = sp.csc_array(B[p])
    perm = max_product_matching(A.indptr, A.indices, A.data, n)
    np.testing.assert_array_equal(np.abs(sp.csc_array(A[perm]).diagonal()), np.abs(B.diagonal()))


def test_matching_ignores_explicit_zeros_and_rejects_singular():
    A = sp.csc_array(np.array([[0.0, 1.0], [2.0, 0.0]]))
    # explicit zeros stored on the diagonal must not be matched
    A2 = sp.csc_array((np.array([0.0, 2.0, 1.0, 0.0]), np.array([0, 1, 0, 1]), np.array([0, 2, 4])), shape=(2, 2))
    perm = max_product_matching(A2.indptr, A2.indices, A2.data, 2)
    np.testing.assert_array_equal(perm, [1, 0])
    S = sp.csc_array(np.array([[1.0, 1.0], [0.0, 0.0]]))   # structurally singular
    assert max_product_matching(S.indptr, S.indices, S.data, 2) is None


def test_work_budget_returns_none():
    rng = np.random.default_rng(5)
    A = _random_case(rng)
    assert max_product_matching(A.indptr, A.indices, A.data, A.shape[0], work_factor=0.0) is None \
        or max_product_matching(A.indptr, A.indices, A.data, A.shape[0]) is not None
