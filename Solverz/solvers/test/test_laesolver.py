"""SuperLU wrapper of ``lu_decomposition``: the factor copies are read on demand (issue #159)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from Solverz.solvers.laesolver import lu_decomposition, sp_decomposition


def _system(n, seed):
    rng = np.random.default_rng(seed)
    A = (sp.random(n, n, density=0.05, format="csc", random_state=rng) + sp.eye(n) * 5).tocsc()
    return A, rng.standard_normal(n)


def test_sp_decomposition_reads_the_factors_lazily():
    A, b = _system(200, 0)
    dec = lu_decomposition(A, backend="superlu")
    assert isinstance(dec, sp_decomposition)
    assert 'L' not in dec.__dict__ and 'U' not in dec.__dict__ and 'nnz' not in dec.__dict__
    assert np.allclose(A @ dec.solve(b), b, atol=1e-10)
    assert 'L' not in dec.__dict__                       # solving does not build the copies
    ref = splu(A)
    assert dec.nnz == ref.nnz
    assert np.array_equal(dec.perm_r, ref.perm_r) and np.array_equal(dec.perm_c, ref.perm_c)
    # the factors are still available, and L U reproduces the permuted matrix
    P = sp.csc_array((np.ones(A.shape[0]), (dec.perm_r, np.arange(A.shape[0]))))
    Q = sp.csc_array((np.ones(A.shape[0]), (np.arange(A.shape[0]), dec.perm_c)))
    assert abs((dec.L @ dec.U) - (P @ A @ Q)).max() < 1e-10
    assert 'L' in dec.__dict__ and dec.L is dec.L          # read once, then kept
