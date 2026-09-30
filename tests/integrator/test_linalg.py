"""The iteration matrix ``W = M - (dt*gamma) J``.

I2: the assembly is the chain of legacy Rodas, byte for byte, and neither it
nor the factorization touches ``J``, even one with unsorted rows and a stored
zero whose arrays another ``J`` shares; a legacy-compatible call starts from
an empty ``KLUCache`` of its own; a solve into ``out`` is byte-equal to a
solve into a new array, and both to legacy's ``lu.solve(rscale * rhs)``, on
each backend, with and without the row matching of the KLU analysis; a
strided ``out`` is refused; a singular ``W`` and a failing ``klu_solve``
reject the attempt through ``StepFailure``; factorizations and solves are
counted.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_array, issparse

from Solverz.integrator import StepFailure
from Solverz.integrator.linalg import IterationMatrix
from Solverz.solvers import klu_backend
from Solverz.solvers.daesolver.rodas.param import Rodas_param
from Solverz.solvers.klu_backend import KLU_AVAILABLE, KLUCache, klu_decomposition, set_klu_matching
from Solverz.solvers.laesolver import linsolver, lu_decomposition, model_cache, resolve_backend
from Solverz.solvers.stats import Stats

from tests.integrator import models
from tests.integrator.legacy_rodas import legacy_iteration_matrix

GAMMAS = [Rodas_param('rodas4').gamma, Rodas_param('rodas5p').gamma]


def _byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _integ(n, legacy_compat=True, dae=None):
    """The fields of an Integrator that the iteration matrix reads."""
    return SimpleNamespace(n=n, dae=dae, stats=Stats('test'),
                           opts=SimpleNamespace(linsolver=resolve_backend(None),
                                                legacy_compat=legacy_compat))


def _point(dae, y0):
    """A state away from the initial one, and the Jacobian there."""
    t = 0.3
    y = y0 * 1.1 + 0.05
    return t, y, dae.J(t, y, dae.p)


def _snapshot(J):
    if issparse(J):
        return J.data.copy(), J.indices.copy(), J.indptr.copy()
    return (J.copy(),)


def _same_snapshot(a, b):
    return len(a) == len(b) and all(_byte_equal(x, y) for x, y in zip(a, b))


@pytest.fixture
def klu_matching_off():
    saved = (klu_backend._MATCHING, klu_backend.MATCHING_MIN_N)
    set_klu_matching(False)
    try:
        yield
    finally:
        set_klu_matching(*saved)


ASSEMBLY = [('dae_test', 'inline_sparse', ()),
            ('dae_test', 'inline_dense', ()),
            ('dae_test', 'rendered', ()),
            ('forced', 'inline_sparse', ()),
            ('forced', 'inline_dense', ()),
            ('trace', 'rendered', ()),
            ('permuted', 'inline_sparse', ()),
            ('ladder', 'inline_sparse', (12,)),
            ('ladder', 'rendered', (12,))]


@pytest.mark.i2
@pytest.mark.filterwarnings('ignore:Mutable Jacobian block:UserWarning')
@pytest.mark.parametrize('dt', [1e-3, 0.37, np.int64(2)], ids=['1e-3', '0.37', 'int'])
@pytest.mark.parametrize('name, variant, args', ASSEMBLY, ids=[f"{a[0]}-{a[1]}" for a in ASSEMBLY])
def test_build_is_the_legacy_chain(model, name, variant, args, dt):
    dae, y0 = model(name, variant, *args)
    t, y, J = _point(dae, y0)
    before = _snapshot(J)
    im = IterationMatrix(_integ(y0.size))
    for gamma in GAMMAS:
        Wm, rscale, dtgamma = im.build(dae.M, J, dt, gamma)
        assert _same_snapshot(before, _snapshot(J)), 'the assembly modified J'
        ref, ref_rscale = legacy_iteration_matrix(dae.M, J, dt, gamma, SimpleNamespace(rscale_to_dense=None))
        assert _byte_equal(rscale, ref_rscale)
        assert dtgamma == dt * gamma and type(dtgamma) is type(dt * gamma)
        assert type(Wm) is type(ref)
        if issparse(ref):
            assert Wm.format == ref.format == 'csc' and Wm.shape == ref.shape
            assert _byte_equal(Wm.data, ref.data)
            assert _byte_equal(Wm.indices, ref.indices)
            assert _byte_equal(Wm.indptr, ref.indptr)
        else:
            assert _byte_equal(np.asarray(Wm), np.asarray(ref))
    assert im.rscale_to_dense == issparse(dae.M - J)
    # a rendered model hands out one indices and one indptr array with every J
    assert _same_snapshot(before, _snapshot(dae.J(t, y, dae.p)))


def _shared_noncanonical_jacobians():
    """``(M, J1, J2)``: two Jacobians that share one ``indices`` and one
    ``indptr`` array, as those of a rendered model do, with the rows of every
    column unsorted and, in ``J1``, a stored exact zero, as a pinned switched
    row gives. ``sort_indices``, ``eliminate_zeros`` and ``sum_duplicates``
    each rewrite these arrays in place. ``M`` keeps an explicit zero on its
    algebraic row, as ``ModeSwitch`` does."""
    indptr = np.array([0, 3, 5, 7], dtype=np.int32)
    indices = np.array([2, 0, 1, 1, 0, 2, 1], dtype=np.int32)
    J1 = csc_array((np.array([0.5, -2.0, 1.0, 0.0, 0.3, -1.5, 0.7]), indices, indptr), shape=(3, 3))
    J2 = csc_array((np.array([0.5, -2.0, 1.0, 0.4, 0.3, -1.5, 0.7]), indices, indptr), shape=(3, 3))
    M = csc_array((np.array([1.0, 0.0, 1.0]), np.arange(3, dtype=np.int32),
                   np.arange(4, dtype=np.int32)), shape=(3, 3))
    return M, J1, J2


@pytest.mark.i2
def test_build_leaves_a_shared_noncanonical_J_untouched(backend):
    M, J1, J2 = _shared_noncanonical_jacobians()
    assert np.shares_memory(J1.indices, J2.indices) and np.shares_memory(J1.indptr, J2.indptr)
    assert not J1.has_sorted_indices and 0.0 in J1.data and 0.0 in M.data
    before = [_snapshot(A) for A in (M, J1, J2)]

    def untouched():
        return all(_same_snapshot(b, _snapshot(A)) for b, A in zip(before, (M, J1, J2)))

    im = IterationMatrix(_integ(3))
    for J in (J1, J2):
        Wm, rscale, _ = im.build(M, J, 0.1, GAMMAS[0])
        assert untouched(), 'the assembly modified M or a Jacobian'
        ref, ref_rscale = legacy_iteration_matrix(M, J, 0.1, GAMMAS[0], SimpleNamespace(rscale_to_dense=None))
        assert _byte_equal(Wm.data, ref.data) and _byte_equal(Wm.indices, ref.indices)
        assert _byte_equal(Wm.indptr, ref.indptr) and _byte_equal(rscale, ref_rscale)
        W = im.factorize(M, J, 0.1, GAMMAS[0])
        W.solve(np.ones(3))
        assert untouched(), 'the factorization modified M or a Jacobian'
    assert im.stats.ndecomp == 2 and im.stats.nsolve == 2


@pytest.mark.i2
def test_build_hands_over_descending_row_indices(model):
    """The row scaling emits each column's rows in descending order, and the
    factorization receives them so; a sorted assembly would fail here."""
    dae, y0 = model('ladder', 'inline_sparse', 12)
    t, y, J = _point(dae, y0)
    Wm, _, _ = IterationMatrix(_integ(y0.size)).build(dae.M, J, 0.1, GAMMAS[0])
    cols = [Wm.indices[Wm.indptr[j]:Wm.indptr[j + 1]] for j in range(Wm.shape[1])]
    assert any(np.any(np.diff(c) < 0) for c in cols)


@pytest.mark.i2
def test_build_drops_an_exact_zero_as_legacy_does():
    M = csc_array(np.eye(2))
    J = csc_array(np.array([[2.0, 1.0], [0.0, 3.0]]))
    Wm, rscale, _ = IterationMatrix(_integ(2)).build(M, J, 2.0, 0.25)
    ref, ref_rscale = legacy_iteration_matrix(M, J, 2.0, 0.25, SimpleNamespace(rscale_to_dense=None))
    assert Wm.nnz == 2 and J.nnz == 3
    assert _byte_equal(Wm.data, ref.data) and _byte_equal(Wm.indices, ref.indices)
    assert _byte_equal(Wm.indptr, ref.indptr) and _byte_equal(rscale, ref_rscale)


@pytest.mark.i2
def test_legacy_compatible_call_starts_from_an_empty_cache(backend):
    dae, y0 = models.build('dae_test')
    t, y, J = _point(dae, y0)
    shared = model_cache(dae)
    lu_decomposition(csc_array(dae.M - 0.1 * J), backend=backend, cache=shared)
    held = (shared.symbolic, shared.superlu)
    assert held != (None, None)
    im = IterationMatrix(_integ(y0.size, legacy_compat=True, dae=dae))
    assert im.cache is not shared
    assert im.cache.symbolic is None and im.cache.superlu is None
    W = im.factorize(dae.M, J, 0.1, GAMMAS[0])
    assert (shared.symbolic, shared.superlu) == held
    if backend == 'klu':
        assert im.cache.symbolic is W.lu.symbolic
    else:
        assert im.cache.superlu is not None
    # a cache shared across calls would already be filled here
    second = IterationMatrix(_integ(y0.size, legacy_compat=True, dae=dae))
    assert second.cache is not im.cache and second.cache is not shared
    assert second.cache.symbolic is None and second.cache.superlu is None


SOLVE = [('dae_test', 'inline_sparse', ()),
         ('dae_test', 'inline_dense', ()),
         ('dae_test', 'rendered', ()),
         ('permuted', 'inline_sparse', ()),
         ('ladder', 'inline_sparse', (12,))]


def _check_solve(model, backend, name, variant, args, matching):
    dae, y0 = model(name, variant, *args)
    n = y0.size
    t, y, J = _point(dae, y0)
    im = IterationMatrix(_integ(n))
    W = im.factorize(dae.M, J, 0.1, GAMMAS[0])
    if backend == 'klu' and issparse(J):
        assert isinstance(W.lu, klu_decomposition)
        assert (W.lu.symbolic.perm is not None) == matching

    rng = np.random.default_rng(n)
    for b in (rng.standard_normal(n), dae.F(t, y, dae.p)):
        keep = b.copy()
        x = W.solve(b)
        assert x.dtype == np.float64 and x.shape == (n,)
        buf = np.full(n, np.nan)
        assert W.solve(b, out=buf) is buf
        assert _byte_equal(buf, x)
        assert _byte_equal(b, keep)

        Wm, rscale = legacy_iteration_matrix(dae.M, J, 0.1, GAMMAS[0], SimpleNamespace(rscale_to_dense=None))
        lu = lu_decomposition(Wm, backend=backend, cache=KLUCache())
        assert _byte_equal(x, np.asarray(lu.solve(rscale * b), dtype=np.float64))

        c = b.copy()
        assert W.solve(c, out=c) is c
        assert _byte_equal(c, x)


@pytest.mark.i2
@pytest.mark.parametrize('name, variant, args', SOLVE, ids=[f"{a[0]}-{a[1]}" for a in SOLVE])
def test_solve_into_out_is_byte_equal(model, backend, klu_matching_off, name, variant, args):
    _check_solve(model, backend, name, variant, args, matching=False)


@pytest.mark.i2
@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('name, variant, args', [a for a in SOLVE if a[1] != 'inline_dense'],
                         ids=[f"{a[0]}-{a[1]}" for a in SOLVE if a[1] != 'inline_dense'])
def test_solve_into_out_is_byte_equal_under_the_row_matching(model, klu_matching_low, name, variant, args):
    with linsolver('klu'):
        _check_solve(model, 'klu', name, variant, args, matching=True)


@pytest.mark.i2
def test_out_must_be_a_contiguous_float64_vector(model, backend):
    dae, y0 = model('ladder', 'inline_sparse', 12)
    n = y0.size
    t, y, J = _point(dae, y0)
    im = IterationMatrix(_integ(n))
    W = im.factorize(dae.M, J, 0.1, GAMMAS[0])
    b = np.ones(n)
    K = np.zeros((n, 6))
    readonly = np.zeros(n)
    readonly.flags.writeable = False
    for out in (K[:, 0], np.zeros(n + 1), np.zeros(n, dtype=np.float32), np.zeros((n, 1)), readonly,
                list(range(n))):
        with pytest.raises(ValueError):
            W.solve(b, out=out)
    with pytest.raises(ValueError):
        W.solve(np.ones(n + 1))
    with pytest.raises(ValueError):
        W.solve(np.ones((n, 1)))
    assert im.stats.nsolve == 0
    W.solve(b, out=K[:, 0].copy())
    assert im.stats.nsolve == 1


def _singular():
    """``W = -(dt*gamma) J`` with ``J`` of rank one, exactly singular after the
    row scaling."""
    return csc_array((2, 2)), csc_array(np.array([[1.0, 1.0], [1.0, 1.0]]))


@pytest.mark.i2
def test_a_singular_W_rejects_the_attempt(backend):
    M, J = _singular()
    im = IterationMatrix(_integ(2))
    with pytest.raises(StepFailure, match='factorization of W failed') as info:
        im.factorize(M, J, 0.1, 0.25)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert im.stats.ndecomp == 0

    # the dense path factorizes inside the solve
    im = IterationMatrix(_integ(2))
    W = im.factorize(M, J.toarray(), 0.1, 0.25)
    assert im.stats.ndecomp == 1
    with pytest.raises(StepFailure, match='solve with W failed') as info:
        W.solve(np.ones(2))
    assert isinstance(info.value.__cause__, np.linalg.LinAlgError)
    with pytest.raises(StepFailure):
        W.solve(np.ones(2), out=np.empty(2))
    assert im.stats.nsolve == 0


@pytest.mark.i2
@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
def test_a_failing_klu_solve_rejects_the_attempt(model):
    dae, y0 = model('dae_test')
    t, y, J = _point(dae, y0)
    with linsolver('klu'):
        im = IterationMatrix(_integ(y0.size))
    W = im.factorize(dae.M, J, 0.1, GAMMAS[0])
    lu = W.lu
    assert isinstance(lu, klu_decomposition)
    b = np.ones(y0.size)
    x = W.solve(b)
    # klu_solve refuses a missing numeric factorization with KLU_INVALID
    num, lu._num = lu._num, None
    try:
        with pytest.raises(StepFailure, match='klu_solve failed') as info:
            W.solve(b)
        assert isinstance(info.value.__cause__, RuntimeError)
        with pytest.raises(StepFailure, match='klu_solve failed'):
            W.solve(b, out=np.empty(y0.size))
    finally:
        lu._num = num
    assert im.stats.nsolve == 1
    assert _byte_equal(W.solve(b), x)
    assert im.stats.nsolve == 2


@pytest.mark.i2
@pytest.mark.parametrize('variant', ['inline_sparse', 'inline_dense'])
def test_factorizations_and_solves_are_counted(model, backend, variant):
    dae, y0 = model('dae_test', variant)
    t, y, J = _point(dae, y0)
    im = IterationMatrix(_integ(y0.size))
    assert (im.stats.ndecomp, im.stats.nsolve) == (0, 0)
    W1 = im.factorize(dae.M, J, 0.1, GAMMAS[0])
    W2 = im.factorize(dae.M, J, 0.1, GAMMAS[1])
    assert im.stats.ndecomp == 2
    b = np.ones(y0.size)
    W1.solve(b)
    W1.solve(b, out=np.empty(y0.size))
    W2.solve(b)
    assert im.stats.nsolve == 3
    assert W1.dtgamma == 0.1 * GAMMAS[0] and W2.dtgamma == 0.1 * GAMMAS[1]
    assert W1.rscale.shape == (y0.size,)


def _antidiagonal(n):
    """A matrix whose largest entries lie on the antidiagonal, so that the
    maximum-product row matching is not the identity."""
    A = np.eye(n) * 0.5 + np.fliplr(np.eye(n)) * 10.0 + np.eye(n, k=1) * 0.25
    return csc_array(A)


@pytest.mark.i2
@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('matching', [False, True], ids=['plain', 'matching'])
def test_klu_solve_into_is_byte_equal_to_solve(matching):
    n = 7
    dec = klu_decomposition(_antidiagonal(n), matching=matching)
    perm = dec.symbolic.perm
    assert (perm is not None) == matching
    if matching:
        assert not np.array_equal(perm, np.arange(n))
    b = np.random.default_rng(3).standard_normal(n)
    keep = b.copy()
    out = np.full(n, np.nan)
    assert dec.solve_into(b, out) is out
    assert _byte_equal(out, dec.solve(b))
    assert _byte_equal(b, keep)
    for bad in (np.zeros((n, 2))[:, 0], np.zeros(n - 1), np.zeros(n, dtype=np.float32)):
        with pytest.raises(ValueError):
            dec.solve_into(b, bad)
    for bad in (np.zeros(n + 1), np.zeros(1), np.zeros(n, dtype=np.int64)):
        with pytest.raises(ValueError):
            dec.solve_into(bad, out)
