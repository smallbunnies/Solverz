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

I5, the default configuration: a KLU analysis without a row matching is
shared with later calls on the same model through ``model_cache(dae)``, and
a second call reuses the same object while the pattern is unchanged; an
analysis with a matching never leaves its call, nor does one without a
matching once the current setting would compute a matching, so under the
matching two calls from the same state are byte-equal; the SuperLU ordering
starts empty in every call, in both configurations; a dense ``W`` is
factorized once by LAPACK, agrees with ``np.linalg.solve``, and a singular
one fails at the factorization.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_array, issparse

from Solverz.integrator import Rodas4, StepFailure, init, solve
from Solverz.integrator.linalg import DenseLU, IterationMatrix
from Solverz.solvers import klu_backend
from Solverz.solvers.daesolver.rodas.param import Rodas_param
from Solverz.solvers.klu_backend import KLU_AVAILABLE, KLUCache, klu_decomposition, set_klu_matching
from Solverz.solvers.laesolver import linsolver, lu_decomposition, model_cache, resolve_backend
from Solverz.solvers.option import Opt
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
    # the scaling by rscale broadcasts a b of length 1 or a scalar without an
    # error, so only the check of b's shape refuses them
    for b_bad in (np.ones(n + 1), np.ones((n, 1)), np.ones(1), np.float64(1.0)):
        with pytest.raises(ValueError, match='b must be a vector of length'):
            W.solve(b_bad)
    assert im.stats.nsolve == 0
    W.solve(b, out=K[:, 0].copy())
    assert im.stats.nsolve == 1


@pytest.mark.i2
@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
def test_a_klu_solve_goes_through_solve_into(model, monkeypatch):
    """Under KLU every solve writes into its target through ``solve_into``;
    ``klu_decomposition.solve`` would allocate the result and copy it."""
    dae, y0 = model('ladder', 'inline_sparse', 12)
    n = y0.size
    t, y, J = _point(dae, y0)
    with linsolver('klu'):
        im = IterationMatrix(_integ(n))
    W = im.factorize(dae.M, J, 0.1, GAMMAS[0])
    assert isinstance(W.lu, klu_decomposition)
    b = dae.F(t, y, dae.p)
    ref = W.lu.solve(W.rscale * b)

    def allocating(self, rhs):
        raise AssertionError('klu_decomposition.solve was called')

    monkeypatch.setattr(klu_decomposition, 'solve', allocating)
    assert _byte_equal(W.solve(b), ref)
    buf = np.full(n, np.nan)
    assert W.solve(b, out=buf) is buf and _byte_equal(buf, ref)


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


# -- I5: the default configuration ----------------------------------------------


def _same_run(a, b):
    return _byte_equal(a.T, b.T) and _byte_equal(a.Y, b.Y)


def _second_state():
    """A consistent initial state of ``dae_test`` and ``permuted`` other than ``x = y = 1``."""
    return np.array([0.8, np.sqrt(2.0 - 0.8 * 0.8)])


@pytest.mark.i5
@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
def test_a_default_call_reuses_the_analysis_of_an_earlier_one():
    dae, y0 = models.build('dae_test')
    assert not (klu_backend.klu_matching_enabled() and y0.size >= klu_backend.MATCHING_MIN_N)
    opt = Opt(rtol=1e-6, atol=1e-8)
    shared = model_cache(dae)
    with linsolver('klu'):
        first = init(dae, [0, 20], y0, opt=opt)
        assert first.linalg.cache.symbolic is None and shared.symbolic is None
        sol1 = first.solve()
        sym = shared.symbolic
        assert sym is not None and sym.perm is None and first.linalg.cache.symbolic is sym

        second = init(dae, [0, 20], y0, opt=opt)
        assert second.linalg.cache.symbolic is sym
        sol2 = second.solve()
        # the pattern of W never changes on dae_test, so no call analyses again
        assert second.linalg.cache.symbolic is sym and shared.symbolic is sym
        assert second.stats.ndecomp > 0 and _same_run(sol1, sol2)

        # a model of its own analyses afresh and gives the same bits
        fresh, _ = models.build('dae_test')
        sol3 = solve(fresh, [0, 20], y0, opt=opt)
        assert model_cache(fresh).symbolic is not sym and _same_run(sol1, sol3)
        # the legacy-compatible configuration neither reads nor writes the shared analysis
        legacy = init(dae, [0, 20], y0, alg=Rodas4(legacy_compat=True), opt=opt)
        assert legacy.linalg.cache.symbolic is None
        legacy.solve()
        assert shared.symbolic is sym


@pytest.mark.i5
@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
def test_an_analysis_with_a_row_matching_never_leaves_its_call(klu_matching_low):
    dae, y0 = models.build('permuted')
    opt = Opt(rtol=1e-6, atol=1e-8)
    grid = np.linspace(0, 20, 201)
    shared = model_cache(dae)
    with linsolver('klu'):
        runs, analyses = [], []
        for y in (y0, _second_state(), y0):
            integ = init(dae, grid, y, opt=opt)
            assert integ.linalg.cache.symbolic is None
            runs.append(integ.solve())
            sym = integ.linalg.cache.symbolic
            assert sym.perm is not None and not np.array_equal(sym.perm, np.arange(y0.size))
            assert all(sym is not a for a in analyses)
            analyses.append(sym)
            assert shared.symbolic is None
        assert _same_run(runs[0], runs[2]) and not _same_run(runs[0], runs[1])

        # an analysis without a matching, stored while the matching was off,
        # is not taken once the setting would compute one
        set_klu_matching(False)
        solve(dae, grid, y0, opt=opt)
        assert shared.symbolic is not None and shared.symbolic.perm is None
        set_klu_matching(True, min_n=2)
        integ = init(dae, grid, y0, opt=opt)
        assert integ.linalg.cache.symbolic is None
        assert _same_run(integ.solve(), runs[0])


@pytest.mark.i5
@pytest.mark.parametrize('legacy_compat', [False, True], ids=['default', 'compatible'])
def test_the_superlu_ordering_starts_empty_in_every_call(legacy_compat):
    dae, y0 = models.build('dae_test')
    shared = model_cache(dae)
    runs = []
    with linsolver('superlu'):
        for _ in range(2):
            integ = init(dae, [0, 20], y0, alg=Rodas4(legacy_compat=legacy_compat), opt=Opt(rtol=1e-6, atol=1e-8))
            assert integ.linalg.cache.superlu is None
            runs.append(integ.solve())
            assert integ.linalg.cache.superlu is not None
            assert shared.superlu is None and shared.symbolic is None
    assert _same_run(*runs)


DENSE = [('dae_test', ()), ('forced', ()), ('trace', ()), ('vdp', ())]


@pytest.mark.i5
@pytest.mark.parametrize('name, args', DENSE, ids=[d[0] for d in DENSE])
def test_the_default_dense_factorization(model, name, args):
    dae, y0 = model(name, 'inline_dense', *args)
    n = y0.size
    t, y, J = _point(dae, y0)
    im = IterationMatrix(_integ(n, legacy_compat=False, dae=dae))
    rng = np.random.default_rng(n)
    for k, gamma in enumerate(GAMMAS):
        W = im.factorize(dae.M, J, 0.1, gamma)
        assert isinstance(W.lu, DenseLU) and im.stats.ndecomp == k + 1
        Wm, rscale = legacy_iteration_matrix(dae.M, J, 0.1, gamma, SimpleNamespace(rscale_to_dense=None))
        for b in (rng.standard_normal(n), dae.F(t, y, dae.p)):
            keep = b.copy()
            x = W.solve(b)
            ref = np.linalg.solve(np.asarray(Wm), rscale * b)
            assert np.max(np.abs(x - ref)) <= 1e-12 * np.max(np.abs(ref))
            assert _byte_equal(b, keep)
            buf = np.full(n, np.nan)
            assert W.solve(b, out=buf) is buf and _byte_equal(buf, x)
            c = b.copy()
            assert W.solve(c, out=c) is c and _byte_equal(c, x)
    assert im.stats.nsolve == 3 * 2 * len(GAMMAS)


@pytest.mark.i5
def test_a_singular_dense_W_fails_at_its_factorization():
    M, J = _singular()
    im = IterationMatrix(_integ(2, legacy_compat=False))
    with pytest.raises(StepFailure, match='factorization of W failed') as info:
        im.factorize(M, J.toarray(), 0.1, 0.25)
    assert isinstance(info.value.__cause__, np.linalg.LinAlgError)
    assert im.stats.ndecomp == 0
