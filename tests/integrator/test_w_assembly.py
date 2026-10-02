"""The assembly of ``W`` on the union pattern of ``M`` and ``J``, in the default configuration.

Against the SciPy chain of legacy Rodas, which the legacy-compatible
configuration keeps: every entry that the chain stores has the same bits,
every entry that it drops is an explicit zero, and ``rscale`` is the same, on
the models of ``models.py`` inline and rendered and on hand-built matrices
with a singular ``M``, duplicates in ``M`` and in ``J``, summed in storage
order and found also when they are not adjacent, exact cancellation and
non-finite entries, and at every assembly on one pattern. The pattern is
recognized by memory for a rendered ``J``, only for the whole of a buffer,
and by value otherwise, rebuilt when ``J`` changes its indices or only its
``indptr`` and when the rows of ``M`` change in place before ``reset``, and
the values of ``M`` are read at every assembly, also after
``model_modified``. Two factorizations of different ``gamma`` alive in one
attempt do not disturb each other. The read-only pattern arrays shared by
every ``W`` of a call are unchanged after factorizations on both backends.
The default configuration integrates the models as the chain does, to the
tolerance, and the legacy-compatible configuration never leaves the chain.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_array, csr_array

import Solverz.integrator.linalg as linalg
from Solverz.integrator import Algorithm, Rodas4, init
from Solverz.integrator.linalg import IterationMatrix
from Solverz.integrator.testing import MAX_DY_IN_RTOL, MAX_STEP_DIFFERENCE
from Solverz.solvers.daesolver.rodas.param import Rodas_param
from Solverz.solvers.klu_backend import KLU_AVAILABLE
from Solverz.solvers.laesolver import linsolver, resolve_backend
from Solverz.solvers.option import Opt
from Solverz.solvers.stats import Stats

from tests.integrator import models
from tests.integrator.legacy_rodas import legacy_iteration_matrix

GAMMAS = [Rodas_param('rodas4').gamma, Rodas_param('rodas5p').gamma]


def _byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _matrix(n, legacy_compat=False, dae=None):
    """An ``IterationMatrix`` on the fields of an Integrator that it reads."""
    integ = SimpleNamespace(n=n, dae=dae, stats=Stats('test'),
                            opts=SimpleNamespace(linsolver=resolve_backend(None), legacy_compat=legacy_compat))
    return IterationMatrix(integ)


def _point(dae, y0):
    """A state away from the initial one, and the Jacobian there."""
    t = 0.3
    y = y0 * 1.1 + 0.05
    return t, y, dae.J(t, y, dae.p)


def _keys(A):
    """``col * n + row`` of the stored entries of the CSC matrix ``A``, in storage order."""
    n = A.shape[0]
    indptr = np.asarray(A.indptr, dtype=np.int64)
    col = np.repeat(np.arange(A.shape[1], dtype=np.int64), np.diff(indptr))
    return col * n + np.asarray(A.indices[:indptr[-1]], dtype=np.int64)


def _check(Wm, rscale, M, J, dt, gamma):
    """``Wm`` and ``rscale`` against the chain; returns the chain's matrix."""
    with np.errstate(all='ignore'):
        ref, ref_rscale = legacy_iteration_matrix(M, J, dt, gamma, SimpleNamespace(rscale_to_dense=None))
    assert _byte_equal(rscale, ref_rscale)
    assert type(Wm) is csc_array and Wm.shape == ref.shape and Wm.has_canonical_format
    key = _keys(Wm)
    assert np.all(np.diff(key) > 0), 'the rows of a column are not sorted and unique'
    assert np.array_equal(key, np.union1d(_keys(csc_array(M)), _keys(csc_array(J))))
    ref_key = _keys(ref)
    pos = np.searchsorted(key, ref_key)
    assert np.all(pos < key.size) and np.array_equal(key[pos], ref_key)
    assert _byte_equal(Wm.data[pos], ref.data)
    dropped = np.ones(key.size, dtype=bool)
    dropped[pos] = False
    assert np.all(Wm.data[dropped] == 0.0)
    return ref


def _snapshot(*arrays):
    return [a.copy() for a in arrays]


def _unchanged(before, *arrays):
    return all(_byte_equal(b, a) for b, a in zip(before, arrays))


# -- (a) the values of the chain ---------------------------------------------------

SPARSE = [('dae_test', 'inline_sparse', ()),
          ('dae_test', 'rendered', ()),
          ('forced', 'inline_sparse', ()),
          ('trace', 'rendered', ()),
          ('permuted', 'inline_sparse', ()),
          ('vdp', 'inline_sparse', ()),
          ('orbit', 'inline_sparse', ()),
          ('alloc', 'rendered', (6,)),
          ('ladder', 'inline_sparse', (12,)),
          ('ladder', 'rendered', (12,))]


@pytest.mark.filterwarnings('ignore:Mutable Jacobian block:UserWarning')
@pytest.mark.parametrize('dt', [1e-3, 0.37, np.int64(2)], ids=['1e-3', '0.37', 'int'])
@pytest.mark.parametrize('name, variant, args', SPARSE, ids=[f"{a[0]}-{a[1]}" for a in SPARSE])
def test_the_values_of_the_chain_on_the_models(model, name, variant, args, dt):
    dae, y0 = model(name, variant, *args)
    t, y, J = _point(dae, y0)
    before = _snapshot(J.data, J.indices, J.indptr, dae.M.data, dae.M.indices, dae.M.indptr)
    im = _matrix(y0.size)
    for gamma in GAMMAS:
        Wm, rscale, dtgamma = im.build(dae.M, J, dt, gamma)
        assert dtgamma == dt * gamma and type(dtgamma) is type(dt * gamma)
        _check(Wm, rscale, dae.M, J, dt, gamma)
        assert _unchanged(before, J.data, J.indices, J.indptr, dae.M.data, dae.M.indices, dae.M.indptr)
    assert im.rscale_to_dense is None, 'the chain ran'


def _discriminating(rng, dtgamma, m=None):
    """Two values ``a, b`` of duplicates of ``J`` whose scaled sum differs in
    its last bits from the sum scaled, or, with the value ``m`` of ``M`` at
    their position, for which ``(m - dtgamma a) - dtgamma b`` differs from
    ``m - (dtgamma a + dtgamma b)``, so that an assembly that scales after
    summing, or subtracts the duplicates one by one, is detected."""
    for _ in range(10000):
        a, b = rng.uniform(-3.0, 3.0, 2)
        x, y = dtgamma * a, dtgamma * b
        if (x + y != dtgamma * (a + b)) if m is None else ((m - x) - y != m - (x + y)):
            return a, b
    raise AssertionError('no discriminating pair found')


def _order_sensitive(rng, scale):
    """Three values whose scaled sum in storage order, ``(s a + s b) + s c``,
    differs from the sum in reverse order, so that an assembly that sums
    duplicates in another order than SciPy is detected."""
    for _ in range(10000):
        a, b, c = rng.uniform(-3.0, 3.0, 3)
        x, y, z = scale * a, scale * b, scale * c
        if (x + y) + z != (z + y) + x:
            return a, b, c
    raise AssertionError('no order-sensitive triple found')


def _hand_built(dtgamma):
    """``(M, J)`` of order 4: ``M`` singular with a stored zero, ``J`` with
    unsorted rows, duplicates where ``M`` is absent, zero and one, and a
    stored zero, its arrays shared with nothing else."""
    rng = np.random.default_rng(7)
    a, b = _discriminating(rng, dtgamma)
    c, d = _discriminating(rng, dtgamma)
    e, f = _discriminating(rng, dtgamma, m=1.0)
    M = csc_array((np.array([1.0, 0.0, 1.0]), np.array([0, 1, 3], dtype=np.int32),
                   np.array([0, 1, 2, 2, 3], dtype=np.int32)), shape=(4, 4))
    # column 0: rows 2, 0, 2, 1; column 1: rows 1, 1, 3; column 2: row 0, a
    # stored zero; column 3: rows 3, 2, 3
    indptr = np.array([0, 4, 7, 8, 11], dtype=np.int32)
    indices = np.array([2, 0, 2, 1, 1, 1, 3, 0, 3, 2, 3], dtype=np.int32)
    data = np.array([a, 0.7, b, -0.4, c, d, 1.3, 0.0, e, 0.9, f])
    J = csc_array((data, indices, indptr), shape=(4, 4))
    assert not J.has_canonical_format
    return M, J


def _three_duplicates(dtgamma):
    """``(M, J)`` of order 3: three duplicates of ``J`` at ``(0, 0)``, where
    ``M`` stores nothing, whose scaled sum depends on its order."""
    a, b, c = _order_sensitive(np.random.default_rng(3), dtgamma)
    M = csc_array((np.array([1.0, 1.0]), np.array([1, 2], dtype=np.int32),
                   np.array([0, 0, 1, 2], dtype=np.int32)), shape=(3, 3))
    J = csc_array((np.array([a, b, c, 0.7, -0.4, 0.3, 1.1]), np.array([0, 0, 0, 1, 2, 1, 2], dtype=np.int32),
                   np.array([0, 3, 5, 7], dtype=np.int32)), shape=(3, 3))
    return M, J


def _duplicates_apart(dtgamma):
    """``(M, J)`` of order 3: the only duplicates of ``J``, rows 0, 1, 0 of
    column 0, are not adjacent in storage, at the position where ``M`` is 1,
    with values for which subtracting them one by one differs from
    subtracting their sum."""
    a, b = _discriminating(np.random.default_rng(5), dtgamma, m=1.0)
    M = csc_array(np.diag([1.0, 1.0, 1.0]))
    J = csc_array((np.array([a, 0.6, b, 0.8, -0.2, 1.4]), np.array([0, 1, 0, 1, 2, 2], dtype=np.int32),
                   np.array([0, 3, 5, 6], dtype=np.int32)), shape=(3, 3))
    return M, J


def _duplicates_in_M(dtgamma):
    """``(M, J)`` of order 3: three duplicates of ``M`` at ``(0, 0)``, whose
    sum depends on its order, and an entry of ``J`` there; ``J`` has no
    duplicates."""
    a, b, c = _order_sensitive(np.random.default_rng(9), 1.0)
    M = csc_array((np.array([a, b, c, 1.0, 1.0]), np.array([0, 0, 0, 1, 2], dtype=np.int32),
                   np.array([0, 3, 4, 5], dtype=np.int32)), shape=(3, 3))
    J = csc_array(np.array([[2.0, 1.0, 0.0], [1.0, -3.0, 0.5], [0.0, 0.5, 1.0]]))
    return M, J


NON_CANONICAL = [_hand_built, _three_duplicates, _duplicates_apart, _duplicates_in_M]


@pytest.mark.parametrize('make', NON_CANONICAL, ids=[f.__name__.strip('_') for f in NON_CANONICAL])
def test_duplicates_in_M_and_J_are_summed_as_by_the_chain(make):
    """Every assembly is made on one ``IterationMatrix``, for two values of
    ``dt`` and of ``gamma``: the pattern, and with it the scratch for the
    sums of the duplicates of ``J``, is shared by all of them, as within a
    run, and each must give the chain's values."""
    im = pat = None
    for dt in (1e-3, 0.37):
        for gamma in GAMMAS:
            M, J = make(dt * gamma)
            if im is None:
                im = _matrix(M.shape[0])
            before = _snapshot(M.data, M.indices, M.indptr, J.data, J.indices, J.indptr)
            Wm, rscale, _ = im.build(M, J, dt, gamma)
            if pat is None:
                pat = im._pattern
            assert im._pattern is pat
            j_key = _keys(J)
            assert pat.j_dup == (np.unique(j_key).size < j_key.size)
            _check(Wm, rscale, M, J, dt, gamma)
            assert _unchanged(before, M.data, M.indices, M.indptr, J.data, J.indices, J.indptr)
    assert im._pattern.j_dup == (make is not _duplicates_in_M)


def test_exact_cancellation_and_a_zero_row():
    """``M - (dt*gamma) J`` cancels exactly at ``(0, 0)``, and row 1 cancels
    entirely: the chain drops both and gives row 1 the scale ``inf``, the
    assembly keeps them as explicit zeros, not as ``0 * inf``."""
    M = csc_array(np.diag([1.0, 1.0, 1.0]))
    J = csc_array((np.array([4.0, 0.0, 2.0, 4.0, 3.0]), np.array([0, 1, 0, 1, 2], dtype=np.int32),
                   np.array([0, 2, 4, 5], dtype=np.int32)), shape=(3, 3))
    im = _matrix(3)
    Wm, rscale, _ = im.build(M, J, 1.0, 0.25)
    ref = _check(Wm, rscale, M, J, 1.0, 0.25)
    assert rscale[1] == np.inf
    assert ref.nnz == 2 and Wm.nnz == 5
    dense = Wm.toarray()
    assert dense[0, 0] == 0.0 and np.all(dense[1] == 0.0) and np.isfinite(dense).all()


def test_non_finite_entries_as_by_the_chain():
    """An infinite entry gives its row the scale 0, which the chain drops
    with the whole row, and a NaN propagates through the row maximum to the
    whole row."""
    M = csc_array(np.diag([1.0, 1.0, 1.0, 1.0]))
    J = csc_array(np.array([[0.5, np.inf, 0.0, 0.0],
                            [0.0, 0.5, 0.0, 1.0],
                            [0.2, 0.0, np.nan, 0.3],
                            [0.0, 0.0, 0.0, 2.0]]))
    im = _matrix(4)
    Wm, rscale, _ = im.build(M, J, 0.1, GAMMAS[0])
    _check(Wm, rscale, M, J, 0.1, GAMMAS[0])
    assert rscale[0] == 0.0 and np.isnan(rscale[2])
    dense = Wm.toarray()
    assert np.all(dense[0] == 0.0) and np.isnan(dense[2, [0, 2, 3]]).all()


def test_another_format_or_a_dense_matrix_takes_the_chain():
    M = csc_array(np.diag([1.0, 0.0, 1.0]))
    Jd = np.array([[2.0, 1.0, 0.0], [1.0, -3.0, 0.5], [0.0, 0.5, 1.0]])
    for J in (csr_array(Jd), Jd):
        im = _matrix(3)
        Wm, rscale, _ = im.build(M, J, 0.1, GAMMAS[0])
        ref, ref_rscale = legacy_iteration_matrix(M, J, 0.1, GAMMAS[0], SimpleNamespace(rscale_to_dense=None))
        assert im._pattern is None and im.rscale_to_dense is not None
        assert type(Wm) is type(ref) and _byte_equal(rscale, ref_rscale)
        if isinstance(ref, np.ndarray):
            assert _byte_equal(Wm, ref)
        else:
            assert _byte_equal(Wm.data, ref.data) and _byte_equal(Wm.indices, ref.indices)


# -- (b) a pattern that changes, values that change ---------------------------------


def _counting(monkeypatch):
    calls = []
    equal = linalg._equal_pattern

    def counted(A, indptr, indices):
        calls.append(A)
        return equal(A, indptr, indices)

    monkeypatch.setattr(linalg, '_equal_pattern', counted)
    return calls


def test_a_rendered_J_is_recognized_by_memory(model, monkeypatch):
    dae, y0 = model('ladder', 'rendered', 12)
    t, y, J1 = _point(dae, y0)
    J2 = dae.J(t + 0.1, y * 0.9, dae.p)
    # SciPy hands out a new view of the shared indices with every Jacobian
    assert J2.indices is not J1.indices and np.shares_memory(J2.indices, J1.indices)
    calls = _counting(monkeypatch)
    im = _matrix(y0.size)
    Wm1, _, _ = im.build(dae.M, J1, 0.1, GAMMAS[0])
    pat = im._pattern
    for J in (J2, J1, J2):
        Wm, rscale, _ = im.build(dae.M, J, 0.1, GAMMAS[1])
        _check(Wm, rscale, dae.M, J, 0.1, GAMMAS[1])
        assert im._pattern is pat and Wm.indptr is pat.indptr
    assert calls == []


def test_an_inline_J_is_compared_by_value(model, monkeypatch):
    dae, y0 = model('ladder', 'inline_sparse', 12)
    t, y, J1 = _point(dae, y0)
    J2 = dae.J(t + 0.1, y * 0.9, dae.p)
    assert not np.shares_memory(J2.indices, J1.indices)
    calls = _counting(monkeypatch)
    im = _matrix(y0.size)
    im.build(dae.M, J1, 0.1, GAMMAS[0])
    pat = im._pattern
    Wm, rscale, _ = im.build(dae.M, J2, 0.1, GAMMAS[0])
    _check(Wm, rscale, dae.M, J2, 0.1, GAMMAS[0])
    assert im._pattern is pat and len(calls) == 1
    # J2 is now recognized by its memory
    im.build(dae.M, J2, 0.1, GAMMAS[1])
    assert im._pattern is pat and len(calls) == 1


def test_a_J_of_another_pattern_rebuilds_the_union(monkeypatch):
    M = csc_array(np.diag([1.0, 0.0, 1.0]))
    A = np.array([[2.0, 1.0, 0.0], [1.0, -3.0, 0.5], [0.0, 0.5, 1.0]])
    J1 = csc_array(A)
    B = A.copy()
    B[2, 0] = 0.25
    J2 = csc_array(B)
    J3 = csc_array(A)
    # the indptr of J3 with other rows in column 0, the case of #184
    C = A.copy()
    C[1, 0], C[2, 0] = 0.0, 1.0
    J4 = csc_array(C)
    assert _byte_equal(J4.indptr, J3.indptr) and not _byte_equal(J4.indices, J3.indices)
    # the indices of J5 with another indptr, in new arrays
    J5 = csc_array((np.array([2.0, 1.0, 0.5, 1.5]), np.array([0, 1, 2, 2], dtype=np.int32),
                    np.array([0, 2, 3, 4], dtype=np.int32)), shape=(3, 3))
    J6 = csc_array((np.array([2.0, -3.0, 0.5, 1.5]), np.array([0, 1, 2, 2], dtype=np.int32),
                    np.array([0, 1, 3, 4], dtype=np.int32)), shape=(3, 3))
    assert _byte_equal(J6.indices, J5.indices) and not _byte_equal(J6.indptr, J5.indptr)
    calls = _counting(monkeypatch)
    im = _matrix(3)
    patterns = []
    for J in (J1, J1, J2, J3, J3, J4, J5, J6):
        Wm, rscale, _ = im.build(M, J, 0.37, GAMMAS[0])
        _check(Wm, rscale, M, J, 0.37, GAMMAS[0])
        patterns.append(im._pattern)
    p1, p1b, p2, p3, p3b, p4, p5, p6 = patterns
    assert p1b is p1 and p2 is not p1 and p3 is not p2 and p3b is p3 and p4 is not p3
    assert p5 is not p4 and p6 is not p5
    assert p2.indices.size == p1.indices.size + 1 and _byte_equal(p3.indices, p1.indices)
    # J1 to J1 and J3 to J3 by memory; J2, J3, J4, J5 and J6 compared once each
    assert len(calls) == 5


def test_a_view_of_another_part_of_one_buffer_is_another_pattern(monkeypatch):
    """The indices of ``J2`` are a view of the same buffer as those of
    ``J1``, of the same size, and the two share one ``indptr``: only the
    whole of a buffer is recognized by memory, so ``J2`` is compared by
    value and rebuilds the union."""
    M = csc_array(np.diag([1.0, 0.0, 1.0]))
    buf = np.array([0, 1, 2, 2, 1, 2, 0, 2], dtype=np.int32)
    indptr = np.array([0, 2, 3, 4], dtype=np.int32)
    J1 = csc_array((np.array([2.0, 1.0, 0.5, 1.5]), buf[:4], indptr), shape=(3, 3))
    J2 = csc_array((np.array([1.0, -3.0, 0.5, 1.5]), buf[4:], indptr), shape=(3, 3))
    assert J1.indptr is J2.indptr and J1.indices.base is buf and J2.indices.base is buf
    calls = _counting(monkeypatch)
    im = _matrix(3)
    Wm, rscale, _ = im.build(M, J1, 0.37, GAMMAS[0])
    _check(Wm, rscale, M, J1, 0.37, GAMMAS[0])
    p1 = im._pattern
    Wm, rscale, _ = im.build(M, J2, 0.37, GAMMAS[0])
    _check(Wm, rscale, M, J2, 0.37, GAMMAS[0])
    assert im._pattern is not p1 and len(calls) == 1 and calls[0] is J2


def test_the_values_of_M_are_read_at_every_assembly():
    dae, y0 = models.build('dae_test')
    t, y, J = _point(dae, y0)
    im = _matrix(y0.size)
    im.build(dae.M, J, 0.1, GAMMAS[0])
    pat = im._pattern
    for value in (2.5, 0.0, 1.0):
        # in place, as the modification protocol allows; 0 keeps the pattern
        dae.M.data[0] = value
        Wm, rscale, _ = im.build(dae.M, J, 0.1, GAMMAS[0])
        _check(Wm, rscale, dae.M, J, 0.1, GAMMAS[0])
        assert im._pattern is pat


def test_rows_of_M_changed_in_place_are_seen_after_reset():
    """After ``reset`` the pattern of ``M`` is compared by value with copies
    taken when the union was built, so a change of its index arrays in
    place, which leaves their memory where it was, rebuilds the union."""
    M = csc_array(np.diag([1.0, 0.0, 1.0]))
    J = csc_array(np.array([[2.0, 1.0, 0.0], [1.0, -3.0, 0.5], [0.0, 0.5, 1.0]]))
    im = _matrix(3)
    Wm, rscale, _ = im.build(M, J, 0.37, GAMMAS[0])
    _check(Wm, rscale, M, J, 0.37, GAMMAS[0])
    pat = im._pattern
    # the entry of column 2 moves from row 2 to row 0, where J stores nothing
    M.indices[1] = 0
    im.reset()
    Wm, rscale, _ = im.build(M, J, 0.37, GAMMAS[0])
    _check(Wm, rscale, M, J, 0.37, GAMMAS[0])
    assert im._pattern is not pat


class _Recorder:
    """Records every assembly of ``IterationMatrix.build`` with copies of its inputs."""

    def __init__(self, monkeypatch):
        self.records = []
        build = IterationMatrix.build
        records = self.records

        def recorded(im, M, J, dt, gamma):
            Wm, rscale, dtgamma = build(im, M, J, dt, gamma)
            records.append((M.copy(), J.copy(), dt, gamma, Wm.copy(), rscale.copy(), im._pattern))
            return Wm, rscale, dtgamma

        monkeypatch.setattr(IterationMatrix, 'build', recorded)

    def check(self, start=0):
        for M, J, dt, gamma, Wm, rscale, _ in self.records[start:]:
            _check(Wm, rscale, M, J, dt, gamma)


def test_M_changed_through_model_modified(monkeypatch):
    rec = _Recorder(monkeypatch)
    dae, y0 = models.build('dae_test')
    integ = init(dae, [0, 2], y0, opt=Opt(rtol=1e-6, atol=1e-8))
    for _ in range(3):
        integ.step()
    pat = integ.linalg._pattern
    assert pat is not None and len(rec.records) > 0
    k = len(rec.records)

    # a value of M, in place; the pattern is compared by value once and kept
    dae.M.data[0] = 2.0
    integ.model_modified()
    assert pat.m.indptr is None and pat.j.indptr is None
    for _ in range(3):
        integ.step()
    assert not integ.failed and integ.linalg._pattern is pat
    assert all(r[0].data[0] == 2.0 for r in rec.records[k:])
    k2 = len(rec.records)

    # M rebound to another pattern, an explicit zero on the algebraic row
    dae.M = csc_array((np.array([2.0, 0.0]), np.array([0, 1], dtype=np.int32),
                       np.array([0, 1, 2], dtype=np.int32)), shape=(2, 2))
    integ.model_modified()
    for _ in range(3):
        integ.step()
    assert not integ.failed
    new = integ.linalg._pattern
    assert new is not pat and new.m_slot.size == 2
    assert all(r[0].nnz == 2 and r[6] is new for r in rec.records[k2:])
    rec.check()


# -- (c) factorizations alive together ----------------------------------------------


@pytest.mark.parametrize('name, variant, args', [('dae_test', 'inline_sparse', ()),
                                                 ('ladder', 'rendered', (12,))],
                         ids=['dae_test', 'ladder-rendered'])
def test_two_factorizations_of_one_attempt_are_independent(model, backend, name, variant, args):
    dae, y0 = model(name, variant, *args)
    n = y0.size
    t, y, J = _point(dae, y0)
    im = _matrix(n)
    Wm1, rscale1, _ = im.build(dae.M, J, 0.1, GAMMAS[0])
    data1, scale1 = _snapshot(Wm1.data, rscale1)
    Wm2, rscale2, _ = im.build(dae.M, J, 0.1, GAMMAS[1])
    # the pattern is shared, the values and the scale are not
    assert Wm2.indices.base is Wm1.indices.base and Wm2.indptr is Wm1.indptr
    assert not np.shares_memory(Wm1.data, Wm2.data) and not np.shares_memory(rscale1, rscale2)
    assert _unchanged([data1, scale1], Wm1.data, rscale1)

    b = dae.F(t, y, dae.p)
    W1 = im.factorize(dae.M, J, 0.1, GAMMAS[0])
    x1 = W1.solve(b)
    W2 = im.factorize(dae.M, J, 0.1, GAMMAS[1])
    x2 = W2.solve(b)
    for _ in range(2):
        assert _byte_equal(W1.solve(b), x1) and _byte_equal(W2.solve(b), x2)
    # the first equals a factorization that never had a second one beside it
    assert _byte_equal(_matrix(n).factorize(dae.M, J, 0.1, GAMMAS[0]).solve(b), x1)


class _TwoGammas(Algorithm):
    """Backward Euler over the step and over two half steps, the second
    ``W`` built while the first is alive; each solve with ``W(1)`` is repeated
    after ``W(1/2)`` exists and must give the same bits."""

    scheme = 'two_gammas'
    order = 1
    error_order = 2
    adaptive = True

    def __init__(self):
        self.checks = []

    def perform_step(self, s):
        W1 = s.W(1.0)
        rhs = s.M @ s.y0
        x = W1.solve(rhs)
        y_full = s.implicit(s.t + s.h, 1.0, rhs)
        y_half = s.implicit(s.t + 0.5 * s.h, 0.5, rhs)
        y = s.implicit(s.t + s.h, 0.5, s.M @ y_half, y=y_half)
        self.checks.append(_byte_equal(W1.solve(rhs), x) and s.W(1.0) is W1)
        return y, y - y_full


@pytest.mark.parametrize('name, variant, args', [('dae_test', 'inline_sparse', ()),
                                                 ('ladder', 'rendered', (12,))],
                         ids=['dae_test', 'ladder-rendered'])
def test_a_formula_method_with_two_gammas(model, backend, name, variant, args):
    dae, y0 = model(name, variant, *args)
    alg = _TwoGammas()
    integ = init(dae, [0, 1], y0, alg=alg, opt=Opt(rtol=1e-4, atol=1e-6))
    sol = integ.solve()
    assert sol.stats.ret == 'success' and integ.linalg._pattern is not None
    assert len(alg.checks) > 5 and all(alg.checks)


# -- (d) the shared pattern arrays ----------------------------------------------------


def _factorize_repeatedly(im, M, J, b):
    """Factorizations of two values of gamma, twice each, and solves with each."""
    for gamma in GAMMAS + GAMMAS:
        W = im.factorize(M, J, 0.1, gamma)
        W.solve(b)
        W.solve(b, out=np.empty(b.size))


@pytest.mark.parametrize('name, variant, args', [('dae_test', 'inline_sparse', ()),
                                                 ('permuted', 'inline_sparse', ()),
                                                 ('ladder', 'rendered', (12,))],
                         ids=['dae_test', 'permuted', 'ladder-rendered'])
@pytest.mark.parametrize('matching', [False, True], ids=['plain', 'matching'])
def test_factorizations_leave_the_shared_pattern_unchanged(request, model, backend, name, variant, args,
                                                          matching):
    if matching:
        if backend != 'klu':
            pytest.skip('the row matching is a KLU analysis')
        request.getfixturevalue('klu_matching_low')
    dae, y0 = model(name, variant, *args)
    t, y, J = _point(dae, y0)
    im = _matrix(y0.size)
    Wm, _, _ = im.build(dae.M, J, 0.1, GAMMAS[0])
    pat = im._pattern
    # every W hands the backend the shared arrays themselves
    assert Wm.indptr is pat.indptr and Wm.indices.base is pat.indices
    assert not pat.indices.flags.writeable and not pat.indptr.flags.writeable
    before = _snapshot(pat.indices, pat.indptr)
    b = dae.F(t, y, dae.p)
    _factorize_repeatedly(im, dae.M, J, b)
    if backend == 'klu':
        assert (im.cache.symbolic.perm is not None) == matching
    else:
        assert im.cache.superlu is not None
    # the SuperLU ordering and a KLU analysis with a matching are computed again
    im.reset()
    _factorize_repeatedly(im, dae.M, J, b)
    assert _unchanged(before, pat.indices, pat.indptr)
    assert im._pattern is pat and im.stats.ndecomp == 2 * len(GAMMAS + GAMMAS)


def test_the_shared_pattern_cannot_be_sorted_or_written_in_place(model):
    dae, y0 = model('ladder', 'inline_sparse', 12)
    t, y, J = _point(dae, y0)
    Wm, _, _ = _matrix(y0.size).build(dae.M, J, 0.1, GAMMAS[0])
    before = _snapshot(Wm.indices, Wm.indptr)
    # flagged canonical, so the in-place canonicalizations of SciPy return at once
    Wm.sum_duplicates()
    Wm.sort_indices()
    assert _unchanged(before, Wm.indices, Wm.indptr)
    with pytest.raises(ValueError, match='read-only'):
        Wm.indices[0] = 1
    with pytest.raises(ValueError, match='read-only'):
        Wm.indptr[1] = 0


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
def test_a_value_that_cancels_keeps_the_pattern_and_the_analysis():
    """The chain drops an entry once it cancels, which changes the pattern of
    ``W`` and costs a new KLU analysis; the union pattern keeps it."""
    M = csc_array(np.diag([1.0, 1.0, 1.0]))
    J = csc_array(np.array([[4.0, 1.0, 0.0], [1.0, -3.0, 0.5], [0.0, 0.5, 2.0]]))
    with linsolver('klu'):
        im = _matrix(3)
        legacy = _matrix(3, legacy_compat=True)
        analyses, legacy_analyses = [], []
        # dt = 1 cancels W[0, 0] exactly
        for dt in (0.1, 1.0, 0.3):
            im.factorize(M, J, dt, 0.25)
            legacy.factorize(M, J, dt, 0.25)
            analyses.append(im.cache.symbolic)
            legacy_analyses.append(legacy.cache.symbolic)
    assert all(a is analyses[0] for a in analyses) and analyses[0].nnz == 7
    assert len({id(a) for a in legacy_analyses}) == 3
    assert [a.nnz for a in legacy_analyses] == [7, 6, 7]


# -- (e) the integration ----------------------------------------------------------------

RUNS = [('dae_test', 'inline_sparse', ()),
        ('dae_test', 'rendered', ()),
        ('permuted', 'inline_sparse', ()),
        ('forced', 'inline_sparse', ()),
        ('trace', 'rendered', ()),
        ('vdp', 'inline_sparse', ()),
        ('ladder', 'inline_sparse', (12,)),
        ('ladder', 'rendered', (12,))]


@pytest.mark.filterwarnings('ignore:Mutable Jacobian block:UserWarning')
@pytest.mark.parametrize('name, variant, args', RUNS, ids=[f"{a[0]}-{a[1]}" for a in RUNS])
def test_the_default_configuration_integrates_as_the_chain_does(model, backend, name, variant, args):
    dae, y0 = model(name, variant, *args)
    tspan = np.linspace(0, 1, 11)
    rtol = 1e-6
    runs = []
    for fixed in (True, False):
        integ = init(dae, tspan, y0, opt=Opt(rtol=rtol, atol=1e-8))
        assert integ.linalg.fixed_pattern
        integ.linalg.fixed_pattern = fixed
        runs.append(integ.solve())
        assert (integ.linalg._pattern is not None) == fixed
        assert (integ.linalg.rscale_to_dense is None) == fixed
    fast, chain = runs
    assert fast.stats.ret == chain.stats.ret == 'success'
    assert np.array_equal(fast.T, chain.T)
    dsteps = abs(fast.stats.nstep - chain.stats.nstep)
    dY = float(np.max(np.abs(np.asarray(fast.Y) - np.asarray(chain.Y))))
    print(f"{name}-{variant}: accepted steps {fast.stats.nstep} on the union pattern, {chain.stats.nstep} "
          f"by the chain; max|dY| = {dY:.3e}")
    assert dsteps <= MAX_STEP_DIFFERENCE and dY <= MAX_DY_IN_RTOL * rtol


@pytest.mark.parametrize('name, variant, args', [('dae_test', 'rendered', ()), ('ladder', 'inline_sparse', (12,))],
                         ids=['dae_test-rendered', 'ladder'])
def test_the_legacy_compatible_configuration_keeps_the_chain(model, backend, name, variant, args):
    dae, y0 = model(name, variant, *args)
    integ = init(dae, [0, 1], y0, alg=Rodas4(legacy_compat=True), opt=Opt(rtol=1e-6, atol=1e-8))
    assert not integ.linalg.fixed_pattern
    sol = integ.solve()
    assert sol.stats.ret == 'success' and sol.stats.ndecomp > 0
    assert integ.linalg._pattern is None and integ.linalg.rscale_to_dense is not None
