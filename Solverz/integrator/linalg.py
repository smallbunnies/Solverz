"""The iteration matrix ``W = M - (dt*gamma) J`` of an integration: its
assembly, its factorization and its solves."""
import numpy as np
from numba import njit
from scipy.linalg.lapack import dgetrf, dgetrs
from scipy.sparse import csc_array, diags_array

# MATCHING_MIN_N is read as an attribute at the time of use, since
# set_klu_matching rebinds it
import Solverz.solvers.klu_backend as klu_backend
from Solverz.solvers.klu_backend import KLUCache, klu_decomposition, klu_matching_enabled
from Solverz.solvers.laesolver import lu_decomposition, model_cache
from Solverz.integrator.algorithm import StepFailure

__all__ = []


@njit(cache=True, error_model='numpy')
def _assemble_w(m_slot, m_val, j_slot, j_val, dtgamma, indices, j_dup, j_sum, w, rscale):
    """Write the row-scaled ``M - dtgamma J`` into ``w`` and its row scaling
    into ``rscale``, on the union pattern, with the operations of the SciPy
    chain of ``IterationMatrix.build`` in their order.

    ``m_slot[k]`` and ``j_slot[k]`` are the positions in ``w`` of the
    ``k``-th stored entry of ``M`` and of ``J``, and ``indices[p]`` is the row
    of position ``p``. ``j_sum`` is scratch of the size of ``w`` when ``j_dup``
    is true. NumPy's error model makes ``1 / 0`` give ``inf``, as the chain
    does, where the Python model would raise ``ZeroDivisionError``.
    """
    nnz = w.shape[0]
    n = rscale.shape[0]
    # SciPy's subtraction sums the duplicates of each operand from zero, in
    # storage order, and subtracts the two sums, so the scaled duplicates of J
    # are summed apart before the subtraction; without duplicates the sum is
    # the one scaled entry, which is subtracted directly
    for p in range(nnz):
        w[p] = 0.0
    for k in range(m_slot.shape[0]):
        w[m_slot[k]] += m_val[k]
    if j_dup:
        for p in range(nnz):
            j_sum[p] = 0.0
        for k in range(j_slot.shape[0]):
            j_sum[j_slot[k]] += dtgamma * j_val[k]
        for p in range(nnz):
            w[p] = w[p] - j_sum[p]
    else:
        for k in range(j_slot.shape[0]):
            p = j_slot[k]
            w[p] = w[p] - dtgamma * j_val[k]
    # the row maxima of |M - dtgamma J|, into rscale; the maximum is exact in
    # any order, and a NaN is kept once met, as np.maximum keeps it
    for r in range(n):
        rscale[r] = 0.0
    for p in range(nnz):
        a = abs(w[p])
        r = indices[p]
        if a > rscale[r] or a != a:
            rscale[r] = a
    for r in range(n):
        rscale[r] = 1.0 / rscale[r]
    # SciPy drops an exact zero of M - dtgamma J and a zero scale, that of a
    # row whose maximum is inf, before the product, so neither turns into the
    # NaN of 0 * inf; such an entry is an explicit zero here
    for p in range(nnz):
        v = w[p]
        s = rscale[indices[p]]
        if v != 0.0 and s != 0.0:
            w[p] = v * s
        else:
            w[p] = 0.0


def _same_memory(a, ref):
    """Whether the array ``a`` is ``ref``, or a contiguous view of exactly the
    memory of ``ref``.

    SciPy builds every ``csc_array`` with ``indices[:nnz]``, a new view of the
    array it is given, so the ``indices`` of two Jacobians of a rendered
    model are two views of one array, not one object. A view whose bytes
    cover the whole of the array that owns the memory starts at its first
    byte. ``ref`` is kept alive by the caller, so the owner cannot be freed
    and its memory handed to another array.
    """
    if a is ref:
        return True
    if ref is None:
        return False
    owner = a if a.base is None else a.base
    ref_owner = ref if ref.base is None else ref.base
    return (owner is ref_owner and type(owner) is np.ndarray and owner.base is None
            and a.dtype == ref.dtype and a.shape == ref.shape
            and a.flags.c_contiguous and ref.flags.c_contiguous and a.nbytes == owner.nbytes)


def _equal_pattern(A, indptr, indices):
    """Whether the CSC matrix ``A`` has the pattern ``(indptr, indices)``, by value."""
    return (A.indptr.shape == indptr.shape and A.indices.shape == indices.shape
            and np.array_equal(A.indptr, indptr) and np.array_equal(A.indices, indices))


class _InputPattern:
    """The pattern of ``M`` or ``J`` for which the slots of a ``_UnionPattern``
    were computed.

    ``indptr`` and ``indices`` are the caller's arrays, kept to recognize the
    next matrix by its memory, which costs no pass over the pattern. A matrix
    with other arrays is compared by value with the private copies, since
    each Jacobian of an inline model, for one, comes with new arrays of the
    same pattern; it then becomes the matrix recognized by memory.
    """

    __slots__ = ('indptr', 'indices', 'indptr_copy', 'indices_copy')

    def __init__(self, A):
        self.indptr, self.indices = A.indptr, A.indices
        self.indptr_copy, self.indices_copy = A.indptr.copy(), A.indices.copy()

    def holds(self, A):
        if _same_memory(A.indptr, self.indptr) and _same_memory(A.indices, self.indices):
            return True
        if _equal_pattern(A, self.indptr_copy, self.indices_copy):
            self.indptr, self.indices = A.indptr, A.indices
            return True
        return False

    def forget(self):
        """Compare the next matrix by value, after its arrays may have been changed in place."""
        self.indptr = self.indices = None


def _keys(A, n):
    """``col * n + row`` of every stored entry of the CSC matrix ``A``, in storage order."""
    indptr = np.asarray(A.indptr, dtype=np.int64)
    col = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    return col * n + np.asarray(A.indices[:indptr[-1]], dtype=np.int64)


class _UnionPattern:
    """The canonical CSC pattern of ``M + J``, on which the default
    configuration assembles ``W``, and the position in it of every stored
    entry of ``M`` and of ``J``.

    The pattern holds every entry that ``M`` or ``J`` stores, so an entry
    that SciPy would drop as an exact zero stays as an explicit zero, and the
    pattern of ``W`` is that of the model rather than of its values. Its rows
    are sorted and unique, so ``W`` is canonical by construction. ``indices``
    and ``indptr`` are read-only, since every ``W`` of the pattern holds them.
    """

    __slots__ = ('indptr', 'indices', 'm', 'j', 'm_slot', 'j_slot', 'j_dup', 'j_sum')

    def __init__(self, M, J, n):
        m_key, j_key = _keys(M, n), _keys(J, n)
        key = np.concatenate((m_key, j_key))
        key.sort()
        first = np.ones(key.size, dtype=bool)
        first[1:] = key[1:] != key[:-1]
        key = key[first]
        # int32 when it fits, so that neither csc_array nor KLU converts the
        # index arrays at every factorization
        idx = np.int32 if max(key.size, n) < np.iinfo(np.int32).max else np.int64
        self.indices = (key % n).astype(idx)
        self.indptr = np.zeros(n + 1, dtype=idx)
        self.indptr[1:] = np.cumsum(np.bincount(key // n, minlength=n))
        self.indices.flags.writeable = False
        self.indptr.flags.writeable = False
        self.m, self.j = _InputPattern(M), _InputPattern(J)
        self.m_slot = np.searchsorted(key, m_key)
        self.j_slot = np.searchsorted(key, j_key)
        j_sorted = np.sort(j_key)
        self.j_dup = bool(np.any(j_sorted[1:] == j_sorted[:-1]))
        self.j_sum = np.empty(key.size if self.j_dup else 0)


def _csc_float64(A, n):
    """Whether ``A`` is a SciPy CSC matrix of float64 entries and shape ``(n, n)``."""
    return getattr(A, 'format', None) == 'csc' and A.shape == (n, n) and A.data.dtype == np.float64


class IterationMatrix:
    """Assembles and factorizes ``W = M - (dt*gamma) J`` for one integration.

    In the legacy-compatible configuration, and in the default one for a
    dense ``M`` or ``J`` or a sparse one in another format than CSC, the
    assembly is the operation chain of legacy Rodas, so the matrix handed
    to the factorization is the legacy one bit for bit, including three
    effects that depend on the values. SciPy drops exact zeros in the
    subtraction and in the row scaling, so the pattern of ``W`` can change
    from step to step. The row scaling emits the row indices of each column
    in descending order, which KLU receives unsorted and compares verbatim
    when it decides whether to reuse its analysis; KLU's factors depend on
    that order, so no assembly that sorts them is used in this
    configuration. A row whose largest entry is zero gets an infinite scale,
    and the factorization or the solve then fails or returns non-finite
    values.

    In the default configuration a sparse ``W`` with float64 CSC ``M`` and
    ``J`` is assembled by ``_assemble_w`` on the union of the patterns of
    ``M`` and ``J`` (``fixed_pattern``). The chain spends 6 to 40 us in each
    SciPy operation, 146 to 570 us per assembly on the benchmark models of 28
    to 10 000 unknowns, against 6.5 to 46 us here, of which 5 us build the
    ``csc_array``. The kernel performs the operations of the chain in their
    order, so every entry that the chain stores has the same bits and every
    entry that it drops is an explicit zero; ``rscale`` is the same, ``inf``
    for a zero row included. The pattern of ``W`` is then fixed, and the
    factorization receives sorted rows and the explicit zeros, so its last
    bits differ from legacy's under KLU, whose factors depend on the order of
    the rows within a column; the KLU analysis and the SuperLU ordering are
    computed once per pattern instead of whenever a value becomes exactly
    zero.

    The pattern is checked at every assembly. ``M`` and ``J`` with the arrays
    of the previous assembly, as every Jacobian of a rendered model has, are
    recognized by memory; other arrays are compared by value, and a different
    pattern, as a user's ``J`` may return, rebuilds the union. ``M.data`` is
    read at every assembly, so a value changed by the modification protocol
    enters the next ``W``; ``reset`` makes the next assembly compare both
    patterns by value, since the protocol may also have changed the arrays of
    ``M`` in place.

    Buffers. Every assembly writes a new data array and a new ``rscale``. A
    factorization keeps its ``rscale`` for its solves, and the KLU object
    holds the data array it factorized; several factorizations are alive
    within one attempt, one per distinct ``gamma``, and the Newton iteration
    of ``implicit`` keeps one across its iterations, so an array shared
    between assemblies would be overwritten under a live factorization, the
    defect of #187. The two allocations cost 0.2 us. The read-only ``indices``
    and ``indptr`` of the pattern are shared by every ``W`` of the call: ``W``
    is flagged canonical, which it is, so ``sum_duplicates`` and
    ``sort_indices``, which ``sp_decomposition`` and ``splu`` call and which
    would sort in place, return at once; KLU and SuperLU only read them; and
    any write raises instead of changing the pattern of another ``W``.

    ``J`` is never modified: either assembly only reads it and hands only the
    scaled matrix to a backend. The Jacobians of a rendered model share one
    ``indices`` and one ``indptr`` array.

    ``backend`` is the backend resolved once for the call. ``cache`` is the
    call's ``KLUCache``; it carries the KLU analysis, or the SuperLU column
    ordering, from one factorization of the call to the next of the same
    pattern. In the legacy-compatible configuration it starts empty, as in
    legacy Rodas. In the default configuration it starts with the KLU
    analysis that ``model_cache(dae)`` holds, if that analysis contains no
    row matching and the current setting would compute none for this size,
    and every analysis without a matching that a factorization makes is
    stored back there. Such an analysis depends only on the pattern, so it
    gives the bits of a fresh one. An analysis with a matching depends on
    the values of the matrix that triggered it, and the SuperLU ordering
    can break an exact pivot tie on another row than COLAMD does, so
    neither leaves the call: a call's result never depends on the calls
    made before it.

    A failed factorization or solve raises ``StepFailure``, which rejects the
    attempt. ``ndecomp`` counts every factorization that succeeds and
    ``nsolve`` every solve that succeeds. A dense ``W`` is factorized once by
    LAPACK in the default configuration, and goes to the legacy
    ``dense_decomposition`` in the legacy-compatible one, whose ``solve``
    runs ``np.linalg.solve`` and so repeats the factorization at every
    solve.
    """

    __slots__ = ('backend', 'cache', 'shared', 'legacy', 'fixed_pattern', 'n', 'stats', 'rscale_to_dense',
                 '_scaled', '_pattern')

    def __init__(self, integ):
        self.backend = integ.opts.linsolver
        self.cache = KLUCache()
        self.legacy = legacy = integ.opts.legacy_compat
        self.fixed_pattern = not legacy
        self.shared = None
        self.n = n = integ.n
        if not legacy:
            self.shared = model_cache(integ.dae)
            sym = self.shared.symbolic
            if (sym is not None and sym.perm is None
                    and not (klu_matching_enabled() and n >= klu_backend.MATCHING_MIN_N)):
                self.cache.symbolic = sym
        self.stats = integ.stats
        # Whether the row maxima of W come back sparse is fixed for a call,
        # so it is decided once, on the first assembly, as legacy decides it.
        self.rscale_to_dense = None
        # The scaled right-hand side, consumed by the backend before a solve
        # returns, so the factorizations of a call share it.
        self._scaled = np.empty(integ.n)
        self._pattern = None

    def reset(self):
        """Return the cache to the state from which a new call starts, after
        the model changed within the call.

        The SuperLU ordering is dropped, and so is a KLU analysis that holds
        a row matching, which depends on the values of the matrix that
        triggered it; an analysis without a matching depends only on the
        pattern and is kept. A run changed at ``t1`` is then byte-equal to
        two calls split at ``t1``, on either backend. The next assembly
        compares the patterns of ``M`` and ``J`` by value.
        """
        cache = self.cache
        cache.superlu = None
        sym = cache.symbolic
        if sym is not None and sym.perm is not None:
            cache.symbolic = None
        if self._pattern is not None:
            self._pattern.m.forget()
            self._pattern.j.forget()

    def build(self, M, J, dt, gamma):
        """``(Wm, rscale, dtgamma)``: the row-scaled matrix ``diag(rscale) (M -
        (dt*gamma) J)``, the row scaling and ``dt * gamma``."""
        dtgamma = dt * gamma
        n = self.n
        if self.fixed_pattern and _csc_float64(M, n) and _csc_float64(J, n):
            pat = self._pattern
            if pat is None or not (pat.m.holds(M) and pat.j.holds(J)):
                pat = self._pattern = _UnionPattern(M, J, n)
            m_val, j_val = M.data, J.data
            # the kernel does not check its bounds
            if m_val.shape[0] < pat.m_slot.shape[0] or j_val.shape[0] < pat.j_slot.shape[0]:
                raise ValueError("the data of M or J holds fewer entries than its pattern")
            w = np.empty(pat.indices.shape[0])
            rscale = np.empty(n)
            _assemble_w(pat.m_slot, m_val, pat.j_slot, j_val, float(dtgamma), pat.indices, pat.j_dup,
                        pat.j_sum, w, rscale)
            Wm = csc_array((w, pat.indices, pat.indptr), shape=(n, n))
            Wm.has_canonical_format = True
            return Wm, rscale, dtgamma
        Miter = M - dtgamma * J
        row_max = np.max(np.abs(Miter), axis=1)
        if self.rscale_to_dense is None:
            self.rscale_to_dense = hasattr(row_max, 'toarray')
        if self.rscale_to_dense:
            row_max = row_max.toarray()
        rscale = (1.0 / np.asarray(row_max)).ravel()
        Wm = diags_array(rscale, format='csc') @ Miter
        return Wm, rscale, dtgamma

    def factorize(self, M, J, dt, gamma):
        """Assemble ``W`` and factorize it with the call's backend and cache."""
        Wm, rscale, dtgamma = self.build(M, J, dt, gamma)
        try:
            if not self.legacy and isinstance(Wm, np.ndarray):
                lu = DenseLU(Wm)
            else:
                lu = lu_decomposition(Wm, backend=self.backend, cache=self.cache)
        except (RuntimeError, np.linalg.LinAlgError) as e:
            raise StepFailure(f"the factorization of W failed: {e}") from e
        shared = self.shared
        if shared is not None and isinstance(lu, klu_decomposition) and lu.symbolic.perm is None:
            shared.symbolic = lu.symbolic
        self.stats.ndecomp += 1
        return Factorization(self, lu, rscale, dtgamma)


class DenseLU:
    """The LU factors of a dense ``W`` from one LAPACK ``getrf``, solved with ``getrs``.

    The legacy dense path runs ``np.linalg.solve`` at every solve and so
    factorizes ``W`` once per stage. A zero pivot, or any other nonzero
    ``info``, raises ``LinAlgError``.
    """

    __slots__ = ('lu', 'piv')

    def __init__(self, A):
        lu, piv, info = dgetrf(np.asarray(A))
        if info != 0:
            raise np.linalg.LinAlgError(f"getrf returned info = {info}")
        self.lu = lu
        self.piv = piv

    def solve(self, b):
        return self.solve_into(b, np.empty(self.lu.shape[0]))

    def solve_into(self, b, out):
        """Solve into ``out``, a C-contiguous float64 vector that does not share memory with ``b``."""
        np.copyto(out, b)
        # a contiguous vector is solved in place, without the copy getrs makes otherwise
        x, info = dgetrs(self.lu, self.piv, out, overwrite_b=True)
        if info != 0:
            raise np.linalg.LinAlgError(f"getrs returned info = {info}")
        if x is not out:
            np.copyto(out, x)
        return out


class Factorization:
    """The factorization of one ``W``: ``rscale``, ``dtgamma`` and ``solve``.

    ``lu`` is the backend's factorization of the row-scaled matrix.
    """

    __slots__ = ('lu', 'rscale', 'dtgamma', '_owner', '_into')

    def __init__(self, owner, lu, rscale, dtgamma):
        self._owner = owner
        self.lu = lu
        self.rscale = rscale
        self.dtgamma = dtgamma
        self._into = lu.solve_into if isinstance(lu, (klu_decomposition, DenseLU)) else None

    def solve(self, b, out=None):
        """Solve ``W x = b``; into ``out`` when it is given, else into a new array.

        ``b`` is scaled by ``rscale`` into the scratch buffer of the
        ``IterationMatrix``, which all its factorizations share and which the
        backend consumes before the solve returns; this gives the bits of
        legacy's ``rscale * rhs`` without an allocation, and the scaled system
        is solved. The result is byte-equal with and without ``out``, and
        ``out`` may be ``b`` itself. KLU and the dense LAPACK factors solve in
        place in ``out``; SuperLU and the legacy dense path return a new
        array, which is copied into ``out``.
        ``out`` must be a writeable C-contiguous float64 vector of length
        ``n``, otherwise ``ValueError``: a strided view, such as a column of a
        C-ordered stage matrix, cannot be handed to ``klu_solve``.
        """
        owner = self._owner
        n = owner.n
        if np.shape(b) != (n,):
            raise ValueError(f"b must be a vector of length {n}, not of shape {np.shape(b)}")
        if out is not None and not (isinstance(out, np.ndarray) and out.dtype == np.float64
                                    and out.shape == (n,) and out.flags.c_contiguous
                                    and out.flags.writeable):
            raise ValueError(f"out must be a writeable C-contiguous float64 vector of length {n}; "
                             f"solve into a buffer of that form and copy a strided target from it")
        scaled = np.multiply(self.rscale, b, out=owner._scaled)
        try:
            if self._into is not None:
                if out is None:
                    out = np.empty(n)
                self._into(scaled, out)
            else:
                x = self.lu.solve(scaled)
                if out is None:
                    out = x
                else:
                    np.copyto(out, x)
        except (RuntimeError, np.linalg.LinAlgError) as e:
            raise StepFailure(f"the solve with W failed: {e}") from e
        owner.stats.nsolve += 1
        return out
