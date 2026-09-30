"""The iteration matrix ``W = M - (dt*gamma) J`` of an integration: its
assembly, its factorization and its solves."""
import numpy as np
from scipy.linalg.lapack import dgetrf, dgetrs
from scipy.sparse import diags_array

# MATCHING_MIN_N is read as an attribute at the time of use, since
# set_klu_matching rebinds it
import Solverz.solvers.klu_backend as klu_backend
from Solverz.solvers.klu_backend import KLUCache, klu_decomposition, klu_matching_enabled
from Solverz.solvers.laesolver import lu_decomposition, model_cache
from Solverz.integrator.algorithm import StepFailure

__all__ = []


class IterationMatrix:
    """Assembles and factorizes ``W = M - (dt*gamma) J`` for one integration.

    The assembly is the operation chain of legacy Rodas, so the matrix handed
    to the factorization is the legacy one bit for bit, including three
    effects that depend on the values. SciPy drops exact zeros in the
    subtraction and in the row scaling, so the pattern of ``W`` can change
    from step to step. The row scaling emits the row indices of each column
    in descending order, which KLU receives unsorted and compares verbatim
    when it decides whether to reuse its analysis; whether KLU's numeric
    factorization depends on that order is not known, so no assembly that
    sorts them is used. A row whose largest entry is zero gets an infinite
    scale, and the factorization or the solve then fails or returns
    non-finite values.

    ``J`` is never modified: the chain builds new matrices and hands only the
    scaled one to a backend, which may sort it in place. The Jacobians of a
    rendered model share one ``indices`` and one ``indptr`` array.

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

    __slots__ = ('backend', 'cache', 'shared', 'legacy', 'n', 'stats', 'rscale_to_dense', '_scaled')

    def __init__(self, integ):
        self.backend = integ.opts.linsolver
        self.cache = KLUCache()
        self.legacy = legacy = integ.opts.legacy_compat
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

    def build(self, M, J, dt, gamma):
        """``(Wm, rscale, dtgamma)``: the row-scaled matrix ``diag(rscale) (M -
        (dt*gamma) J)``, the row scaling and ``dt * gamma``."""
        dtgamma = dt * gamma
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
