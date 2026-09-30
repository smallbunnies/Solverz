"""The iteration matrix ``W = M - (dt*gamma) J`` of an integration: its
assembly, its factorization and its solves."""
import numpy as np
from scipy.sparse import diags_array

from Solverz.solvers.klu_backend import KLUCache, klu_decomposition
from Solverz.solvers.laesolver import lu_decomposition
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
    call's ``KLUCache``, empty when the call starts, as in legacy Rodas; it
    carries the KLU analysis, or the SuperLU column ordering, from one
    factorization of the call to the next of the same pattern.

    A failed factorization or solve raises ``StepFailure``, which rejects the
    attempt. ``ndecomp`` counts every factorization that succeeds and
    ``nsolve`` every solve that succeeds. A dense ``W`` goes to the legacy
    ``dense_decomposition``, whose ``solve`` runs ``np.linalg.solve``, so
    its LAPACK factorization is repeated at every solve.
    """

    __slots__ = ('backend', 'cache', 'n', 'stats', 'rscale_to_dense', '_scaled')

    def __init__(self, integ):
        self.backend = integ.opts.linsolver
        self.cache = KLUCache()
        self.n = integ.n
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
            lu = lu_decomposition(Wm, backend=self.backend, cache=self.cache)
        except RuntimeError as e:
            raise StepFailure(f"the factorization of W failed: {e}") from e
        self.stats.ndecomp += 1
        return Factorization(self, lu, rscale, dtgamma)


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
        self._into = lu.solve_into if isinstance(lu, klu_decomposition) else None

    def solve(self, b, out=None):
        """Solve ``W x = b``; into ``out`` when it is given, else into a new array.

        ``b`` is scaled by ``rscale`` into the scratch buffer of the
        ``IterationMatrix``, which all its factorizations share and which the
        backend consumes before the solve returns; this gives the bits of
        legacy's ``rscale * rhs`` without an allocation, and the scaled system
        is solved. The result is byte-equal with and without ``out``, and
        ``out`` may be ``b`` itself. KLU solves in place in ``out``; SuperLU
        and the dense path return a new array, which is copied into ``out``.
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
