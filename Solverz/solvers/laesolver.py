import os
import contextvars
from typing import Union

import numpy as np
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, linalg as sla

# from scikits import umfpack
# umfpack is slow compared with superlu on Apple M4 with MACOS 15.6.1, scikits-umfpack 0.4.2, and suite-sparse 7.10.3.
# Also, it was found that umfpack was not accurate enough, causing non-convergence issues.
#
# KLU (SuiteSparse) is the default backend when libklu is installed: see
# klu_backend.py. Unlike umfpack it is BTF+AMD-ordered for circuit/network
# matrices and is roughly 2x faster than superlu on the IEGS Jacobians. superlu
# is the fallback (used when libklu is absent, the matrix is complex/dense, or
# the linsolver is set to 'superlu'). KLU is fastest when its symbolic ordering
# is reused across steps (pass a KLUCache via lu_decomposition's cache arg).

from Solverz.solvers.klu_backend import KLU_AVAILABLE, klu_decomposition, KLUCache  # noqa: F401

splu = sla.splu

# --------------------------------------------------------------------------- #
# Global linear-solver selection.
#
# The backend is read from a ContextVar so it can be set once for a whole script
# (set_linsolver), scoped to a block (the ``linsolver`` context manager), or
# overridden per-solve via Opt(linsolver=...). The default is 'klu', overridable
# at import via the SOLVERZ_LINSOLVER environment variable. Resolution always
# degrades to 'superlu' when libklu is unavailable, so the default is safe on any
# machine. The ContextVar (not a plain global) keeps this thread- and async-safe
# and correctly scoped for nested solves.
# --------------------------------------------------------------------------- #
_LINSOLVER = contextvars.ContextVar(
    'solverz_linsolver',
    default=os.environ.get('SOLVERZ_LINSOLVER', 'klu').lower())


def _check_name(name):
    name = str(name).lower()
    if name not in ('klu', 'superlu'):
        raise ValueError(f"linsolver must be 'klu' or 'superlu', got {name!r}")
    return name


def set_linsolver(name):
    """Set the default linear-solver backend for the rest of the script."""
    _LINSOLVER.set(_check_name(name))


def get_linsolver():
    """Return the currently selected backend name ('klu' or 'superlu')."""
    return _LINSOLVER.get()


class linsolver:
    """Context manager scoping the linear-solver backend for a block::

        with linsolver('superlu'):
            sol = Rodas(mdl, tspan, y0)
    """

    def __init__(self, name):
        self.name = _check_name(name)
        self._token = None

    def __enter__(self):
        self._token = _LINSOLVER.set(self.name)
        return self

    def __exit__(self, *exc):
        _LINSOLVER.reset(self._token)
        return False


def resolve_backend(backend=None):
    """Resolve an explicit backend (or None -> the global selection) to the
    effective backend, degrading 'klu' to 'superlu' when libklu is absent."""
    b = _check_name(backend) if backend is not None else _LINSOLVER.get()
    if b == 'klu' and not KLU_AVAILABLE:
        return 'superlu'
    return b


def model_cache(obj):
    """Return a KLUCache attached to ``obj`` (a model/solver object), creating
    it on first use. Lets a solver reuse the KLU symbolic ordering across its
    factorizations (the iteration-matrix pattern is fixed for a model) with a
    single ``cache=model_cache(dae)`` at the factorization site, rather than
    threading a cache object through the loop."""
    c = getattr(obj, '_klu_cache', None)
    if c is None:
        c = KLUCache()
        try:
            obj._klu_cache = c
        except (AttributeError, TypeError):
            pass
    return c


def solve(A, b, backend=None, cache=None):
    """Single linear solve. ``cache`` (a KLUCache) reuses the KLU symbolic
    ordering across calls of the same pattern, e.g. across Newton iterations
    where the Jacobian structure is fixed."""
    if isinstance(A, (csc_array, csc_matrix, csr_array, csr_matrix)):
        if (resolve_backend(backend) == 'klu'
                and not np.iscomplexobj(A.data) and not np.iscomplexobj(b)):
            try:
                sym = cache.symbolic if cache is not None else None
                dec = klu_decomposition(A, symbolic=sym)
                if cache is not None:
                    cache.symbolic = dec.symbolic
                return dec.solve(b)
            except (NotImplementedError, OverflowError, RuntimeError):
                return sla.spsolve(A, b)
        return sla.spsolve(A, b)
    else:
        return np.linalg.solve(A, b)


def lu_decomposition(A: Union[np.ndarray, csc_array, csc_matrix],
                     backend: str = None,
                     cache: 'KLUCache' = None):
    """Factorize ``A`` and return an object exposing ``.solve(b)``.

    backend : None (default) | 'klu' | 'superlu'.
        None defers to the global selection (see set_linsolver / linsolver).
        'klu' silently falls back to superlu when libklu is unavailable or
        ``A`` is complex/dense.
    cache : KLUCache, optional
        Holds the reusable KLU symbolic ordering across calls of the same
        sparsity pattern. Owned by the model and threaded in by the solver.
    """
    if isinstance(A, np.ndarray):
        return dense_decomposition(A)
    if resolve_backend(backend) == 'klu' and not np.iscomplexobj(A.data):
        sym = cache.symbolic if cache is not None else None
        try:
            dec = klu_decomposition(A, symbolic=sym)
        except (NotImplementedError, OverflowError, RuntimeError):
            return sp_decomposition(A)
        if cache is not None:
            cache.symbolic = dec.symbolic
        return dec
    return sp_decomposition(A)


class dense_decomposition:
    def __init__(self,
                 A: np.ndarray):
        self.A = A

    def solve(self, b):
        return solve(self.A, b)


class sp_decomposition:
    def __init__(self,
                 A: Union[(csc_array, csc_matrix)]):
        self.splu = splu(A)
        self.perm_r = self.splu.perm_r
        self.perm_c = self.splu.perm_c
        self.L = self.splu.L
        self.U = self.splu.U
        self.nnz = self.splu.nnz

    def solve(self, b):
        return self.splu.solve(b)
