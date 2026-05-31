"""Optional SuiteSparse KLU sparse-LU backend.

Pure-Python ``ctypes`` binding to an already-installed ``libklu`` (SuiteSparse).
There is no build step: the shared library is ``dlopen``-ed at runtime, so
Solverz stays a pure-Python package and KLU is an *optional* backend that
activates only when ``libklu`` is found. When it is absent (or its ABI does not
match), :data:`KLU_AVAILABLE` is ``False`` and callers fall back to scipy
SuperLU.

KLU's symbolic factorization (BTF + AMD ordering) depends only on the matrix
*sparsity pattern*, which is invariant across a Solverz integration: the code
generator bakes fixed coordinate arrays, so only the values change with
``t / y / p``. :class:`klu_decomposition` therefore accepts a previously
computed :class:`KLUSymbolic` and reuses it (calling ``klu_factor`` with the
cached ordering), which is the regime that makes KLU roughly twice as fast as
SuperLU on power-system / IEGS Jacobians. The reusable symbolic is meant to be
held on the model object via a :class:`KLUCache` for the lifetime of a run.

Real matrices only. Complex matrices raise :class:`NotImplementedError`; the
``lu_decomposition`` dispatcher checks for this and routes complex systems to
scipy instead.

Set the environment variable ``SOLVERZ_LIBKLU`` to point at a specific
``libklu`` if automatic discovery does not find it.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from ctypes import (
    Structure, POINTER, byref,
    c_double, c_int, c_int32, c_size_t, c_void_p,
)

import numpy as np

__all__ = ["KLU_AVAILABLE", "libklu_path", "klu_decomposition", "KLUSymbolic", "KLUCache"]


# --------------------------------------------------------------------------- #
# Library discovery + dlopen (runtime only, no compilation)
# --------------------------------------------------------------------------- #
def _load_libklu():
    cands = []
    env = os.environ.get("SOLVERZ_LIBKLU")
    if env:
        cands.append(env)
    found = ctypes.util.find_library("klu")
    if found:
        cands.append(found)
    cands += [
        "libklu.so", "libklu.dylib", "klu.dll",
        "/opt/homebrew/lib/libklu.dylib", "/usr/local/lib/libklu.dylib",
        "/usr/local/lib/libklu.so", "/usr/lib/libklu.so",
    ]
    for c in cands:
        if not c:
            continue
        try:
            return ctypes.CDLL(c), c
        except OSError:
            continue
    return None, None


_lib, libklu_path = _load_libklu()
KLU_AVAILABLE = _lib is not None


# --------------------------------------------------------------------------- #
# klu_common: 24 fields, native alignment (mirrors klu.h)
# --------------------------------------------------------------------------- #
class _kluCommon(Structure):
    _fields_ = [
        ("tol", c_double), ("memgrow", c_double), ("initmem_amd", c_double),
        ("initmem", c_double), ("maxwork", c_double),
        ("btf", c_int), ("ordering", c_int), ("scale", c_int),
        ("user_order", c_void_p), ("user_data", c_void_p),
        ("halt_if_singular", c_int), ("status", c_int), ("nrealloc", c_int),
        ("structural_rank", c_int32), ("numerical_rank", c_int32),
        ("singular_col", c_int32), ("noffdiag", c_int32),
        ("flops", c_double), ("rcond", c_double), ("condest", c_double),
        ("rgrowth", c_double), ("work", c_double),
        ("memusage", c_size_t), ("mempeak", c_size_t),
    ]


_KLU_OK = 0

if KLU_AVAILABLE:
    _lib.klu_defaults.argtypes = [POINTER(_kluCommon)]
    _lib.klu_defaults.restype = c_int
    _lib.klu_analyze.argtypes = [c_int32, POINTER(c_int32), POINTER(c_int32), POINTER(_kluCommon)]
    _lib.klu_analyze.restype = c_void_p
    _lib.klu_factor.argtypes = [POINTER(c_int32), POINTER(c_int32), POINTER(c_double), c_void_p, POINTER(_kluCommon)]
    _lib.klu_factor.restype = c_void_p
    _lib.klu_solve.argtypes = [c_void_p, c_void_p, c_int32, c_int32, POINTER(c_double), POINTER(_kluCommon)]
    _lib.klu_solve.restype = c_int
    _lib.klu_free_symbolic.argtypes = [POINTER(c_void_p), POINTER(_kluCommon)]
    _lib.klu_free_symbolic.restype = c_int
    _lib.klu_free_numeric.argtypes = [POINTER(c_void_p), POINTER(_kluCommon)]
    _lib.klu_free_numeric.restype = c_int

    # Validate the struct layout once: klu_defaults must return KLU's documented
    # defaults. A mismatch means our ABI mirror is wrong for this build -> disable.
    _c = _kluCommon()
    _lib.klu_defaults(byref(_c))
    if not (_c.tol == 0.001 and _c.btf == 1 and _c.ordering == 0 and _c.scale == 2):
        KLU_AVAILABLE = False


def _fresh_common():
    c = _kluCommon()
    _lib.klu_defaults(byref(c))
    return c


# --------------------------------------------------------------------------- #
# Reusable symbolic factorization (owns the klu_symbolic* and frees it)
# --------------------------------------------------------------------------- #
class KLUSymbolic:
    """Owns a ``klu_symbolic*`` plus the pattern fingerprint it was built from.

    The fingerprint is ``(shape, nnz, indptr)``. Solverz guarantees the pattern
    is invariant within a run, so an ``indptr`` match (cheap, O(ncol)) together
    with the same shape and nnz is a sufficient reuse condition in practice.
    """

    __slots__ = ("ptr", "shape", "nnz", "indptr", "_common")

    def __init__(self, ptr, shape, nnz, indptr, common):
        self.ptr = ptr
        self.shape = shape
        self.nnz = nnz
        self.indptr = indptr           # int32 copy
        self._common = common          # keep the Common used at analyze alive

    def matches(self, shape, nnz, indptr):
        return (shape == self.shape and nnz == self.nnz
                and indptr.shape == self.indptr.shape
                and np.array_equal(indptr, self.indptr))

    def __del__(self):
        ptr = getattr(self, "ptr", None)
        if ptr and KLU_AVAILABLE:
            p = c_void_p(ptr)
            try:
                _lib.klu_free_symbolic(byref(p), byref(self._common))
            except Exception:
                pass
            self.ptr = None


class KLUCache:
    """Per-model holder for the reusable symbolic factorization.

    Stored on the model (``dae``/``ae``/``fdae``) by the solver and threaded
    into :func:`Solverz.solvers.laesolver.lu_decomposition` so the BTF+AMD
    ordering is computed once and reused for every step of a run.
    """

    __slots__ = ("symbolic",)

    def __init__(self):
        self.symbolic = None


def _as_int32_csc(A):
    A = A.tocsc()
    if np.iscomplexobj(A.data):
        raise NotImplementedError("KLU backend handles real matrices only")
    indptr = A.indptr
    indices = A.indices
    if indptr.dtype != np.int32:
        if int(indptr[-1]) >= 2 ** 31:
            raise OverflowError("nnz exceeds int32; KLU long (klu_l_*) API not wired")
        indptr = indptr.astype(np.int32)
    if indices.dtype != np.int32:
        indices = indices.astype(np.int32)
    data = np.ascontiguousarray(A.data, dtype=np.float64)
    return A.shape, indptr, indices, data


# --------------------------------------------------------------------------- #
# Decomposition object: matches sp_decomposition's .solve(b) interface
# --------------------------------------------------------------------------- #
class klu_decomposition:
    """KLU factorization of a scipy CSC matrix, with optional symbolic reuse.

    Parameters
    ----------
    A : scipy.sparse csc matrix/array (real)
    symbolic : KLUSymbolic, optional
        A symbolic factorization from a prior decomposition of the *same*
        pattern. When supplied and the fingerprint matches, ``klu_analyze`` is
        skipped and the cached ordering is reused.

    Use :attr:`symbolic` after construction to cache the (possibly newly
    created) ordering for the next step.
    """

    def __init__(self, A, symbolic=None, tol=None):
        if not KLU_AVAILABLE:
            raise RuntimeError("libklu not available")
        self._common = _fresh_common()
        if tol is not None:
            # KLU pivot tolerance: 0.001 (default) prefers the diagonal (fast,
            # circuit-style); 1.0 is partial pivoting (most accurate, scipy-like).
            self._common.tol = float(tol)
        shape, indptr, indices, data = _as_int32_csc(A)
        # keep buffers alive for the duration of analyze/factor
        self._indptr, self._indices, self._data = indptr, indices, data
        self.shape = shape
        self.nnz = int(indptr[-1])
        n = shape[0]
        Ap = indptr.ctypes.data_as(POINTER(c_int32))
        Ai = indices.ctypes.data_as(POINTER(c_int32))
        Ax = data.ctypes.data_as(POINTER(c_double))

        if symbolic is not None and symbolic.ptr and symbolic.matches(shape, self.nnz, indptr):
            self.symbolic = symbolic
        else:
            ptr = _lib.klu_analyze(n, Ap, Ai, byref(self._common))
            if not ptr:
                raise RuntimeError(f"klu_analyze failed (status {self._common.status})")
            self.symbolic = KLUSymbolic(ptr, shape, self.nnz, indptr.copy(), self._common)

        self._num = _lib.klu_factor(Ap, Ai, Ax, self.symbolic.ptr, byref(self._common))
        if not self._num or self._common.status != _KLU_OK:
            raise RuntimeError(f"klu_factor failed (status {self._common.status})")

    def solve(self, b):
        b = np.asarray(b)
        n = self.shape[0]
        if b.ndim == 1:
            x = np.array(b, dtype=np.float64, copy=True)
            nrhs = 1
        else:
            x = np.array(b, dtype=np.float64, order="F", copy=True)
            nrhs = x.shape[1]
        Xp = x.ctypes.data_as(POINTER(c_double))
        _lib.klu_solve(self.symbolic.ptr, self._num, n, nrhs, Xp, byref(self._common))
        if self._common.status != _KLU_OK:
            raise RuntimeError(f"klu_solve failed (status {self._common.status})")
        return x

    def __del__(self):
        num = getattr(self, "_num", None)
        if num and KLU_AVAILABLE:
            p = c_void_p(num)
            try:
                _lib.klu_free_numeric(byref(p), byref(self._common))
            except Exception:
                pass
            self._num = None
