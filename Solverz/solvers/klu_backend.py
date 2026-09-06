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

Before the symbolic analysis the rows are permuted by the maximum-product
matching of :mod:`Solverz.solvers.matching`, so that every column has its
largest entry, or an entry close to it, on the diagonal, and the block
triangular form is switched off so that KLU keeps that diagonal as its pivot
sequence. KLU's own transversal is blind to the magnitudes, and for a Jacobian
whose structural diagonal is empty, which every LoopEqn model produces, the
AMD ordering of that arbitrary matching can carry several times the fill of
the matched one; on the SolUtil power flow of MATPOWER case_ACTIVSg70k the
factorization drops from 1.3 s to 52 ms. The permutation is part of the cached
:class:`KLUSymbolic`, and :meth:`klu_decomposition.solve` applies it to the
right-hand side, so callers see the same interface as before. Systems below
``MATCHING_MIN_N`` unknowns (1000 by default) keep KLU's own transversal,
since the fixed cost of the matching exceeds their whole factorization. Set
the environment variable ``SOLVERZ_KLU_MATCHING=0`` or call
:func:`set_klu_matching` to fall back to KLU's own transversal everywhere.

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

__all__ = ["KLU_AVAILABLE", "libklu_path", "klu_decomposition", "KLUSymbolic", "KLUCache",
           "set_klu_matching", "klu_matching_enabled"]

# Row matching before the KLU ordering (see the module docstring). Off when
# the environment variable is "0"; toggled at runtime by set_klu_matching.
# Below MATCHING_MIN_N unknowns the matching is skipped: its fixed cost of
# 0.1 to 0.5 ms per pattern exceeds the whole factorization of such systems,
# and the fill it saves there is negligible. Measured gains start around
# 2 000 unknowns and reach a factor of 8 in the Newton solve at 140 000.
_MATCHING = os.environ.get("SOLVERZ_KLU_MATCHING", "1").strip() != "0"
MATCHING_MIN_N = int(os.environ.get("SOLVERZ_KLU_MATCHING_MIN_N", "1000"))


def set_klu_matching(enabled: bool, min_n: int = None) -> None:
    """Enable or disable the maximum-product row matching before the KLU
    symbolic analysis, and optionally set the smallest number of unknowns it
    applies to. Takes effect for the next symbolic analysis; a cached
    :class:`KLUSymbolic` keeps the setting it was built with."""
    global _MATCHING, MATCHING_MIN_N
    _MATCHING = bool(enabled)
    if min_n is not None:
        MATCHING_MIN_N = int(min_n)


def klu_matching_enabled() -> bool:
    return _MATCHING


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

    ``perm`` is the row permutation applied before the analysis (``None`` when
    the matching is off or failed), ``indices_p`` the row indices of the
    permuted pattern and ``gather`` the entry order that turns the caller's
    CSC data into the data of the permuted matrix, so every later numeric
    factorization of the same pattern costs one gather and no sort.
    """

    __slots__ = ("ptr", "shape", "nnz", "indptr", "_common", "perm", "indices_p", "gather")

    def __init__(self, ptr, shape, nnz, indptr, common, perm=None, indices_p=None, gather=None):
        self.ptr = ptr
        self.shape = shape
        self.nnz = nnz
        self.indptr = indptr           # int32 copy
        self._common = common          # keep the Common used at analyze alive
        self.perm = perm
        self.indices_p = indices_p
        self.gather = gather

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


def _row_matching(n, indptr, indices, data):
    """Maximum-product row permutation of the CSC matrix and the arrays that
    apply it to the pattern once and to the data at every factorization.

    Returns ``(perm, indices_p, gather)`` or ``(None, None, None)`` when the
    matching is unavailable. ``A[perm]`` has the matched entries on its
    diagonal; ``gather`` reorders the entries of ``A.data`` so that
    ``(data[gather], indices_p, indptr)`` is the CSC form of ``A[perm]`` with
    sorted row indices, which ``klu_analyze`` and ``klu_factor`` expect.
    """
    try:
        from Solverz.solvers.matching import max_product_matching
        perm = max_product_matching(indptr, indices, data, n)
    except Exception:
        return None, None, None
    if perm is None:
        return None, None, None
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    new_rows = inv[np.asarray(indices, dtype=np.int64)]
    # one stable sort on the combined (column, new row) key; a lexsort on
    # two keys costs an order of magnitude more on large patterns
    key = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr)) * n + new_rows
    gather = np.argsort(key, kind='stable')
    indices_p = np.ascontiguousarray(new_rows[gather], dtype=np.int32)
    return perm, indices_p, gather


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

    def __init__(self, A, symbolic=None, tol=None, matching=None):
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

        if symbolic is not None and symbolic.ptr and symbolic.matches(shape, self.nnz, indptr):
            self.symbolic = symbolic
        else:
            # ``matching=None`` follows the global switch and the size
            # threshold; an explicit True or False overrides both.
            use_matching = (_MATCHING and n >= MATCHING_MIN_N) if matching is None else bool(matching)
            perm = indices_p = gather = None
            if use_matching and n > 1:
                perm, indices_p, gather = _row_matching(n, indptr, indices, data)
            # KLU keeps the diagonal it is given only without the block
            # triangular form; with BTF on it would recompute a structural
            # transversal and discard the matching.
            self._common.btf = 0 if perm is not None else 1
            Ai_a = (indices_p if indices_p is not None else indices).ctypes.data_as(POINTER(c_int32))
            ptr = _lib.klu_analyze(n, Ap, Ai_a, byref(self._common))
            if not ptr:
                raise RuntimeError(f"klu_analyze failed (status {self._common.status})")
            self.symbolic = KLUSymbolic(ptr, shape, self.nnz, indptr.copy(), self._common,
                                        perm=perm, indices_p=indices_p, gather=gather)

        sym = self.symbolic
        if sym.perm is not None:
            self._common.btf = 0
            self._indices = sym.indices_p
            self._data = data = np.ascontiguousarray(data[sym.gather])
        Ai = self._indices.ctypes.data_as(POINTER(c_int32))
        Ax = data.ctypes.data_as(POINTER(c_double))
        self._num = _lib.klu_factor(Ap, Ai, Ax, sym.ptr, byref(self._common))
        if not self._num or self._common.status != _KLU_OK:
            raise RuntimeError(f"klu_factor failed (status {self._common.status})")

    def solve(self, b):
        b = np.asarray(b)
        n = self.shape[0]
        perm = self.symbolic.perm
        if b.ndim == 1:
            x = np.array(b[perm] if perm is not None else b, dtype=np.float64, copy=True)
            nrhs = 1
        else:
            x = np.array(b[perm] if perm is not None else b, dtype=np.float64, order="F", copy=True)
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
