"""Maximum-product row matching for sparse LU factorizations.

KLU keeps the structural diagonal as its pivot sequence, because with its
default threshold of 0.001 it leaves the diagonal only for a pivot a
thousand times smaller than the largest entry of its column, and it obtains
that diagonal from the maximum transversal of the block triangular form,
which is blind to the magnitudes. For a Jacobian whose structural diagonal
is empty, as every LoopEqn model produces (rows follow the equation blocks,
columns follow the variables), the transversal picks one zero-free matching
among many, and the AMD ordering of that matching can carry several times
the fill of a magnitude-aware one. Measured on the SolUtil power flow of
MATPOWER case_ACTIVSg70k, the default matching gives 12.4 million nonzeros
in the factors and a 1.3 s factorization, the maximum-product matching
2.5 million and 52 ms.

:func:`max_product_matching` returns the row permutation ``perm`` such that
``A[perm]`` carries on its diagonal a set of entries whose product of
magnitudes is maximal, the criterion of the MC64 routine of HSL. It is a
minimum-cost assignment with the costs ``log max_i|a_ij| - log|a_ij|``,
solved by successive shortest augmenting paths with Johnson potentials
(Dijkstra on the reduced costs) after a greedy assignment of the zero-cost
entries, compiled with Numba. A work budget bounds the total number of heap
operations, so a pathological matrix costs a bounded amount of time and the
caller falls back to the unpermuted factorization.

SciPy's ``min_weight_full_bipartite_matching`` solves the same problem but
runs a Hopcroft-Karp feasibility pass first, and that pass does not
terminate in reasonable time on some chain-like patterns, for example the
iteration matrix of the cookbook IES model, which this implementation
matches in a few milliseconds.
"""
from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["max_product_matching"]

_INF = np.inf


@njit(cache=True)
def _heap_push(hd, hv, size, key, val):
    i = size
    hd[i] = key
    hv[i] = val
    while i > 0:
        p = (i - 1) >> 1
        if hd[p] <= hd[i]:
            break
        hd[p], hd[i] = hd[i], hd[p]
        hv[p], hv[i] = hv[i], hv[p]
        i = p
    return size + 1


@njit(cache=True)
def _heap_pop(hd, hv, size):
    key = hd[0]
    val = hv[0]
    size -= 1
    if size > 0:
        hd[0] = hd[size]
        hv[0] = hv[size]
        i = 0
        while True:
            l = 2 * i + 1
            r = l + 1
            m = i
            if l < size and hd[l] < hd[m]:
                m = l
            if r < size and hd[r] < hd[m]:
                m = r
            if m == i:
                break
            hd[m], hd[i] = hd[i], hd[m]
            hv[m], hv[i] = hv[i], hv[m]
            i = m
    return key, val, size


@njit(cache=True)
def _ssp_assignment(n, indptr, indices, cost, work_cap):
    """Successive shortest augmenting paths on the bipartite graph of the
    CSC pattern. Columns are nodes ``0..n-1``, rows ``n..2n-1``. Returns
    ``(match_col, ok, work)`` with ``match_col[j]`` the row assigned to
    column ``j``."""
    nnz = indptr[n]
    match_col = np.full(n, -1, np.int64)
    match_row = np.full(n, -1, np.int64)
    match_edge = np.full(n, -1, np.int64)
    pi = np.zeros(2 * n)
    for j in range(n):
        for k in range(indptr[j], indptr[j + 1]):
            i = indices[k]
            if cost[k] == 0.0 and match_row[i] < 0:
                match_col[j] = i
                match_row[i] = j
                match_edge[i] = k
                break
    d = np.full(2 * n, _INF)
    pred = np.full(2 * n, -1, np.int64)
    pred_edge = np.full(2 * n, -1, np.int64)
    touched = np.empty(2 * n, np.int64)
    settled = np.zeros(2 * n, np.bool_)
    cap = nnz + 2 * n + 8
    hd = np.empty(cap)
    hv = np.empty(cap, np.int64)
    work = 0
    for s in range(n):
        if match_col[s] >= 0:
            continue
        ntouched = 0
        size = 0
        d[s] = 0.0
        touched[ntouched] = s
        ntouched += 1
        size = _heap_push(hd, hv, size, 0.0, s)
        found = -1
        D = _INF
        while size > 0:
            dist, x, size = _heap_pop(hd, hv, size)
            work += 1
            if work > work_cap:
                return match_col, False, work
            if dist > d[x] or settled[x]:
                continue
            if dist >= D:
                break
            # A settled node keeps its predecessor. Rounding in the reduced
            # costs could otherwise re-relax it and close a cycle in the
            # predecessor chain that the augmentation walks.
            settled[x] = True
            if x < n:
                j = x
                for k in range(indptr[j], indptr[j + 1]):
                    i = indices[k]
                    if match_col[j] == i:
                        continue
                    node = n + i
                    if settled[node]:
                        continue
                    nd = dist + cost[k] + pi[j] - pi[node]
                    if nd < d[node]:
                        if d[node] == _INF:
                            touched[ntouched] = node
                            ntouched += 1
                        d[node] = nd
                        pred[node] = j
                        pred_edge[node] = k
                        if match_row[i] < 0:
                            if nd < D:
                                D = nd
                                found = i
                        else:
                            size = _heap_push(hd, hv, size, nd, node)
            else:
                i = x - n
                j2 = match_row[i]
                if j2 < 0 or settled[j2]:
                    continue
                k = match_edge[i]
                nd = dist - cost[k] + pi[x] - pi[j2]
                if nd < d[j2]:
                    if d[j2] == _INF:
                        touched[ntouched] = j2
                        ntouched += 1
                    d[j2] = nd
                    pred[j2] = x
                    size = _heap_push(hd, hv, size, nd, j2)
        if found < 0:
            return match_col, False, work
        for t in range(ntouched):
            x = touched[t]
            if d[x] < D:
                pi[x] -= (D - d[x])
        node = n + found
        while True:
            j = pred[node]
            k = pred_edge[node]
            i = node - n
            prev = match_col[j]
            match_col[j] = i
            match_row[i] = j
            match_edge[i] = k
            if j == s:
                break
            node = n + prev
        for t in range(ntouched):
            x = touched[t]
            d[x] = _INF
            pred[x] = -1
            pred_edge[x] = -1
            settled[x] = False
    return match_col, True, work


def max_product_matching(indptr, indices, data, n, work_factor=200.0):
    """Row permutation that maximizes the product of the diagonal magnitudes.

    Parameters
    ----------
    indptr, indices, data : arrays
        CSC arrays of a square real matrix (explicit zeros are ignored).
    n : int
        Order of the matrix.
    work_factor : float
        The work budget is ``work_factor * nnz + 1000 * n`` heap pops.

    Returns
    -------
    perm : ndarray of int64 or None
        ``perm[j]`` is the row placed on the diagonal of column ``j``, so
        ``A[perm]`` has the matched entries on its diagonal. ``None`` when a
        column is structurally empty, the matrix is structurally singular,
        or the work budget is exceeded.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    absdata = np.abs(np.asarray(data, dtype=np.float64))
    keep = absdata > 0
    if not keep.all():
        col_of = np.repeat(np.arange(n), np.diff(indptr))[keep]
        indices = indices[keep]
        absdata = absdata[keep]
        counts = np.bincount(col_of, minlength=n)
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    if n == 0 or np.any(np.diff(indptr) == 0):
        return None
    colmax = np.zeros(n)
    np.maximum.at(colmax, np.repeat(np.arange(n), np.diff(indptr)), absdata)
    cost = np.log(colmax[np.repeat(np.arange(n), np.diff(indptr))]) - np.log(absdata)
    cost[cost < 0] = 0.0
    work_cap = int(work_factor * absdata.size + 1000 * n)
    match_col, ok, _ = _ssp_assignment(n, indptr, indices, cost, work_cap)
    if not ok:
        return None
    return np.asarray(match_col, dtype=np.int64)
