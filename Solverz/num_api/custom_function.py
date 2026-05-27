from __future__ import annotations

from functools import reduce

import warnings
import numpy as np
from numpy import linalg
from scipy.sparse import diags, csc_array, coo_array, linalg as sla, issparse
from numba import njit


def mutable_mat_fallback_extract(expr_value, rows, cols):
    """Fallback index extraction for a mutable-matrix Jacobian block.

    ``expr_value`` may be either a scipy.sparse matrix/array (common
    case: the expression contains ``Mat_Mul`` of sparse params) or a
    dense ``numpy.ndarray`` (common case: the expression mixes in a
    ``dim=2 sparse=False`` parameter). We dispatch on type so that the
    generated J_ wrapper evaluates the expression once and indexes it
    correctly regardless of which shape came out.

    Parameters
    ----------
    expr_value : scipy.sparse matrix/array or numpy.ndarray
        The evaluated mutable-matrix block expression.
    rows, cols : sequence of int
        Row / column indices into ``expr_value``; same length.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``len(rows)`` with ``expr_value[r_i, c_i]``.
    """
    if issparse(expr_value):
        return np.asarray(expr_value.tocsr()[rows, cols]).ravel()
    return np.asarray(expr_value)[rows, cols]


# from cvxopt.umfpack import linsolve
# from cvxopt import matrix, spmatrix


def sol_slice(*args):
    """
    This is used to convert the slice arguments to int
    """
    return slice(*[int(arg_[0]) if isinstance(arg_, np.ndarray) else arg_ for arg_ in args])


def Slice(*args):
    """
    This is used to evaluate the slice index of IdxVar/IdxParam/IdxConst
    """
    return sol_slice(*args)


def ix_(arg: np.ndarray):
    return arg.reshape(-1, )


def _sign(arg):
    return np.sign(arg)


@njit(cache=True)
def Heaviside(x):
    return np.where(x >= 0, 1.0, 0.0)


@njit(cache=True)
def switch(*args):
    flag = args[-1]
    flag_shape = args[-1].shape
    v_list = list(args[0:len(args) - 1])

    for i in range(len(v_list)):
        v = v_list[i]
        if isinstance(v, (int, float)):
            v_list[i] = v * np.ones(flag_shape)
        elif isinstance(v, np.ndarray):
            if v.shape[0] == 1:
                v_list[i] = v * np.ones(flag_shape)
    shapes = [v.shape[0] for v in v_list]
    if all(x.shape[0] == v_list[0].shape[0] for x in v_list):
        conditions = [flag == i for i in range(len(args) - 1)]
        choice_list = v_list
    else:
        raise ValueError(f"Length of Input array not consistent {shapes}")
    return np.select(conditions, choice_list, 0)


@njit(cache=True)
def Saturation(x, xmin, xmax):
    # np.minimum / np.maximum are element-wise and shape-preserving
    # under Numba @njit: scalar input -> scalar output, ndarray input
    # -> ndarray output. The previous ``np.asarray(x).reshape((-1,))``
    # path forced a 1-D array return even for scalar input, which
    # made the function unusable inside ``LoopEqn`` bodies: the
    # rendered ``out[i] = ... Saturation(scalar) ...`` becomes
    # ``out[i] = <1-D array>``, which Numba's nopython mode rejects
    # as an invalid setitem signature. The new implementation
    # preserves scalar-ness, so both scalar Eqn and LoopEqn bodies
    # JIT cleanly without losing array semantics for callers that
    # do pass arrays.
    return np.minimum(xmax, np.maximum(xmin, x))


@njit(cache=True)
def In(x, xmin, xmax):
    # Shape-preserving sibling of :func:`Saturation`'s derivative
    # (returns 1 inside ``[xmin, xmax]``, 0 outside). The previous
    # ``np.asarray(x).reshape((-1,))`` path forced a 1-D return even
    # for scalar input, which made the function unusable inside the
    # ``LoopEqn`` Jacobian kernels. ``np.where`` is element-wise and
    # preserves input shape under Numba ``@njit``.
    return np.where((x >= xmin) & (x <= xmax), np.int32(1), np.int32(0))


@njit(cache=True)
def GreaterThan(x, y):
    x = np.asarray(x).reshape((-1,))
    return (x > y).astype(np.int32)


@njit(cache=True)
def LessThan(x, y):
    x = np.asarray(x).reshape((-1,))
    return (x < y).astype(np.int32)


@njit(cache=True)
def And(x, y):
    x = np.asarray(x).reshape((-1,))
    return x & y


@njit(cache=True)
def Or(x, y):
    x = np.asarray(x).reshape((-1,))
    return x | y


@njit(cache=True)
def Not(x):
    x = np.asarray(x).reshape((-1,))
    return np.ones_like(x) - x


def diag(x) -> np.ndarray:
    """
    Generate diagonal matrix of given vector X
    :PARAM X: vector
    :return: diagonal matrix
    """
    if not isinstance(x, np.ndarray):
        return diags(x.toarray().reshape(-1, ), 0, format='csc')
    else:
        return np.diagflat(x)


def DT_conv(*args, method='conv') -> np.ndarray:
    r"""
    Perform the convolutions in DT computations.

    Explanation
    ===========



    Parameters
    ==========

    args : np.ndarray

        DT series.

    method : str

        the method used to compute DT convolution

    """
    if len(args) <= 2 and method == 'conv':  # if input two vectors, then use scalar multiplications and additions
        x = args[0].reshape((1, -1))
        y = np.flip(args[1].reshape((-1, 1)), 0)
        return x @ y

    if len(args) > 2 or method == 'fft':  # if input more than three vectors, use fft and ifft
        k = args[0].shape[0]
        y = []
        m = 2 * (k - 1) + 1  # ensure that we have enough function values to recover the coefficients by ifft
        n = np.ceil(np.log2(k))  # fft is the fastest when the length of the series is the power of 2
        for arg in args:
            # extend the length of the vector to the power of 2
            arg = np.pad(arg, (0, int(np.maximum(m, n) - k)), constant_values=0)
            y += [np.fft.fft(arg)]
        return np.array(np.real(np.fft.ifft(reduce(lambda a, b: a * b, y))[k - 1]))


def linspace(start, end) -> np.ndarray:
    r"""

    Parameters
    ==========

    start:



    end:



    """
    return np.arange(start, end, dtype=int)[:, np.newaxis]


@njit(cache=True)
def csc_matvec(data, indices, indptr, shape, x):
    res = np.zeros(shape[0], dtype=float)
    for j in range(len(indptr) - 1):
        start, end = indptr[j], indptr[j + 1]
        for idx in range(start, end):
            i = indices[idx]
            res[i] += data[idx] * x[j]
    return res
