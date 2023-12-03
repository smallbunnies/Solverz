from __future__ import annotations

from functools import reduce

import numpy as np
from numpy import linalg
from scipy.sparse import diags, csc_array, linalg as sla
from cvxopt.umfpack import linsolve
from cvxopt import matrix, spmatrix

numerical_interface = {}


def implements_nfunc(nfunc_name: str):
    """Register an DT function implementation for sympy Expr."""

    def decorator(func):
        numerical_interface[nfunc_name] = func
        return func

    return decorator


@implements_nfunc('slice_')
def slice_(*args):
    """
    This is used to convert the slice arguments to int
    """
    return slice(*[int(arg_) if isinstance(arg_, np.ndarray) else arg_ for arg_ in args])


@implements_nfunc('Slice')
def Slice(*args):
    """
    This is used to evaluate the slice index of IdxVar/IdxParam/IdxConst
    """
    return slice_(*args)


@implements_nfunc('ix_')
def ix_(arg: np.ndarray):
    return arg.reshape(-1,)


@implements_nfunc('Sign')
def _sign(arg):
    return np.sign(arg)


@implements_nfunc('minmod')
def minmod(*args):
    if len(args) != 3:
        raise ValueError("Input arg length must be 3")

    shapes = [arg.shape[0] for arg in args]

    a, b, c = args
    stacked_array = np.hstack((a, b, c))
    if all(x == shapes[0] for x in shapes):
        cd1 = (a > 0) & (b > 0) & (c > 0)
        cd2 = (a < 0) & (b < 0) & (c < 0)
        conditions = [cd1,
                      cd2]
        choice_list = [np.min(stacked_array, axis=1).reshape((-1, 1)),
                       np.max(stacked_array, axis=1).reshape((-1, 1))]
    else:
        raise ValueError(f"Length of Input array not consistent {shapes}")
    return np.select(conditions, choice_list, 0)


@implements_nfunc('minmod_flag')
def minmod_flag(*args):
    if len(args) != 3:
        raise ValueError("Input arg length must be 3")

    shapes = [arg.shape[0] for arg in args]

    a, b, c = args
    stacked_array = np.hstack((a, b, c))
    if all(x == shapes[0] for x in shapes):
        cd1 = (a > 0) & (b > 0) & (c > 0)
        cd2 = (a < 0) & (b < 0) & (c < 0)
        conditions = [cd1,
                      cd2]
        choice_list = [np.argmin(stacked_array, axis=1).reshape((-1, 1)),
                       np.argmax(stacked_array, axis=1).reshape((-1, 1))]
    else:
        raise ValueError(f"Length of Input array not consistent {shapes}")
    return np.select(conditions, choice_list, 3)


@implements_nfunc('switch')
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


@implements_nfunc('Diag')
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


@implements_nfunc('dConv_s')
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


@implements_nfunc('dLinspace')
def linspace(start, end) -> np.ndarray:
    r"""

    Parameters
    ==========

    start:



    end:



    """
    return np.arange(start, end, dtype=int)[:, np.newaxis]


def inv(mat: np.ndarray):
    return linalg.inv(mat)


def solve(A, b):
    if isinstance(A, spmatrix):
        # return splinalg.spsolve(A, b)
        c = matrix(b)
        linsolve(A, c)
        return np.array(c)
    elif isinstance(A, csc_array):
        return sla.spsolve(A, b)
