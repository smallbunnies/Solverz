from __future__ import annotations

from functools import reduce

import numpy as np
from numpy import linalg
from scipy.sparse import diags, linalg as splinalg

numerical_interface = {}


def implements_nfunc(nfunc_name: str):
    """Register an DT function implementation for sympy Expr."""

    def decorator(func):
        numerical_interface[nfunc_name] = func
        return func

    return decorator


@implements_nfunc('Sign')
def _sign(arg):
    return np.sign(arg)


@implements_nfunc('Diag')
def diag(x) -> np.ndarray:
    """
    Generate diagonal matrix of given vector X
    :PARAM X: vector
    :return: diagonal matrix
    """
    if not isinstance(x, np.ndarray):
        return diags(x.toarray().reshape(-1,), 0, format='csc')
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
    return splinalg.spsolve(A, b)
