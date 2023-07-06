from __future__ import annotations

from functools import reduce

import numpy as np
from numpy import linalg

numerical_interface = {}


def implements_nfunc(nfunc_name: str):
    """Register an DT function implementation for sympy Expr."""

    def decorator(func):
        numerical_interface[nfunc_name] = func
        return func

    return decorator


@implements_nfunc('Mat_Mul')
def matmul(*args: np.ndarray) -> np.ndarray:
    """
    np.multiply supports only two arguments
    So a new function is presented to generate multiplication of np.ndarray
    :PARAM args: np.ndarray
    :return:
    """

    return reduce(np.dot, args)


@implements_nfunc('Diagonal')
def diag(x: np.ndarray) -> np.ndarray:
    """
    Generate diagonal matrix of given vector X
    :PARAM X: vector
    :return: diagonal matrix
    """
    return np.diag(x.reshape(-1, ))


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
