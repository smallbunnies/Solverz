from __future__ import annotations

from functools import reduce

import numpy as np

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
def conv(*args) -> np.ndarray:
    """
    Generate diagonal matrix of given vector X
    :PARAM X: vector
    :return: diagonal matrix
    """
    return np.diag(x.reshape(-1, ))
