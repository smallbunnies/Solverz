from __future__ import annotations

from functools import reduce
from typing import Type

import numpy as np
import sympy as sp
from numpy import linalg

from Solverz.sas.sas_alg import DT, Slice, search_for_func, dLinspace, Index

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
def DT_conv(*args, method='conv') -> float:
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
        y = np.flip(args[0].reshape((-1, 1)), 0)
        return (x@y).tolist()[0][0]

    if len(args) > 2 or method == 'fft':  # if input more than three vectors, use fft and ifft
        k = args[0].shape[0]
        y = []
        m = 2 * (k - 1) + 1  # ensure that we have enough function values to recover the coefficients by ifft
        n = np.ceil(np.log2(k))  # fft is the fastest when the length of the series is the power of 2
        for arg in args:
            # extend the length of the vector to the power of 2
            arg = np.pad(arg, (0, int(np.maximum(m, n) - k)), constant_values=0)
            y += [np.fft.fft(arg)]
        return np.real(np.fft.ifft(reduce(lambda a, b: a * b, y))[k - 1])


@implements_nfunc('dLinspace')
def linspace(start, end) -> np.ndarray:
    r"""

    Parameters
    ==========

    start:



    end:



    """
    return np.arange(start, end, dtype=int)[:, np.newaxis]


def lambdify(expr: Type[sp.Expr], modules=None):
    r"""
    Convert symbolic DT expressions into numerical functions, with the sub-purpose of extending the `Slice` and
    `dLinspace` objects by one.

    Examples
    ========

    >>> import numpy as np
    >>> from sympy import Derivative
    >>> from sympy.abc import y, t
    >>> import inspect
    >>> from Solverz.num.num_interface import lambdify
    >>> from Solverz.sas.sas_alg import dtify
    >>> from Solverz.eqn import Ode
    >>> test = Ode(name='test',eqn='2*(y-cos(t))',diff_var='y')
    >>> a=dtify(Derivative(y,t)-test.EQN,etf=True, k=Index('k'))
    >>> k = Index('k')
    >>> test_a = lambdify(a[1][1])
    >>> inspect.getsource(test_a)
    'def _lambdifygenerated(k):\n    return psi_t[k] + dConv_s(phi_t[0:k]*(k - dLinspace(0, k))/k, t[1:k + 1])\n'

    """
    expr_ = expr
    for DT_ in list(expr.free_symbols):
        if isinstance(DT_, DT):
            if isinstance(DT_.index, Slice):
                expr_ = expr_.subs(DT_, DT(DT_.symbol, Slice(DT_.index.start, DT_.index.end + 1)))

    dlinspaces = search_for_func(expr_, dLinspace)

    if len(dlinspaces) > 0:
        for dlinspace in dlinspaces:
            start = dlinspace.args[0]
            end = dlinspace.args[1]
            expr_ = expr_.subs(dlinspace, dLinspace(start, end + 1))

    return sp.lambdify([Index('k')], expr_, modules)


def inv(mat: np.ndarray):
    return linalg.inv(mat)
