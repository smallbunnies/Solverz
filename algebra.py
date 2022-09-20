import numpy as np
from sympy import Function, sympify, Symbol, Mul, symbols, Matrix, lambdify, diag, Abs

Sympify_Mapping = {}


def implements_sympify(symbolic_func: str):
    """Register an __array_function__ implementation for PSArray objects."""

    def decorator(func):
        Sympify_Mapping[symbolic_func] = func
        return func

    return decorator


@implements_sympify('Mat_Mul')
class Mat_Mul(Function):

    def fdiff(self, argindex=1):

        if isinstance(self.args[argindex - 1], Diagonal):
            return Mul(Diagonal(Mul(*self.args[argindex:])), *self.args[0:argindex - 1])
        else:
            return Mul(*self.args[0:argindex - 1])


@implements_sympify('Diagonal')
class Diagonal(Function):

    def fdiff(self, argindex=1):
        return 1  # the 1 also means Identity matrix here
