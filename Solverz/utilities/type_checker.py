from sympy import Number as SymNumber, Integer
from sympy import Float
import numpy as np
from numpy import integer as NpInteger
from numbers import Number as PyNumber


def is_number(num):
    if isinstance(num, (PyNumber, SymNumber)):
        return True
    elif isinstance(num, np.ndarray) and num.size == 1:
        return True
    else:
        return False


def is_integer(num):
    if isinstance(num, (int, Integer, NpInteger)):
        return True
    else:
        return False


def is_vector(a: np.ndarray):
    if a.ndim == 1 and a.size > 1:
        return True


def is_scalar(a):
    if is_number(a):
        return True
    elif isinstance(a, np.ndarray):
        if a.ndim == 1 and a.size == 1:
            return True
    else:
        return False


def is_zero(a):
    if is_number(a):
        return Float(a) == Float(0)
    else:
        return False
