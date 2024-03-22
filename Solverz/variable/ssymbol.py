from typing import Callable, Union, List
from numbers import Number

import numpy as np

from sympy import Expr, Symbol

from Solverz.sym_algebra.symbols import Var, AliasVar, Para
from Solverz.num_api.Array import Array
from Solverz.sym_algebra.functions import Abs, sin, cos, exp, Sign, Mat_Mul
from Solverz.utilities.type_checker import is_number


# operator = ['__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__truediv__', 'pow']


def pre_process(other):
    if isinstance(other, sSymBasic):
        return other.symbol
    else:
        return other


class sSymBasic:
    """
    Basic class for Solverz sSym
    """

    def __init__(self, name: str, Type: str, value=None, dim: int = 1, init=None):
        self.name = f'{name}'
        self.dim = dim
        if value is not None:
            self.value = Array(value, dim)
        else:
            self.value = None
        self.Type = Type
        self.init = init
        self.symbol = Var(self.name) if self.Type == 'Var' else Para(self.name, dim=self.dim)

    def __neg__(self):
        return -self.symbol

    def __add__(self, other):
        other = pre_process(other)
        return getattr(self.symbol, '__add__')(other)

    def __radd__(self, other):
        other = pre_process(other)
        return getattr(self.symbol, '__radd__')(other)

    def __mul__(self, other):
        other = pre_process(other)
        return getattr(self.symbol, '__mul__')(other)

    def __rmul__(self, other):
        other = pre_process(other)
        return getattr(self.symbol, '__rmul__')(other)

    def __sub__(self, other):
        other = pre_process(other)
        return getattr(self.symbol, '__sub__')(other)

    def __rsub__(self, other):
        other = pre_process(other)
        return getattr(self.symbol, '__rsub__')(other)

    def __pow__(self, other):
        other = pre_process(other)
        return getattr(self.symbol, '__pow__')(other)

    def __truediv__(self, other):
        other = pre_process(other)
        return getattr(self.symbol, '__truediv__')(other)

    def __getitem__(self, index):
        index = pre_process(index)
        return self.symbol[index]


class sVar(sSymBasic):
    def __init__(self, name: str, value=None, dim: int = 1, init=None):
        super().__init__(name=name, Type='Var', value=value, dim=dim, init=init)


class sAliasVar(sSymBasic):
    def __init__(self, name: str, value=None, dim: int = 1, init=None):
        super().__init__(name=name, Type='Var', value=value, dim=dim, init=init)


def sSym2Sym(x):
    if isinstance(x, sSymBasic):
        return x.symbol
    elif isinstance(x, (Expr, Symbol)):
        return x
    elif is_number(x):
        return x
    else:
        raise TypeError(f'Input type {type(x)} not supported!')


def sAbs(x):
    return Abs(sSym2Sym(x))


def sSin(x):
    return sin(sSym2Sym(x))


def sCos(x):
    return cos(sSym2Sym(x))


def sExp(x):
    return exp(sSym2Sym(x))


def sSign(x):
    return Sign(sSym2Sym(x))


def sMat_Mul(*args):
    return Mat_Mul(*[pre_process(arg) for arg in args])
