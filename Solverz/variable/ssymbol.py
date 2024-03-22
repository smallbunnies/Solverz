from typing import Callable, Union, List
from numbers import Number

import numpy as np

from sympy import Expr, Symbol

from Solverz.sym_algebra.symbols import Var, AliasVar, Para, idx
from Solverz.num_api.Array import Array
from Solverz.sym_algebra.functions import Abs, sin, cos, exp, Sign, Mat_Mul
from Solverz.utilities.type_checker import is_number


# operator = ['__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__truediv__', 'pow']


def pre_process(other):
    if isinstance(other, sSymBasic):
        return other.symbol
    else:
        return other


def convert_idx_in_slice(s: slice):
    if isinstance(s, slice):
        start = s.start + 0 if s.start is not None else s.start
        stop = s.stop + 0 if s.stop is not None else s.stop
        step = s.step + 0 if s.step is not None else s.step
        s = slice(start, stop, step)
        return s


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
        if self.Type == 'Var':
            self.symbol = Var(self.name)
        elif self.Type == 'Para':
            self.symbol = Para(self.name, dim=self.dim)
        elif self.Type == 'idx':
            self.symbol = idx(self.name)

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
        if isinstance(index, tuple):
            index = tuple([convert_idx_in_slice(arg) if isinstance(arg, slice) else arg+0 for arg in index])
        elif isinstance(index, slice):
            index = convert_idx_in_slice(index)
        elif isinstance(index, (int, np.integer, Expr)):
            index = index + 0  # convert IdxParam to idx symbol
        elif isinstance(index, (list, idx)):
            index = index
        else:
            raise TypeError(f"Index type {type(index)} not implemented!")
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
