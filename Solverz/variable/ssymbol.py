from typing import Callable, Union, List
from numbers import Number

import numpy as np

from sympy import Expr, Symbol

from Solverz.sym_algebra.symbols import iVar, iAliasVar, Para, idx
from Solverz.num_api.Array import Array
from Solverz.utilities.type_checker import is_number


# operator = ['__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__truediv__', 'pow']


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
        self.init = sSym2Sym(init) if init is not None else None
        if self.Type == 'iVar':
            self.symbol = iVar(self.name)
        elif self.Type == 'Para':
            self.symbol = Para(self.name, dim=self.dim)
        elif self.Type == 'idx':
            self.symbol = idx(self.name)
        elif self.Type == 'iAliasVar':
            self.symbol = iAliasVar(self.name)

    def __neg__(self):
        return -self.symbol

    def __add__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__add__')(other)

    def __radd__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__radd__')(other)

    def __mul__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__mul__')(other)

    def __rmul__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__rmul__')(other)

    def __sub__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__sub__')(other)

    def __rsub__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__rsub__')(other)

    def __pow__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__pow__')(other)

    def __rpow__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__rpow__')(other)

    def __truediv__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__truediv__')(other)

    def __rtruediv__(self, other):
        other = sSym2Sym(other)
        return getattr(self.symbol, '__rtruediv__')(other)

    def __getitem__(self, index):
        index = index.symbol if isinstance(index, sSymBasic) else index
        if isinstance(index, tuple):
            index = tuple([convert_idx_in_slice(arg) if isinstance(arg, slice) else arg + 0 for arg in index])
        elif isinstance(index, slice):
            index = convert_idx_in_slice(index)
        elif isinstance(index, (int, np.integer, Expr)):
            index = index + 0  # convert IdxParam to idx symbol
        elif isinstance(index, (list, idx)):
            index = index
        else:
            raise TypeError(f"Index type {type(index)} not implemented!")
        return self.symbol[index]


class Var(sSymBasic):
    def __init__(self, name: str, value=None, init=None):
        super().__init__(name=name, Type='iVar', value=value, dim=1, init=init)


class AliasVar(sSymBasic):
    def __init__(self, name: str, step=1, value=None, init=None):
        name = name + '_tag_' + str(step - 1)
        super().__init__(name=name, Type='iAliasVar', value=value, dim=1, init=init)
        self.step = step


def sSym2Sym(x):
    if isinstance(x, sSymBasic):
        return x.symbol
    elif isinstance(x, (Expr, Symbol)):
        return x
    elif is_number(x):
        return x
    else:
        raise TypeError(f'Input type {type(x)} not supported!')


def sVar2Var(var: Union[Var, iVar, List[Union[iVar, Var]]]) -> Union[iVar, List[iVar]]:
    if isinstance(var, list):
        return [arg.symbol if isinstance(arg, Var) else arg for arg in var]
    else:
        return var.symbol if isinstance(var, Var) else var
