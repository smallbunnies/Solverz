from typing import Dict

import numpy as np
from sympy import Symbol, Expr

from Solverz.numerical_interface.Array import Array

Sympify_Mapping = {}


def IndexPrinter(index):
    if not isinstance(index, (int, list, idx, IdxSymBasic, Expr, slice, tuple)):
        raise TypeError(f"Unsupported idx type {type(index)}")

    def print_(index_):
        if isinstance(index_, slice):
            start = index_.start if index_.start is not None else ''
            stop = index_.stop if index_.stop is not None else ''
            step = index_.step if index_.step is not None else ''
            return f'{start}:{stop}:{step}'
        elif isinstance(index_, (int, list, idx, IdxSymBasic, Expr)):
            return f'{index_}'

    if isinstance(index, tuple):
        if len(index) != 2:
            raise ValueError("Support only two element tuples!")
        else:
            return print_(index[0]) + ',' + print_(index[1])
    else:
        return print_(index)


def SymbolExtractor(index) -> Dict:
    if not isinstance(index, (int, list, idx, IdxSymBasic, Expr, slice, tuple)):
        raise TypeError(f"Unsupported idx type {type(index)}")

    temp = dict()

    if isinstance(index, (int, idx, IdxSymBasic)):
        if isinstance(index, idx):
            temp.update({index.name: index})
        elif isinstance(index, IdxSymBasic):
            temp.update({index.name0: index.symbol0})
            temp.update(index.SymInIndex)
    elif isinstance(index, list):
        for i in range(len(index)):
            temp.update(SymbolExtractor(index[i]))
    elif isinstance(index, Expr):
        for var_ in list(index.free_symbols):
            temp.update(SymbolExtractor(var_))
    elif isinstance(index, slice):
        if index.start is not None:
            temp.update(SymbolExtractor(index.start))
        if index.stop is not None:
            temp.update(SymbolExtractor(index.stop))
        if index.step is not None:
            temp.update(SymbolExtractor(index.step))
    elif isinstance(index, tuple):
        if len(index) != 2:
            raise ValueError("Support only two element tuples!")
        else:
            temp.update(SymbolExtractor(index[0]))
            temp.update(SymbolExtractor(index[1]))

    return temp


class SolSymBasic(Symbol):
    """
    Basic class for Solverz Symbols
    """
    _iterable = False  # sp.lambdify gets into infinite loop if _iterable == True

    def __new__(cls, name: str, value=None, dim: int = 1):
        obj = Symbol.__new__(cls, f'{name}')
        obj.name = f'{name}'
        obj.dim = dim
        if value is not None:
            obj.value = Array(value, dim)
        else:
            obj.value = None
        obj.initialized = True if value is not None else False
        return obj

    def __getitem__(self, index):
        pass


class IdxSymBasic(Symbol):
    """
    Basic class for Solverz indexed Symbols
    """

    def __new__(cls, symbol, index, dim):
        if not isinstance(index, (int, list, idx, IdxSymBasic, Expr, slice, tuple)):
            raise TypeError(f"Unsupported idx type {type(index)}")
        if not isinstance(symbol, Symbol):
            raise TypeError(f"Invalid symbol type {type(symbol)}")
        obj = Symbol.__new__(cls, f'{symbol.name}[{index}]')
        obj.symbol0 = symbol
        obj.index = index
        obj.name0 = symbol.name
        obj.name = obj.name0 + '[' + IndexPrinter(index) + ']'
        obj.SymInIndex = SymbolExtractor(index)
        obj.dim = dim

        return obj

    def _numpycode(self, printer, **kwargs):

        def IndexCodePrinter(index, printer):
            if isinstance(index, (int, list)):
                return '{i}'.format(i=printer._print(index))
            elif isinstance(index, idx):
                return '{i}'.format(i=printer._print(index))
            elif isinstance(index, Expr):
                return '{i}'.format(i=printer._print(index))
            elif isinstance(index, slice):
                start = IndexCodePrinter(index.start, printer) if index.start is not None else None
                stop = IndexCodePrinter(index.stop + 1, printer) if index.stop is not None else None
                step = IndexCodePrinter(index.step, printer) if index.step is not None else None
                return 'sol_slice({i}, {j}, {k})'.format(i=start, j=stop, k=step)

        if isinstance(self.index, tuple):
            if len(self.index) != 2:
                raise ValueError("Support only two element tuples!")
            else:
                temp = IndexCodePrinter(self.index[0], printer) + ',' + IndexCodePrinter(self.index[1], printer)
        else:
            temp = IndexCodePrinter(self.index, printer)

        return self.name0 + '[' + temp + ']'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Var(SolSymBasic):
    def __getitem__(self, index):
        return IdxVar(self, index, self.dim)


class Para(SolSymBasic):
    def __getitem__(self, index):
        return IdxPara(self, index, self.dim)


class idx(SolSymBasic):
    def __new__(cls, name: str, value=None, dim: int = 1):
        if dim > 1:
            raise ValueError("idx can only be one-dimensional")
        obj = SolSymBasic.__new__(cls, name, value, dim=1)
        if obj.value is not None:
            obj.value = Array(value, dim=1, dtype=int)
        return obj

    def __getitem__(self, index):
        return Idxidx(self, index, dim=1)


class IdxVar(IdxSymBasic):
    pass


class IdxPara(IdxSymBasic):
    pass


class Idxidx(IdxSymBasic):
    pass
