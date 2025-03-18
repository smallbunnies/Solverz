from typing import Dict

import numpy as np
from sympy import Symbol, Expr

from Solverz.num_api.Array import Array

Sympify_Mapping = {}


def IndexPrinter(index):
    if not isinstance(index, (int, np.integer, list, idx, IdxSymBasic, Expr, slice, tuple)):
        raise TypeError(f"Unsupported idx type {type(index)}")

    def print_(index_):
        if isinstance(index_, slice):
            start = index_.start if index_.start is not None else ''
            stop = index_.stop if index_.stop is not None else ''
            slice_str = f'{start}:{stop}'
            slice_str += f':{index_.step}' if index_.step is not None else ''
            return slice_str
        elif isinstance(index_, (int, np.integer, list, idx, IdxSymBasic, Expr)):
            return f'{index_}'

    if isinstance(index, tuple):
        if len(index) != 2:
            raise ValueError("Support only two element tuples!")
        else:
            return print_(index[0]) + ',' + print_(index[1])
    else:
        return print_(index)


def SymbolExtractor(index) -> Dict:
    if not isinstance(index, (int, np.integer, list, idx, IdxSymBasic, Expr, slice, tuple)):
        raise TypeError(f"Unsupported idx type {type(index)}")

    temp = dict()

    if isinstance(index, (int, np.integer, idx, IdxSymBasic)):
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


Solverz_internal_name = ['y_', 'F_', 'p_', 'J_',
                         'row', 'col', 'data', '_F_', 'data_', 'Hvp_', 'v_', 'row_hvp', 'col_hvp', 'data_hvp',
                         '_data_', '_data_hvp']


class SolSymBasic(Symbol):
    """
    Basic class for Solverz Symbols
    """
    _iterable = False  # sp.lambdify gets into infinite loop if _iterable == True
    is_real = True

    def __new__(cls, name: str, value=None, dim: int = 1, internal_use=False):
        if any([name == built_in_name for built_in_name in Solverz_internal_name]):
            if not internal_use:
                raise ValueError(
                    f"Solverz built-in name {name}, cannot be used as variable name.")
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
    is_real = True

    def __new__(cls, symbol, index, dim):
        if not isinstance(index, (int, np.integer, list, idx, IdxSymBasic, Expr, slice, tuple)):
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
            if isinstance(index, (int, np.integer, list)):
                return '{i}'.format(i=printer._print(index))
            elif isinstance(index, idx):
                return '{i}'.format(i=printer._print(index))
            elif isinstance(index, Expr):
                return '{i}'.format(i=printer._print(index))
            elif isinstance(index, slice):
                start = index.start
                stop = index.stop
                step = index.step
                if any([isinstance(arg, (idx, Expr)) for arg in [start, stop, step]]):
                    start = IndexCodePrinter(
                        start, printer) if start is not None else None
                    stop = IndexCodePrinter(
                        stop, printer) if stop is not None else None
                    step = IndexCodePrinter(
                        step, printer) if step is not None else None
                    return 'sol_slice({i}, {j}, {k})'.format(i=start, j=stop, k=step)
                else:
                    start = IndexCodePrinter(
                        start, printer) if start is not None else ''
                    stop = IndexCodePrinter(
                        stop, printer) if stop is not None else ''
                    slice_str = '{i}:{j}'.format(i=start, j=stop)
                    slice_str += f':{IndexCodePrinter(step, printer)}' if step is not None else ''
                    return slice_str

        if isinstance(self.index, tuple):
            if len(self.index) != 2:
                raise ValueError("Support only two element tuples!")
            else:
                temp = IndexCodePrinter(
                    self.index[0], printer) + ',' + IndexCodePrinter(self.index[1], printer)
        else:
            temp = IndexCodePrinter(self.index, printer)

        return self.name0 + '[' + temp + ']'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class iVar(SolSymBasic):
    def __getitem__(self, index):
        return IdxVar(self, index, self.dim)


class iAliasVar(SolSymBasic):
    def __getitem__(self, index):
        return IdxAliasVar(self, index, self.dim)


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


class IdxAliasVar(IdxSymBasic):
    pass


class IdxPara(IdxSymBasic):
    pass


class Idxidx(IdxSymBasic):
    pass


class SolDict(Symbol):
    """
    Solverz' Dict class for numerical equation code printer, which accepts only str index
    """
    _iterable = False  # sp.lambdify gets into infinite loop if _iterable == True

    def __new__(cls, name: str):
        obj = Symbol.__new__(cls, f'{name}')
        obj.name = f'{name}'
        return obj

    def __getitem__(self, index):
        return IdxDict(self, index)


class IdxDict(Symbol):
    """
    Indexed Dicts
    """

    def __new__(cls, symbol, index):
        if not isinstance(index, str):
            raise TypeError(f"Unsupported idx type {type(index)}")
        if not isinstance(symbol, SolDict):
            raise TypeError(f"Invalid symbol type {type(symbol)}")
        obj = Symbol.__new__(cls, f'{symbol.name}["{index}"]')
        obj.index = index

        return obj

    def _numpycode(self, printer, **kwargs):
        return self.name

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class coo_2_csc(Symbol):

    def __new__(cls, eqn_size: int, vsize: int):
        obj = Symbol.__new__(cls, f'coo_2_csc')
        obj.eqn_size = eqn_size
        obj.vsize = vsize
        return obj

    def _numpycode(self, printer, **kwargs):
        return f'sps.coo_array((data, (row, col)), ({self.eqn_size}, {self.vsize})).tocsc()'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class coo_2_csc_hvp(Symbol):

    def __new__(cls, eqn_size: int, vsize: int):
        obj = Symbol.__new__(cls, f'coo_2_csc')
        obj.eqn_size = eqn_size
        obj.vsize = vsize
        return obj

    def _numpycode(self, printer, **kwargs):
        return f'sps.coo_array((data_hvp, (row_hvp, col_hvp)), ({self.eqn_size}, {self.vsize})).tocsc()'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)
