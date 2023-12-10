from typing import List, Dict

from sympy import Symbol, Function, Mul, Expr, symbols, Number, S, Add, Integer, sin, cos
from sympy import exp as spexp
from sympy.core.function import ArgumentIndexError
import numpy as np
from functools import reduce

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
            obj.value = np.array(value)
            if obj.value.ndim < 2:
                obj.value = obj.value.reshape((-1, 1))
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
            obj.value = obj.value.reshape(-1, )
        return obj

    def __getitem__(self, index):
        return Idxidx(self, index, dim=1)


class IdxVar(IdxSymBasic):
    pass


class IdxPara(IdxSymBasic):
    pass


class Idxidx(IdxSymBasic):
    pass


class Sum_(Function):  # no repeat name of sympy built-in func.
    r"""
    This function returns the summation.

    ``Sum_(f(k), (k, B, h))`` represents

    .. math::

        \sum_{k\in \mathbb{B}, k\neq h}f(k)

    """

    pass


class Sign(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) != 1:
            raise TypeError(f"Sum takes 1 positional arguments but {len(args)} were given!")

    def fdiff(self, argindex=1):
        # sign function should be treated as a constant.
        if argindex == 1:
            return 0
        raise ArgumentIndexError(self, argindex)


class MatrixFunction(Function):
    """
    The basic Function class of matrix computation
    """
    is_commutative = False
    dim = 2


class transpose(MatrixFunction):
    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            raise TypeError(f'Supports one operand while {len(args)} input!')


class Mat_Mul(MatrixFunction):

    @classmethod
    def eval(cls, *args):
        # This is to simplify the `0`, `1` and `-1` derived by matrix calculus, which shall be removed
        # by further improvements of the matrix calculus module.
        i = 0
        args = list(args)
        for arg in args:
            if arg == S.Zero:
                return 0
            elif arg == S.NegativeOne:
                del args[i]
                if len(args) > 1:
                    return -Mat_Mul(*args)
                else:
                    if i > 0:
                        return - args[0]
                    else:
                        return - args[1]
            elif arg == S.One:
                del args[i]
                if len(args) > 1:
                    return Mat_Mul(*args)
                else:
                    if i > 0:
                        return args[0]
                    else:
                        return args[1]
            i = i + 1

    def __repr__(self):
        return self.__str__()

    def _eval_derivative(self, s):
        # this is a mimic of sp.Mul and assumes that the equation declared satisfies the matrix computation rules
        terms = []
        for i in range(len(self.args)):
            args = list(self.args)
            # if i < len(args) - 1:
            #     args[-1] = Diag(args[-1])
            # Based on above assumption, the last element in Mat_Mul is always vector
            d = args[i].diff(s)
            if d:
                if isinstance(args[i], Diag) and i < len(args) - 1 and not isinstance(d, Number):
                    # Which means the original arg needs Diag() to expand itself as a matrix
                    # The last arg does not need Diag()
                    d = Diag(d)
                    # for arg in args[i+1:]:

                elif i == len(args) - 1 and not isinstance(d, Number):
                    d = Diag(d)
                else:
                    d = d

                #  * is replaced by @ in EqnDiff.__init__()
                terms.append(reduce(lambda x, y: x * y, (args[:i] + [d] + args[i + 1:]), S.One))

        return Add.fromiter(terms)

    def _latex(self, printer, **kwargs):

        arg_latex_str = []
        for arg in self.args:
            if isinstance(arg, Symbol):
                arg_latex_str = [*arg_latex_str, printer._print(arg)]
            else:
                arg_latex_str = [*arg_latex_str, r'\left (' + printer._print(arg) + r'\right )']
        _latex_str = arg_latex_str[0]
        for arg_latex_str_ in arg_latex_str[1:]:
            _latex_str = _latex_str + arg_latex_str_
        if 'exp' in kwargs.keys():
            return r'\left (' + _latex_str + r'\right )^{' + kwargs['exp'] + r'}'
        else:
            return _latex_str

    def _sympystr(self, printer, **kwargs):
        temp = printer._print(self.args[0])
        for arg in self.args[1:]:
            if isinstance(arg, (Symbol, Function)):
                temp += '@{operand}'.format(operand=printer._print(arg))
            else:
                temp += '@({operand})'.format(operand=printer._print(arg))
        return temp

    def _numpycode(self, printer, **kwargs):

        temp = printer._print(self.args[0])
        for arg in self.args[1:]:
            if isinstance(arg, (Symbol, Function)):
                temp += '@{operand}'.format(operand=printer._print(arg))
            else:
                temp += '@({operand})'.format(operand=printer._print(arg))
        return r'(' + temp + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Diag(MatrixFunction):

    def fdiff(self, argindex=1):
        return 1  # the 1 also means Identity matrix here

    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            raise TypeError(f"Diagonal takes 1 positional arguments but {len(args)} were given!")

        if args[0] == S.Zero:
            return 0
        elif isinstance(args[0], Number):
            return args[0]

    def _sympystr(self, printer, **kwargs):

        temp = 'diag({operand})'.format(operand=printer._print(self.args[0]))

        return temp

    def _latex(self, printer, **kwargs):

        _latex_str = r'\operatorname{diag}\left (' + printer._print(self.args[0]) + r'\right )'
        if 'exp' in kwargs.keys():
            return r'\left (' + _latex_str + r'\right )^{' + kwargs['exp'] + r'}'
        else:
            return _latex_str


class ElementwiseFunction(Function):
    pass


class Abs(Function):

    def fdiff(self, argindex=1):
        """
        Get the first derivative of the argument to Abs().

        """
        if argindex == 1:
            return sign(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


class exp(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            raise TypeError(f'Supports one operand while {len(args)} input!')

    def fdiff(self, argindex=1):
        return spexp(*self.args)


class sign(ElementwiseFunction):
    pass


class minmod(ElementwiseFunction):
    """


    """

    @classmethod
    def eval(cls, *args):
        if len(args) != 3:
            raise TypeError(f"minmod takes 3 positional arguments but {len(args)} were given!")

    def _eval_derivative(self, s):
        return switch(*[arg.diff(s) for arg in self.args], 0, minmod_flag(*self.args))


class minmod_flag(Function):
    """
    Different from `minmod`, minmod function outputs the position of args instead of the values of args.
    """

    @classmethod
    def eval(cls, *args):
        if len(args) != 3:
            raise TypeError(f"minmod takes 3 positional arguments but {len(args)} were given!")


class Slice(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) > 3:
            raise TypeError(f"minmod takes at most 3 positional arguments but {len(args)} were given!")


class switch(Function):
    def _eval_derivative(self, s):
        return switch(*[arg.diff(s) for arg in self.args[0:len(self.args) - 1]], self.args[-1])


def traverse_for_mul(node: Expr):
    """
    traverse the expression tree and replace sympy.Mul with Mat_Mul

    .. note::

        This is deprecated because now we should declare matrix/vector operation explicitly.

    This is

    Examples
    ========

    >>> from Solverz import Var, Param_
    >>> from Solverz.num.num_alg import traverse_for_mul
    >>> f = Var('f')
    >>> b = Var('b')
    >>> A = Param_('A', dim=2)
    >>> B = Param_('B', dim=2)
    >>> traverse_for_mul(-A*f)
    -(A@f)
    >>> traverse_for_mul(2*A*B*b)
    2*(A@B@b)
    """

    if isinstance(node, (Symbol, Number, Function)):
        return node
    elif isinstance(node, Mul):
        args = []
        for arg in node.args[1:]:
            args = [*args, traverse_for_mul(arg)]
        if isinstance(node.args[0], Number):
            return node.args[0] * Mat_Mul(*args)
        elif node.args[0] is S.NegativeOne:
            return -Mat_Mul(*args)
        else:
            return Mat_Mul(node.args[0], *args)
    else:
        args = []
        for arg in node.args:
            args = [*args, traverse_for_mul(arg)]
        return node.func(*args)


def pre_lambdify(expr: Expr):
    r"""

    """
    return expr
