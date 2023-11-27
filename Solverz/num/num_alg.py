from sympy import Symbol, Function, Mul, Expr, symbols, Number, S, Add, Integer, sin, cos
from sympy import exp as spexp
from sympy import Abs as spabs
from sympy.core.function import ArgumentIndexError
from scipy.sparse import csc_array
import numpy as np
from typing import Union
from functools import reduce

Sympify_Mapping = {}


def print_slice(slice_obj: slice):
    start = slice_obj.start if slice_obj.start is not None else ''
    stop = slice_obj.stop if slice_obj.stop is not None else ''
    step = slice_obj.step if slice_obj.step is not None else ''

    return f'{start}:{stop}:{step}'


class idx(Symbol):
    is_Integer = True

    def __new__(cls, name: str, value=None):
        obj = Symbol.__new__(cls, f'{name}')
        obj.is_Integer = True
        obj.name = f'{name}'
        if value is not None:
            if isinstance(value, (list, int)):
                value = np.array(value, dtype=int)
            if value.ndim > 1:
                raise TypeError('Only support 1-dim index value')
            obj.value = np.asarray(value, dtype=int).reshape((-1,))
        else:
            obj.value = None
        obj.initialized = True if value is not None else False
        return obj

    def __array__(self):
        return self.value


class Var(Symbol):
    _iterable = False  # sp.lambdify gets into infinite loop if _iterable == True

    def __new__(cls, name, value=None):
        obj = Symbol.__new__(cls, name)
        obj.name = name
        if value is not None:
            obj.value = np.array(value).reshape((-1, 1))
        else:
            obj.value = None
        obj.dim = 1
        obj.initialized = True if value is not None else False
        return obj

    def __getitem__(self, item):
        return IdxVar(self, item)


class IdxVar(Symbol):

    def __new__(cls, symbol, index):
        if not isinstance(index, (idx, int, slice, list, Expr)):
            raise TypeError(f"Unsupported idx type {type(index)}")
        obj = Symbol.__new__(cls, f'{symbol.name}[{index}]')
        obj.symbol = symbol
        obj.index = index
        obj.symbol_name = symbol.name
        if isinstance(index, slice):
            obj.name = f'{symbol.name}[{print_slice(index)}]'
        else:
            obj.name = f'{symbol.name}[{index}]'

        # in case the index is Expression
        obj.symbol_in_index = dict()  # Dict[str, Symbol]
        if isinstance(index, idx):
            obj.symbol_in_index[index.name] = index
        elif isinstance(index, slice):
            # in case start/stop of the slice index is not Number but expression, for example in PDE we have variable
            # like x[1:M-1] where
            if isinstance(index.start, (Expr, Symbol)):
                for symbol_ in list(index.start.free_symbols):
                    obj.symbol_in_index[symbol_.name] = symbol_
            if isinstance(index.stop, (Expr, Symbol)):
                for symbol_ in list(index.stop.free_symbols):
                    obj.symbol_in_index[symbol_.name] = symbol_
            if isinstance(index.step, (Expr, Symbol)):
                for symbol_ in list(index.step.free_symbols):
                    obj.symbol_in_index[symbol_.name] = symbol_
        elif isinstance(index, Expr):
            for symbol_ in list(index.free_symbols):
                obj.symbol_in_index[symbol_.name] = symbol_
        elif isinstance(index, list):
            for ele in index:
                if isinstance(ele, (Expr, Symbol)):
                    for symbol_ in list(ele.free_symbols):
                        obj.symbol_in_index[symbol_.name] = symbol_

        return obj

    def _numpycode(self, printer, **kwargs):
        if isinstance(self.index, idx):
            temp = self.symbol.name + '[ix_({i})]'.format(i=printer._print(self.index))
            return temp
        else:
            if isinstance(self.index, slice):
                return self.symbol.name + '[slice_{i}]'.format(
                    i=printer._print((self.index.start, self.index.stop + 1, self.index.step)))
            else:
                return self.symbol.name + '[{i}]'.format(i=printer._print(self.index))

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Param_(Symbol):
    _iterable = False  # sp.lambdify gets into infinite loop if _iterable == True

    def __new__(cls, name, value=None, dim=1, triggerable=False):
        obj = Symbol.__new__(cls, name)
        obj.name = name
        obj.triggerable = triggerable
        if value is not None:
            if isinstance(value, csc_array):
                obj.value = value
            else:
                temp_value = np.array(value)
                if temp_value.ndim > 1 and temp_value.size > 1:
                    obj.value = csc_array(temp_value)
                else:
                    obj.value = temp_value.reshape((-1, 1))
        else:
            obj.value = None
        obj.dim = dim
        obj.initialized = True if value is not None else False
        return obj

    def __getitem__(self, item):
        if isinstance(item, (idx, int, slice)):
            return IdxParam(self, item, self.dim)
        elif isinstance(item, tuple):
            if all([isinstance(item_, (idx, Integer, int, slice)) for item_ in list(item)]):
                return IdxParam(self, item, self.dim)


class IdxConst(Symbol):

    def __new__(cls, symbol, index, dim):
        if not isinstance(index, (idx, int, slice, tuple)):
            raise TypeError(f"Unsupported idx type {type(index)}")
        if isinstance(index, idx):
            name = f'{symbol.name}[{index.__str__()}]'
        elif isinstance(index, int):
            name = f'{symbol.name}[{index}]'
        elif isinstance(index, slice):
            name = f'{symbol.name}[{print_slice(index)}]'
        elif isinstance(index, tuple):
            temp = []
            for idx_ in list(index):
                if isinstance(idx_, idx):
                    temp += [idx_.__str__()]
                elif isinstance(idx_, int):
                    temp += [str(idx_)]
                elif isinstance(idx_, slice):
                    temp += [str(print_slice(idx_))]
            temp = ', '.join(element for element in temp)
            name = f'{symbol.name}[{temp}]'
        else:
            raise TypeError(f"Unsupported index type {type(index)}")

        obj = Symbol.__new__(cls, name)
        obj.symbol = symbol
        obj.index = index
        obj.symbol_name = symbol.name
        obj.dim = dim

        # in case the index is Expression, for example in PDE we have variable
        # like x[1:M-1] where
        obj.symbol_in_index = dict()  # Dict[str, Symbol]
        if isinstance(index, idx):
            obj.symbol_in_index[index.name] = index
        elif isinstance(index, slice):
            if isinstance(index.start, (Expr, Symbol)):
                for symbol_ in list(index.start.free_symbols):
                    obj.symbol_in_index[symbol_.name] = symbol_
            if isinstance(index.stop, (Expr, Symbol)):
                for symbol_ in list(index.stop.free_symbols):
                    obj.symbol_in_index[symbol_.name] = symbol_
            if isinstance(index.step, (Expr, Symbol)):
                for symbol_ in list(index.step.free_symbols):
                    obj.symbol_in_index[symbol_.name] = symbol_
        elif isinstance(index, Expr):
            for symbol_ in list(index.free_symbols):
                obj.symbol_in_index[symbol_.name] = symbol_
        elif isinstance(index, list):
            for ele in index:
                if isinstance(ele, (Expr, Symbol)):
                    for symbol_ in list(ele.free_symbols):
                        obj.symbol_in_index[symbol_.name] = symbol_

        return obj

    def _numpycode(self, printer, **kwargs):
        if isinstance(self.index, slice):
            return self.symbol.name + '[slice_{i}]'.format(
                i=printer._print((self.index.start, self.index.stop + 1, self.index.step)))
        else:
            return self.symbol.name + '[{i}]'.format(i=printer._print(self.index))

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Const_(Symbol):
    _iterable = False  # sp.lambdify gets into infinite loop if _iterable == True

    def __new__(cls, name, value=None, dim=1):
        obj = Symbol.__new__(cls, name)
        obj.name = name
        if value is not None:
            temp_value = np.array(value)
            if temp_value.ndim > 1 and temp_value.size > 1:
                obj.value = csc_array(temp_value)
            else:
                obj.value = temp_value.reshape((-1, 1))
        else:
            obj.value = None
        obj.dim = dim
        obj.initialized = True if value is not None else False
        return obj

    def __getitem__(self, item):
        if isinstance(item, (idx, int, slice)):
            return IdxConst(self, item, self.dim)
        elif isinstance(item, tuple):
            if all([isinstance(item_, (idx, Integer, int, slice)) for item_ in list(item)]):
                return IdxConst(self, item, self.dim)


class IdxParam(IdxConst):
    pass


class Set(Symbol):

    def __new__(cls, name, value):
        obj = Symbol.__new__(cls, name)
        obj.name = name
        obj.value = value
        return obj

    def __getitem__(self, item):
        if isinstance(item, idx):
            return IdxVar(self, item)
        else:
            return self.value[item]

    def __mul__(self, other):
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


class StateVar(Symbol):

    def __new__(cls, name):
        return Symbol.__new__(cls, name)


class AlgebraVar(Symbol):

    def __new__(cls, name):
        return Symbol.__new__(cls, name)


X = StateVar('X')
Y = AlgebraVar('Y')


class AliasVar(Symbol):

    def __new__(cls, sym: Union[StateVar, AlgebraVar], suffix: str):
        obj = Symbol.__new__(cls, sym.name + suffix)
        obj.alias_of = sym.name
        obj.suffix = suffix
        return obj


class ComputeParam(Symbol):

    def __new__(cls, name):
        obj = Symbol.__new__(cls, name)
        return obj


class F(Function):

    def __new__(cls, *args, **kwargs):
        obj = Function.__new__(cls, *args, **kwargs)
        return obj


class G(Function):

    def __new__(cls, *args, **kwargs):
        obj = Function.__new__(cls, *args, **kwargs)
        return obj


def new_symbols(names, commutative: bool):
    if commutative:
        return symbols(names, real=True)
    else:
        return symbols(names, commutative=commutative)


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
