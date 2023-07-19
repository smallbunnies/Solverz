from sympy import Symbol, Function, Mul, sign, Expr, symbols, Number
from sympy.core.function import ArgumentIndexError
from sympy import Abs as spAbs
import numpy as np
from typing import Union

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
        obj.value = value
        return obj


class Var_(Symbol):
    # To retain the sequence of input variables, Var_ is assumed not to be commutative by default.

    _iterable = False  # sp.lambdify gets into infinite loop if _iterable == True

    def __new__(cls, name, value=None, commutative=False):
        obj = Symbol.__new__(cls, name, commutative=commutative)
        obj.name = name
        if value is not None:
            obj.value = np.array(value).reshape((-1, 1))
        else:
            obj.value = None
        obj.dim = 1
        return obj

    def __getitem__(self, item):
        return IdxVar(self, item)


class IdxVar(Symbol):

    def __new__(cls, symbol, index):
        if not isinstance(index, (idx, int, slice)):
            raise TypeError(f"Unsupported idx type {type(index)}")
        obj = Symbol.__new__(cls, f'{symbol.name}[{index}]')
        obj.symbol = symbol
        obj.idx = index
        obj.symbol_name = symbol.name
        obj.name = f'{symbol.name}[{index}]'
        return obj

    def _numpycode(self, printer, **kwargs):
        if isinstance(self.idx, idx):
            temp = self.symbol.name + '[ix_({i})]'.format(i=printer._print(self.idx))
            return temp
        else:
            return self.symbol.name + '[{i}]'.format(i=printer._print(self.idx))

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Param_(Symbol):
    # To retain the sequence of input variables, Var_ is assumed not to be commutative by default.

    _iterable = False  # sp.lambdify gets into infinite loop if _iterable == True

    def __new__(cls, name, value=None, dim=1, commutative=False, trigger=False):
        obj = Symbol.__new__(cls, name, commutative=False)
        obj.name = name
        if value is not None:
            temp_value = np.array(value)
            if temp_value.ndim > 1:
                obj.value = temp_value
            else:
                obj.value = temp_value.reshape((-1, 1))
        obj.dim = dim

        return obj

    def __getitem__(self, item):
        if isinstance(item, idx):
            return IdxParam(self, item, self.dim)
        elif isinstance(item, tuple):
            if all([isinstance(item_, (idx, Number, slice)) for item_ in list(item)]):
                return IdxParam(self, item, self.dim)
        else:
            return self.value[item]


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
        obj.idx = index
        obj.symbol_name = symbol.name
        obj.dim = dim

        return obj

    def _numpycode(self, printer, **kwargs):
        if isinstance(self.idx, idx):
            temp = self.symbol.name + '[ix_({i})]'.format(i=printer._print(self.idx))
            return temp
        else:
            return self.symbol.name + '[{i}]'.format(i=printer._print(self.idx))

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Const_(Symbol):
    # To retain the sequence of input variables, Var_ is assumed not to be commutative by default.

    _iterable = False  # sp.lambdify gets into infinite loop if _iterable == True

    def __new__(cls, name, value=None, dim=1, commutative=False):
        obj = Symbol.__new__(cls, name, commutative=False)
        obj.name = name
        if value is not None:
            temp_value = np.array(value)
            if temp_value.ndim > 1:
                obj.value = temp_value
            else:
                obj.value = temp_value.reshape((-1, 1))
        obj.dim = dim

        return obj

    def __getitem__(self, item):
        if isinstance(item, idx):
            return IdxConst(self, item, self.dim)
        elif isinstance(item, tuple):
            if all([isinstance(item_, (idx, Number)) for item_ in list(item)]):
                return IdxConst(self, item, self.dim)
        else:
            return self.value[item]


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


class Sum_(Function):
    # no repeat name of sympy built-in func.

    @classmethod
    def eval(cls, *args):
        if len(args) != 3:
            raise TypeError(f"Sum takes 3 positional arguments but {len(args)} were given!")

    def _eval_derivative(self, s):
        # which should be the same as Add
        a = self.args
        return a[0].diff(s)

    def _numpycode(self, printer, **kwargs):

        if len(self.args[1]) > 1:
            loops = (
                'for {i}, {j} in {SET}'.format(
                    i=printer._print(self.args[1][0]),
                    j=printer._print(self.args[1][1]),
                    SET=printer._print(self.args[2])))
            return '(builtins.sum({function} {loops}))'.format(
                function=printer._print(self.args[0]),
                loops=loops)

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


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


def implements_sympify(symbolic_func: str):
    """Register an __array_function__ implementation for PSArray objects."""

    def decorator(func):
        Sympify_Mapping[symbolic_func] = func
        return func

    return decorator


@implements_sympify('Mat_Mul')
class Mat_Mul(Function):
    is_commutative = False

    def fdiff(self, argindex=1):

        if isinstance(self.args[argindex - 1], Diagonal):
            return Mul(*self.args[0:argindex - 1], Diagonal(Mul(*self.args[argindex:])))
        else:
            return Mul(*self.args[0:argindex - 1])

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

    def _numpycode(self, printer, **kwargs):

        temp = printer._print(self.args[0])
        for arg in self.args[1:]:
            temp += '@{operand}'.format(operand=printer._print(arg))
            return temp

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


@implements_sympify('Abs')
class Abs(spAbs):
    pass


@implements_sympify('Diagonal')
class Diagonal(Function):
    is_commutative = False

    def fdiff(self, argindex=1):
        return 1  # the 1 also means Identity matrix here

    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            raise TypeError(f"Diagonal takes 1 positional arguments but {len(args)} were given!")

    def _latex(self, printer, **kwargs):

        _latex_str = r'\operatorname{diag}\left (' + printer._print(self.args[0]) + r'\right )'
        if 'exp' in kwargs.keys():
            return r'\left (' + _latex_str + r'\right )^{' + kwargs['exp'] + r'}'
        else:
            return _latex_str



@implements_sympify('sign')
class sign(Function):
    is_commutative = False


def traverse_for_mul(node: Expr):
    """
    traverse the expression tree and replace sympy.Mul, if the args of which have matrices, with Mat_Mul
    :param node:
    :return:
    """

    if isinstance(node, Symbol) or isinstance(node, Number):
        return node
    elif isinstance(node, Mul):
        arg_len = len(node.args)
        args = node.args
        for i in range(arg_len):
            argi = args[i]
            if isinstance(argi, (Param_, Const_, IdxConst, IdxParam)):
                if argi.dim > 1:  # matrix symbol
                    # If A is a matrix and b is a symbol, then A * b should be Mat_Mul(A, b)
                    if i == arg_len - 1:  # argi is the last operand
                        return Mul(*args)
                    elif i == 0:  # argi is the first operand
                        return Mat_Mul(argi, traverse_for_mul(Mul(*node.args[1:])))
                    else:
                        args0 = node.args[0:i]
                        args1 = node.args[i + 1:]
                        return Mul(*args0) * Mat_Mul(argi, traverse_for_mul(Mul(*args1)))
        return Mul(*args)
    else:
        args = []
        for arg in node.args:
            args = [*args, traverse_for_mul(arg)]
        return node.func(*args)


def pre_lambdify(expr: Expr):
    r"""
    Repalce `*` multiplication between matrix and vector by `mat_mul`.

    Examples
    ========

    >>> from Solverz import Var_, Param_
    >>> from Solverz import Eqn
    >>> from Solverz.num.num_alg import pre_lambdify
    >>> fin = Param_('fin')
    >>> f = Var_('f')
    >>> A = Param_('A', dim=2)
    >>> test = Eqn(name='test',eqn=fin-A*f)
    >>> pre_lambdify(test.RHS)
    fin - Mat_Mul(A, f)

    """
    return traverse_for_mul(expr)
