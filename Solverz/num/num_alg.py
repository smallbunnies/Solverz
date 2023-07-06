from sympy import Symbol, Function, Mul, sign, Expr, symbols, Number
from typing import Union

Sympify_Mapping = {}


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


@implements_sympify('Abs')
class Abs(Function):
    is_commutative = False

    def fdiff(self, argindex=1):
        return Diagonal(sign(*self.args))  # the 1 also means Identity matrix here


@implements_sympify('sign')
class sign(Function):
    is_commutative = False


def traverse_for_mul(node: Expr):
    """
    traverse the expression tree and replace sympy.Mul with Mat_Mul
    :param node:
    :return:
    """

    if isinstance(node, Symbol) or isinstance(node, Number):
        return node
    elif isinstance(node, Mul):
        args = []
        for arg in node.args:
            args = [*args, traverse_for_mul(arg)]
        return Mat_Mul(*args)
    else:
        args = []
        for arg in node.args:
            args = [*args, traverse_for_mul(arg)]
        return node.func(*args)
