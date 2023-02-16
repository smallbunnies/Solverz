from sympy import Symbol, Function, Mul, sign, Expr, symbols
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


class MyExpr(Expr):

    def _eval_nseries(self, x, n, logx, cdir):
        pass

    def __new__(cls, *args, **kwargs):
        pass


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
            return Mul(Diagonal(Mul(*self.args[argindex:])), *self.args[0:argindex - 1])
        else:
            return Mul(*self.args[0:argindex - 1])


@implements_sympify('Diagonal')
class Diagonal(Function):
    is_commutative = False

    def fdiff(self, argindex=1):
        return 1  # the 1 also means Identity matrix here


@implements_sympify('Abs')
class Abs(Function):
    is_commutative = False

    def fdiff(self, argindex=1):
        return Diagonal(sign(*self.args))  # the 1 also means Identity matrix here


@implements_sympify('sign')
class sign(Function):
    is_commutative = False


class SolVar(Function):
    pass
