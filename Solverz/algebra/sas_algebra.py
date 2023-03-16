from typing import Union, Type, Dict, Callable

import numpy as np
from sympy import Symbol, Expr, Add, Mul, Number, sin, Derivative, Sum, cos

dfunc_mapping: Dict[Type[Expr], Callable] = {}

__all__ = ['Index', 'DT', 'traverse_for_DT']


class Index(Symbol):

    def __new__(cls, name: str, sequence: int = -1, commutative=False):
        if sequence < 0:
            obj = Symbol.__new__(cls, f'{name}')
        else:
            obj = Symbol.__new__(cls, f'{name}{sequence}')
        obj.sequence = sequence
        if sequence < 0:
            obj.name = f'{name}'
        else:
            obj.name = f'{name}{sequence}'
        return obj


class DT(Symbol):
    __slots__ = ('index', 'name')

    def __new__(cls, name, index: Union[int, Index], commutative=False):
        if isinstance(index, int) and index < 0:
            raise IndexError("Invalid DT order")
        obj = Symbol.__new__(cls, f'{name}({index})', commutative=commutative)
        obj.index = index
        obj.name = f'{name}({index})'
        return obj


def traverse_for_DT(Node: Expr, k: [Index, Type[Expr], int]):
    """
    replace sympy operator by DT operators/symbols
    :param Node:
    :param k:
    :return:
    """
    args_ = []
    if isinstance(Node, tuple(dfunc_mapping.keys())):
        return dfunc_mapping[Node.func](k, *Node.args)
    elif isinstance(Node, Symbol):
        return DT(Node.name.upper(), k)
    elif isinstance(Node, Number):
        return Node
    else:
        raise TypeError(f"Unsupported Expr {Node.func}!")


def implements_dt_algebra(sym_expr: Type[Expr]):
    """Register an DT function implementation for sympy Expr."""

    def decorator(func):
        dfunc_mapping[sym_expr] = func
        return func

    return decorator


@implements_dt_algebra(Add)
def dAdd(k: [Index, Type[Expr], int], *args):
    """

    :param k: DT index
    :param args: tuple of arguments
    :return:
    """
    return Add(*[traverse_for_DT(arg, k) for arg in tuple(args)])


@implements_dt_algebra(Mul)
def dMul(k: [Index, Type[Expr], int], *args):
    """
    distinguish between scalar production and Expr/Symbol production, supports only two operands
    :param k: DT index
    :param args: tuple of arguments
    :return:
    """
    if isinstance(args[0], Number):
        # assumes that scalar Number is always in front of Symbol
        # for example a=x*3 and a=3*x all derive a=3*x
        return Mul(args[0], traverse_for_DT(Mul(*args[1:]), k))
    elif all([isinstance(arg, tuple(list(dfunc_mapping.keys()) + [Symbol])) for arg in args]):
        if isinstance(k, (Expr, Index)):
            if all([isinstance(symbol, Index) for symbol in k.free_symbols]):
                sequence_new = np.max([symbol.sequence for symbol in k.free_symbols]) + 1
            else:
                raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
            m = Index('k', sequence=sequence_new)
            if sequence_new == 0:
                # Extract X(k)Y(0) and X(0)Y(k) from sigma(X(m)Y(k-m)) if k.sequence == -1,
                # since X(k) and Y(k) are unknown in this case.
                return Sum(Mul(traverse_for_DT(args[0], m), traverse_for_DT(Mul(*args[1:]), k - m)), (m, 1, k - 1)) + \
                    traverse_for_DT(args[0], 0) * traverse_for_DT(Mul(*args[1:]), k) + \
                    traverse_for_DT(args[0], k) * traverse_for_DT(Mul(*args[1:]), 0)
            else:
                return Sum(Mul(traverse_for_DT(args[0], m), traverse_for_DT(Mul(*args[1:]), k - m)), (m, 0, k))
        else:  # integer
            if k > 1:
                m = Index('k', sequence=0)
                return Sum(Mul(traverse_for_DT(args[0], m), traverse_for_DT(Mul(*args[1:]), k - m)), (m, 1, k - 1)) + \
                    traverse_for_DT(args[0], 0) * traverse_for_DT(Mul(*args[1:]), k) + \
                    traverse_for_DT(args[0], k) * traverse_for_DT(Mul(*args[1:]), 0)
            else:  # k==0
                return traverse_for_DT(args[0], 0) * traverse_for_DT(Mul(*args[1:]), 0)


@implements_dt_algebra(sin)
def dSin(k: [Index, Type[Expr]], arg):
    """

    :param k:
    :param arg:
    :return:
    """
    pass


@implements_dt_algebra(cos)
def dCos(k: [Index, Type[Expr], int], arg):
    """

    :param k:
    :param arg:
    :return:
    """
    pass


@implements_dt_algebra(Derivative)
def dDerivative(k: [Index, Type[Expr], int], arg):
    """

    :param k:
    :param arg:
    :return:
    """
    pass
