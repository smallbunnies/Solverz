from typing import Union, Type, Dict, Callable

from sympy import Symbol, Expr, Add, Mul, Number, sin, Derivative, Pow, cos, Function, Integer

dfunc_mapping: Dict[Type[Expr], Callable] = {}

__all__ = ['Index', 'Slice', 'DT', 'dtify', 'dLinspace', 'dConv', 'dDelta']


class Index(Symbol):

    def __new__(cls, name: str, sequence: int = -1, commutative=False):
        if sequence < 0:
            obj = Symbol.__new__(cls, f'{name}')
        else:
            obj = Symbol.__new__(cls, f'{name}{sequence}')
        obj.sequence = sequence
        obj.is_Integer = True
        if sequence < 0:
            obj.name = f'{name}'
        else:
            obj.name = f'{name}{sequence}'
        return obj


class Slice(Symbol):

    def __new__(cls, start: Union[int, Index, Expr], end: Union[int, Index, Expr], commutative=False):
        if isinstance(start, (Index, Expr)):
            if not all([isinstance(arg, (Number, Index)) for arg in start.free_symbols]):
                raise TypeError('Unsupported slice start!')
        else:
            if start < 0:
                raise ValueError('Slice start should > 0!')
        if isinstance(end, (Index, Expr)):
            if not all([isinstance(arg, (Number, Index)) for arg in end.free_symbols]):
                raise TypeError('Unsupported slice end!')
        else:
            if end < 0:
                raise ValueError('Slice end should > 0!')
        obj = Symbol.__new__(cls, f'{start}:{end}', commutative=commutative)
        obj.start = start
        obj.end = end
        obj.name = f'{start}:{end}'
        return obj


class DT(Symbol):
    __slots__ = ('index', 'name', 'symbol_name')

    def __new__(cls, symbol_name, index: Union[int, Index, Slice], commutative=False):
        if isinstance(index, int) and index < 0:
            raise IndexError("Invalid DT order")
        obj = Symbol.__new__(cls, f'{symbol_name}[{index}]', commutative=commutative)
        obj.index = index
        obj.symbol_name = symbol_name
        obj.name = f'{symbol_name}[{index}]'
        return obj


def dtify(Node: Expr, k: [Index, Slice, Type[Expr], int]):
    """
    replace sympy operator by DT operators/symbols
    :param Node:
    :param k:
    :return:
    """
    if isinstance(Node, tuple(dfunc_mapping.keys())):
        return dfunc_mapping[Node.func](k, *Node.args)
    elif isinstance(Node, Symbol):
        if Node.name != 't':
            return DT(Node.name.upper(), k)
        else:  # dt of t
            if isinstance(k, (Expr, Index)) and not isinstance(k, Slice):
                if all([isinstance(symbol, Index) for symbol in k.free_symbols]) is False:
                    raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
                else:
                    return DT('T', k)
            elif isinstance(k, Slice):
                return DT('T', k)
            else:  # integer
                if k < 0:
                    raise ValueError("DT index must be great than zero!")
                elif k == 1:
                    return 1
                else:
                    return 0
    elif isinstance(Node, Number):
        return Node * dDelta(k)
    else:
        raise TypeError(f"Unsupported Expr {Node.func}!")


def implements_dt_algebra(sym_expr: Type[Expr]):
    """Register an DT function implementation for sympy Expr."""

    def decorator(func):
        dfunc_mapping[sym_expr] = func
        return func

    return decorator


@implements_dt_algebra(Add)
def dAdd(k: [Index, Slice, Type[Expr], int], *args):
    """

    :param k: DT index
    :param args: tuple of arguments
    :return:
    """
    return Add(*[dtify(arg, k) for arg in tuple(args)])


@implements_dt_algebra(Mul)
def dMul(k: [Index, Slice, Type[Expr], int], *args):
    """
    distinguish between scalar production and Expr/Symbol production, supports only two operands
    :param k: DT index
    :param args: tuple of arguments
    :return:
    """
    if isinstance(args[0], Number):
        # assumes that scalar Number is always in front of Symbol
        # for example a=x*3 and a=3*x all derive a=3*x
        return Mul(args[0], dtify(Mul(*args[1:]), k))
    elif all([isinstance(arg, tuple(list(dfunc_mapping.keys()) + [Symbol])) for arg in args]):
        if isinstance(k, (Expr, Index)) and not isinstance(k, Slice):
            if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
                raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
            return dConv(dtify(args[0], Slice(0, k)), dtify(Mul(*args[1:]), Slice(0, k)))
        elif isinstance(k, Slice):
            if k.start >= 1:
                temp = dConv(dtify(args[0], Slice(0, k.end - k.start)), dtify(Mul(*args[1:]), k))
                for i in range(k.start):
                    temp = temp + dtify(args[0], Slice(k.start - i, k.end - i)) * dtify(Mul(*args[1:]), i)
                return temp
            else:
                return dConv(dtify(args[0], k), dtify(Mul(*args[1:]), k))
        else:  # integer
            if k >= 1:
                return dConv(dtify(args[0], Slice(0, k)), dtify(Mul(*args[1:]), Slice(0, k)))
            else:  # k==0
                return dtify(args[0], 0) * dtify(Mul(*args[1:]), 0)


@implements_dt_algebra(sin)
def dSin(k: [Index, Slice, Type[Expr]], *args):
    """

    :param k:
    :param args:
    :return:
    """
    if len(args) > 1:
        raise ValueError(f'Sin supports one operand while {len(args)} input!')
    if isinstance(k, (Expr, Index)) and not isinstance(k, Slice):
        if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
            raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
        return dConv((k - dLinspace(0, k - 1)) / k * DT('Psi', Slice(0, k - 1)), dtify(args[0], Slice(1, k)))
    elif isinstance(k, Slice):
        if k.start >= 2:
            Psi1 = DT('Psi', Slice(0, k.end - k.start))
            Psi2 = DT('Psi', Slice(1, k.end - 1))
            return dConv((k.end - dLinspace(0, k.end - k.start)) / k.end * Psi1, dtify(args[0], k)) + \
                dConv((k.end - dLinspace(1, k.end - 1)) / k.end * Psi2, dtify(args[0], 1))
        elif k.start == 1:
            Psi = DT('Psi', Slice(0, k.end - 1))
            return dConv((k.end - dLinspace(0, k.end - 1)) / k.end * Psi, dtify(args[0], k))
        else:
            Phi = DT('Phi', 0)
            Psi = DT('Psi', Slice(0, k.end))
            return dDelta(Slice(0, k.end)) * Phi + dConv((k.end - dLinspace(0, k.end)) / k.end * Psi,
                                                         dtify(args[0], Slice(0, k.end))) * (
                    1 - dDelta(Slice(0, k.end)))
    else:  # integer
        if k > 1:
            return dConv((k - dLinspace(0, k - 1)) / k * DT('Psi', Slice(0, k - 1)), dtify(args[0], Slice(1, k)))
        elif k == 1:
            return DT('Psi', 0) * dtify(args[0], 1)
        elif k == 0:
            return DT('Phi', 0)
        else:
            raise ValueError(f'DT index must be great than zero!')


@implements_dt_algebra(cos)
def dCos(k: [Index, Slice, Type[Expr], int], *args):
    """

    :param k:
    :param args
    :return:
    """
    if len(args) > 1:
        raise ValueError(f'Sin supports one operand while {len(args)} input!')
    if isinstance(k, (Expr, Index)) and not isinstance(k, Slice):
        if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
            raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
        return -dConv(((k - dLinspace(0, k - 1)) / k) * DT('Phi', Slice(0, k - 1)),
                      dtify(args[0], Slice(1, k)))
    elif isinstance(k, Slice):
        if k.start >= 2:
            Phi1 = DT('Phi', Slice(0, k.end - k.start))
            Phi2 = DT('Phi', Slice(1, k.end - 1))
            return -dConv((k.end - dLinspace(0, k.end - k.start)) / k.end * Phi1, dtify(args[0], k)) - \
                dConv((k.end - dLinspace(1, k.end - 1)) / k.end * Phi2, dtify(args[0], 1))
        elif k.start == 1:
            Phi = DT('Phi', Slice(0, k.end - 1))
            return -dConv((k.end - dLinspace(0, k.end - 1)) / k.end * Phi, dtify(args[0], k))
        else:
            Psi = DT('Psi', Slice(0, k.end - 1))
            Phi = DT('Phi', 0)
            return dDelta(k.end) * Phi - dConv((k.end - dLinspace(0, k.end - 1)) / k.end * Psi,
                                               dtify(args[0], Slice(1, k.end))) * (1 - dDelta(k.end))
    else:  # integer
        if k > 1:
            return -dConv((k - dLinspace(0, k - 1)) / k * DT('Phi', Slice(0, k - 1)), dtify(args[0], Slice(1, k)))
        elif k == 1:
            return -DT('Phi', 0) * dtify(args[0], 1)
        elif k == 0:
            return DT('Psi', 0)
        else:
            raise ValueError(f'DT index must be great than zero!')


@implements_dt_algebra(Pow)
def dPow(k: [Index, Slice, Type[Expr], int], *args):
    """

    :param k:
    :param args
    :return:
    """
    pass


@implements_dt_algebra(Derivative)
def dDerivative(k: [Index, Type[Expr], int], *args):
    """

    :param k:
    :param args:
    :return:
    """
    return (k + 1) * dtify(args[0], k + 1)


class dDelta(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            raise ValueError("Support one argument only!")
        if isinstance(args[0], Integer):
            # Function class automatically convert int into sp.Integer.
            if args[0] == 0:
                return 1
            else:
                return 0
        # if args[0].is_integer is False:
        # allow None type Integer assumption, i.e. is_integer == None, which is the default integer assumption of
        # symbols created by sp.symbols()
        # raise TypeError("Inputs should be an integer.")


class dConv(Function):
    is_commutative = False
    is_dConv = True

    @classmethod
    def eval(cls, *args):
        pass

    def __mul__(self, other):
        args = list(self.args)
        args[-1] = Mul(args[-1] * other)
        return self.func(*args)

    def __rmul__(self, other):
        args = list(self.args)
        args[0] = Mul(other * args[0])
        return self.func(*args)

    def _eval_expand_func(self, **hints):
        # traverse to flatten dConv tree
        args = self.args
        i = -1
        for arg in args:
            i = i + 1
            if arg.has(dConv):
                if arg.is_Add:
                    x, y = arg.as_two_terms()
                    temp_args1 = list(args)
                    temp_args1[i] = x
                    temp_args2 = list(args)
                    temp_args2[i] = y
                    return dConv(*temp_args1).expand(func=True, mul=False) + dConv(*temp_args2).expand(func=True,
                                                                                                       mul=False)
                elif arg.func == dConv:
                    # extract the arguments of sub-dConv nodes
                    if i == 0:
                        return self.func(*(list(arg.args) + list(args[1:]))).expand(func=True, mul=False)
                    elif i == len(args) - 1:
                        return self.func(*(list(args[:-1]) + list(arg.args))).expand(func=True, mul=False)
                    else:
                        return self.func(*(list(args[0:i - 1]) + list(arg.args) + list(args[0:i + 1]))).expand(
                            func=True, mul=False)
                # elif arg.is_Mul:
                    # deprecated by overriding multiplication

        return self


class dLinspace(Function):
    is_commutative = False
    is_dLinspace = True

    @classmethod
    def eval(cls, *args):
        pass


# TODO : Extract k-th order terms from conv.
def nConv(a, b, *args):
    """
    numerical convolution of vectors using loop or FFT
    :param a:
    :param b:
    :param args:
    :return:
    """
    pass
