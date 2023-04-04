from typing import Union, Type, Dict, Callable

from sympy import Symbol, Expr, Add, Mul, Number, sin, Derivative, Pow, cos, Function, Integer

# __all__ = ['dAdd', 'dMul', 'dSin', 'dCos', 'dPow', 'dDerivative', 'dConv_s', 'dLinspace', 'dtify', 'dDelta']

dfunc_mapping: Dict[Type[Expr], Callable] = {}


class Index(Symbol):
    r"""
    The Index of DT.

    For example, the $k$-th order DT of $x(t)$ is $X(k)$.

    """
    is_Integer = True

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
    r"""
    The Slice of DT, which is a vector.

    Examples
    ========

    Suppose the start is $k_0$ and the end is $k_1$, then $X[k_0:k_1]$ denotes the vector

    .. math::

        \begin{bmatrix}
        X[k_0]&\cdots &X[k_1]
        \end{bmatrix}

    Parameters
    ==========

    start : start of DT slice
    end   : start of DT slice

    """

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
    """
    The DT object
    """
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
    Replace sympy operator by DT operators/symbols

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
class dAdd(Function):
    r"""
    This function returns DT of addition.

    For expression $\sum_{i}x_i(t)$,

    if the order of DT is an `Index` object `$k$`, it returns

    $$\sum_{i}x_i[k]$$

    Else if the order of DT is a `Slice` object `$k_0:k_1$`, it returns

    .. math::

        \sum_{i}x_i[k_0:k_1]
    """

    @classmethod
    def eval(cls, k, *args):
        if isinstance(k, (Index, Slice, Expr, Integer)):
            if isinstance(k, Expr) and not isinstance(k, (Slice, Integer)):
                # in sympy, Integer objects are instances of Expr
                if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
                    raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
            return Add(*[dtify(arg, k) for arg in tuple(args)])
        else:
            raise ValueError('Non DT index input!')


@implements_dt_algebra(Mul)
class dMul(Function):
    r"""
    This function returns the DT of multiplications.

    For expression $\prod_{i=1}^nx_i$,

    if the order of DT is an `Index` object `$k$`, it returns

    $$\bigotimes_{i=1}^n x_i[0:k]$$

    Else if the order of DT is a `Slice` object `$k_0:k_1$`, it returns

    .. math::

        \begin{cases}
          \overline{\bigotimes}_{i=1}^nx_i[k_0:k_1] & k_0=0 \\
          x_1[0:k_1-k_0]\overline{\bigotimes} y[k_0:k_1]+\sum_{i=0}^{k_0-1}x_1[k_0-i:k_1-i]*y[i] & k_0\geq 1
        \end{cases}

    where  $y(t)=\prod_{i=2}^nx_i(t)$.

    .. note::

        The convolution of numbers and vectors equals the dot product, so we always extract numbers from
        $\prod_{i=1}^nx_i(t)$ first. That is, if $x_1(t)$ is a `Number` object, the `dMul` function will return
        x1*dMul(k, x2*x3*...*xn).

    Explanation
    ===========

    # TODO: the explanation to dMul

    """

    @classmethod
    def eval(cls, k, *args):
        if isinstance(args[0], Number):
            # The scalar Number is always in front of Symbol
            # for example a=x*3 and a=3*x all derive a=3*x
            return Mul(args[0], dtify(Mul(*args[1:]), k))
        else:
            if isinstance(k, (Expr, Index)) and not isinstance(k, (Integer, Slice)):
                if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
                    raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
                return dConv_s(*[dtify(arg, Slice(0, k)) for arg in tuple(args)])
            elif isinstance(k, Slice):
                # Note that k = k.start:k.end here
                if k.start >= 1:
                    temp = dConv_v(dtify(args[0], Slice(0, k.end - k.start)), dtify(Mul(*args[1:]), k))
                    for i in range(k.start):
                        temp = temp + dtify(args[0], Slice(k.start - i, k.end - i)) * dtify(Mul(*args[1:]), i)
                    return temp
                else:
                    return dConv_v(dtify(args[0], k), dtify(Mul(*args[1:]), k))
            else:  # integer
                if k >= 1:
                    return dConv_s(*[dtify(arg, Slice(0, k)) for arg in tuple(args)])
                else:  # k==0
                    return Mul(*[dtify(arg, 0) for arg in tuple(args)])


@implements_dt_algebra(sin)
class dSin(Function):
    r"""
    This function returns DT of $\phi(t)=\sin(x(t))$, which is denoted by $\Phi[k]$.

    For expression $\phi(t)=\sin(x(t))$,

    if the order of DT is an `Index` object $k$, it returns

    .. math::
        \begin{cases}
            \sum_{m=0}^{k-1} \frac{k-m}{k}\Psi[m]x[k-m]=\left(\frac{k-(0:k-1)}{k}*\Psi[0:k-1]\right)\otimes x[1:k]& k\geq 2\\
            \Psi[0]* x[1]& k= 1\\
            \Phi[0]& k=0
        \end{cases}

    Else if the order of DT is a `Slice` object $k_0:k_1$, it returns

    .. math::

        \begin{cases}
          \left(\frac{k_1-(0:k_1-k_0)}{k_1}*\Psi[0:k_1-k_0]\right)\overline{\otimes}
          x[k_0:k_1]+\sum_{i=0}^{k_0-1}\left(\frac{k_1-(k_0-i:k_1-i)}{k_1}*\Psi[k_0-i:k_1-i]\right)* x[i] & k_0\geq 1 \\
          \left(\left(\frac{k_1-(0:k_1)}{k_1}*\Psi[0:k_1]\right)\overline{\otimes} x[0:k_1]\right)*(1-\delta[0:k_1])+\Phi[0]*\delta[0:k_1] & k_0=0
        \end{cases}

    Explanation
    ===========

    $\Phi[0]$ can not be rewritten as

    .. math::

        \begin{aligned}
        \Phi[k]=&\left(\frac{k-(0:k-1)}{k}*\Psi[0:k-1]\right)\otimes x[1:k]\\
               =&\left(\frac{k-(0:k)}{k}*\Psi[0:k]\right)\otimes x[0:k]\quad (k\geq 1).
        \end{aligned}

    See Also
    ========

    dSin

    """

    @classmethod
    def eval(cls, k, *args):
        if len(args) > 1:
            raise ValueError(f'Sin supports one operand while {len(args)} input!')
        if isinstance(k, (Expr, Index)) and not isinstance(k, (Integer, Slice)):
            if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
                raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
            return dConv_s((k - dLinspace(0, k - 1)) / k * DT('Psi', Slice(0, k - 1)),
                           dtify(args[0], Slice(1, k)))
        elif isinstance(k, Slice):
            if k.start >= 1:
                Psi1 = DT('Psi', Slice(0, k.end - k.start))
                temp = dConv_v((k.end - dLinspace(0, k.end - k.start)) / k.end * Psi1, dtify(args[0], k))
                for i in range(k.start):
                    temp = temp + (k.end - dLinspace(k.start - i, k.end - i)) / k.end * \
                           DT('Psi', Slice(k.start - i, k.end - i)) * dtify(args[0], i)
                return temp
            else:
                Phi = DT('Phi', 0)
                Psi = DT('Psi', Slice(0, k.end))
                return dDelta(Slice(0, k.end)) * Phi + \
                    dConv_v((k.end - dLinspace(0, k.end)) / k.end * Psi,
                            dtify(args[0], Slice(0, k.end))) * (1 - dDelta(Slice(0, k.end)))
        else:  # integer
            if k > 1:
                return dConv_s((k - dLinspace(0, k - 1)) / k * DT('Psi', Slice(0, k - 1)), dtify(args[0], Slice(1, k)))
            elif k == 1:
                return DT('Psi', 0) * dtify(args[0], 1)
            elif k == 0:
                return DT('Phi', 0)
            else:
                raise ValueError(f'DT index must be great than zero!')


@implements_dt_algebra(cos)
class dCos(Function):
    r"""
    This function returns DT of $\phi(t)=\cos(x(t))$, which is denoted by $\Phi[k]$.

    For expression $\phi(t)=\cos(x(t))$,

    if the order of DT is an `Index` object $k$, it returns

    .. math::
        \begin{cases}
            -\sum_{m=0}^{k-1} \frac{k-m}{k}\Phi[m]x[k-m]=-\left(\frac{k-(0:k-1)}{k}*\Phi[0:k-1]\right)\otimes x[1:k]& k\geq 2\\
            -\Phi[0]* x[1]& k= 1\\
            \Phi(0)& k=0
        \end{cases}

    Else if the order of DT is a `Slice` object $k_0:k_1$, it returns

    .. math::

        \begin{cases}
          -\left(\frac{k_1-(0:k_1-k_0)}{k_1}*\Phi[0:k_1-k_0]\right)\overline{\otimes}
          x[k_0:k_1]-\sum_{i=0}^{k_0-1}\left(\frac{k_1-(k_0-i:k_1-i)}{k_1}*\Phi[k_0-i:k_1-i]\right)* x[i] & k_0\geq 1 \\
          -\left(\left(\frac{k_1-(0:k_1)}{k_1}*\Phi[0:k_1]\right)\overline{\otimes} x[0:k_1]\right)*(1-\delta[0:k_1])-\Psi[0]*\delta[0:k_1] & k_0=0
        \end{cases}

    See Also
    ========

    dSin

    """

    @classmethod
    def eval(cls, k, *args):
        if len(args) > 1:
            raise ValueError(f'Sin supports one operand while {len(args)} input!')
        if isinstance(k, (Expr, Index)) and not isinstance(k, (Integer, Slice)):
            if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
                raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
            return -dConv_s(((k - dLinspace(0, k - 1)) / k) * DT('Phi', Slice(0, k - 1)),
                            dtify(args[0], Slice(1, k)))
        elif isinstance(k, Slice):
            if k.start >= 1:
                Phi1 = DT('Phi', Slice(0, k.end - k.start))
                temp = -dConv_v((k.end - dLinspace(0, k.end - k.start)) / k.end * Phi1, dtify(args[0], k))
                for i in range(k.start):
                    temp = temp - (k.end - dLinspace(k.start - i, k.end - i)) / k.end * \
                           DT('Phi', Slice(k.start - i, k.end - i)) * dtify(args[0], i)
                return temp
            else:
                Phi = DT('Phi', Slice(0, k.end))
                Psi = DT('Psi', 0)
                return -dDelta(Slice(0, k.end)) * Psi - \
                    dConv_v((k.end - dLinspace(0, k.end)) / k.end * Phi,
                            dtify(args[0], Slice(0, k.end))) * (1 - dDelta(Slice(0, k.end)))
        else:  # integer
            if k > 1:
                return -dConv_s((k - dLinspace(0, k - 1)) / k * DT('Phi', Slice(0, k - 1)), dtify(args[0], Slice(1, k)))
            elif k == 1:
                return -DT('Phi', 0) * dtify(args[0], 1)
            elif k == 0:
                return DT('Psi', 0)
            else:
                raise ValueError(f'DT index must be great than zero!')


@implements_dt_algebra(Pow)
class dPow(Function):
    """
    This function returns DT of power.
    """
    pass


@implements_dt_algebra(Derivative)
class dDerivative(Function):
    r"""
    This function returns DT of $x'(t)$.

    For expression $x'(t)$, if the order of DT is an `Index` object $k$, it returns

    $$(k+1)*x[k+1]$$
    """

    @classmethod
    def eval(cls, k, *args):
        if args[1][0].name is not 't':
            raise ValueError("Support time derivative only!")
        if isinstance(k, (Index, Expr, Slice, Integer)):
            return (k + 1) * dtify(args[0], k + 1)
        else:
            # args[0] is not Index or Slice.
            raise TypeError(f"Invalid inputs type {args[0].__class__.__name__}.")


class dDelta(Function):
    r"""The Kronecker delta function, $\delta(k)$.

    Explanation
    ===========

    This function will evaluate automatically in the
    case $k$ is some integer number, that is,

    .. math::

        \delta[k]=
        \begin{cases}
          1 & k=0 \\
          0 & k\in \mathbb{N}^+
        \end{cases}

    """

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
        if not isinstance(args[0], Index) and not isinstance(args[0], Slice):
            # args[0] is not Index or Slice.
            raise TypeError(f"Invalid inputs type {args[0].__class__.__name__}.")


class dConv_s(Function):
    r"""
    The convolution of vectors, which returns vectors and will not be evaluated automatically.

    Explanation
    ===========

    The arguments of `dConv_s` are vectors $v_i\in \mathbb{R}^{n+1}\ (i=1,2,3,\cdots)$. Suppose each of $v_i$ denotes
    an $n$-th degree polynomial, `dConv_s` returns the coefficients of the $n$-th degree term of the multiplications
    of these polynomials.

    For example, let $x$ and $y$ be two vectors in $\mathbb{R}^{n+1}$, then `dConv_s(x,y)` returns

    .. math::

        \sum_{m=0}^{n}x[m]y[n-m]

    We denote by $\bigotimes$ the `dConv_s` function. So that `dConv_s(x,y)` can be written as $x[0:n]\bigotimes y[0:n]$
    or $x\bigotimes y$ for brevity.

    """

    is_commutative = False
    is_dConv = True

    def __mul__(self, other):
        args = list(self.args)
        args[-1] = Mul(args[-1] * other)
        return self.func(*args)

    def __rmul__(self, other):
        args = list(self.args)
        args[0] = Mul(other * args[0])
        return self.func(*args)

    def _eval_expand_func(self, **hints):
        """
        traverse to flatten dConv_s tree
        """

        args = self.args
        i = -1
        for arg in args:
            i = i + 1
            if arg.has(dConv_v):
                if arg.is_Add and isinstance(arg, Expr):
                    x, y = arg.as_two_terms()
                    temp_args1 = list(args)
                    temp_args1[i] = x
                    temp_args2 = list(args)
                    temp_args2[i] = y
                    return self.func(*temp_args1).expand(func=True, mul=False) +\
                        self.func(*temp_args2).expand(func=True, mul=False)
                elif arg.func == dConv_v:
                    # extract the arguments of sub-dConv_s nodes
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


class dConv_v(Function):
    r"""
    The vectorization of `dConv_s`, which returns vectors and will not be evaluated automatically.

    Explanation
    ===========

    The arguments of `dConv_v` are vectors $v_i\in \mathbb{R}^{n+1}\ (i=1,2,3,\cdots)$. Suppose each of $v_i$ denotes
    an $n$-th degree polynomial, `dConv_v` returns the coefficients of the $0$-th to $n$-th degree term of the multiplications
    of these polynomials.

    For example, let $x$ and $y$ be two vectors in $\mathbb{R}^{n+1}$, then `dConv_v(x,y)` returns

    .. math::

        \begin{bmatrix}
        x[0]y[0]&\cdots&\sum_{m=0}^{n-1}x[m]y[n-m]&\sum_{m=0}^{n}x[m]y[n-m]
        \end{bmatrix}

    We denote by $\overline{\bigotimes}$ the `dConv_v` function. So that `dConv_v(x,y)` can be written as $x\overline{\bigotimes} y$.

    """

    is_commutative = False
    is_dConv = True

    def __mul__(self, other):
        args = list(self.args)
        args[-1] = Mul(args[-1] * other)
        return self.func(*args)

    def __rmul__(self, other):
        args = list(self.args)
        args[0] = Mul(other * args[0])
        return self.func(*args)

    def _eval_expand_func(self, **hints):
        """
        traverse to flatten dConv_s tree
        """

        args = self.args
        i = -1
        for arg in args:
            i = i + 1
            if arg.has(dConv_v):
                if arg.is_Add and isinstance(arg, Expr):
                    x, y = arg.as_two_terms()
                    temp_args1 = list(args)
                    temp_args1[i] = x
                    temp_args2 = list(args)
                    temp_args2[i] = y
                    return self.func(*temp_args1).expand(func=True, mul=False) +\
                        self.func(*temp_args2).expand(func=True, mul=False)
                elif arg.func == dConv_v:
                    # extract the arguments of sub-dConv_s nodes
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
    r"""
    The symbolic linspace function, which will not be evaluated automatically.

    """
    is_commutative = False

    @classmethod
    def eval(cls, m, n):
        if isinstance(m, Slice) or isinstance(n, Slice):
            raise TypeError('Supports only integer input!')

# TODO : Extract k-th order terms from conv.
