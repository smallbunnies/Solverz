from functools import reduce

from sympy import Symbol, Function, Number, S, Integer, sin as Symsin, cos as Symcos
from sympy.core.function import ArgumentIndexError

from Solverz.variable.ssymbol import sSym2Sym


# %%
def VarParser(cls):
    # To convert non-sympy symbols to sympy symbols
    original_new = cls.__new__

    def new_new(cls, *args, **options):
        # check arg types and do the conversions
        args = [sSym2Sym(arg) for arg in args]
        return original_new(cls, *args, **options)

    cls.__new__ = new_new
    return cls


# %% matrix func
@VarParser
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

    def _octave(self, printer, **kwargs):

        temp = printer._print(self.args[0])
        for arg in self.args[1:]:
            if isinstance(arg, (Symbol, Function)):
                temp += '*{operand}'.format(operand=printer._print(arg))
            else:
                temp += '*({operand})'.format(operand=printer._print(arg))
        return r'(' + temp + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Diag(MatrixFunction):

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

    def _numpycode(self, printer, **kwargs):

        return r'np.diagflat(' + printer._print(self.args[0], **kwargs) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _octave(self, printer, **kwargs):
        return r'diag(' + printer._print(self.args[0], **kwargs) + r')'


# %% Univariate func
@VarParser
class UniVarFunc(Function):
    arglength = 1
    is_real = True

    @classmethod
    def eval(cls, *args):
        if len(args) != 1:
            raise TypeError(f'{cls.name} supports {cls.arglength} operand while {len(args)} input!')

    def _numpycode(self, printer, **kwargs):
        pass

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Abs(UniVarFunc):
    r"""
    The element-wise absolute value function:

        .. math::

            \operatorname{Abs}(x)=|x|

    with derivative

        .. math::

            \frac{\mathrm{d}}{\mathrm{d}x}{\operatorname{Abs}}(x)=\operatorname{Sign}(x).

    """

    def fdiff(self, argindex=1):
        """
        Get the first derivative of the argument to Abs().
        """
        if argindex == 1:
            return Sign(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _numpycode(self, printer, **kwargs):
        return r'np.abs(' + printer._print(self.args[0]) + r')'


class exp(UniVarFunc):
    r"""
    The exponential function, $e^x$.
    """

    def fdiff(self, argindex=1):
        return exp(*self.args)

    def _numpycode(self, printer, **kwargs):
        return r'np.exp(' + printer._print(self.args[0]) + r')'


class ln(UniVarFunc):
    r"""
    The ln function, $ln(x)$.
    """

    def fdiff(self, argindex=1):
        return 1 / self.args[0]

    def _numpycode(self, printer, **kwargs):
        return r'np.log(' + printer._print(self.args[0]) + r')'


# Notice: Do not succeed sympy.sin or cos here because the args of symbolic functions are Solverz.ssymbol.
# We have to parse them first.
class sin(UniVarFunc):
    r"""
    The sine function.
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return Symcos(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _numpycode(self, printer, **kwargs):
        return r'np.sin(' + printer._print(self.args[0]) + r')'


class cos(UniVarFunc):
    r"""
    The cosine function.
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -Symsin(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _numpycode(self, printer, **kwargs):
        return r'np.cos(' + printer._print(self.args[0]) + r')'


class Sign(UniVarFunc):
    r"""
    The element-wise indication of the sign of a number

        .. math::

            \operatorname{Sign}(x)=
            \begin{cases}
            1&x> 0\\
            0& x==0\\
            -1&x< 0
            \end{cases}

    """

    def fdiff(self, argindex=1):
        # sign function should be treated as a constant.
        if argindex == 1:
            return 0
        raise ArgumentIndexError(self, argindex)

    def _numpycode(self, printer, **kwargs):
        return r'np.sign(' + printer._print(self.args[0], **kwargs) + r')'

    def _octave(self, printer, **kwargs):
        return r'np.sign(' + printer._print(self.args[0], **kwargs) + r')'


class heaviside(UniVarFunc):
    r"""
    The heaviside step function

        .. math::

            \operatorname{Heaviside}(x)=
            \begin{cases}
            1 & x >= 0\\
            0 & x < 0\\
            \end{cases}

    which should be distinguished from sympy.Heaviside
    """

    def fdiff(self, argindex=1):
        # sign function should be treated as a constant.
        if argindex == 1:
            return 0
        raise ArgumentIndexError(self, argindex)

    def _numpycode(self, printer, **kwargs):
        return r'SolCF.Heaviside(' + printer._print(self.args[0], **kwargs) + r')'

    def _octave(self, printer, **kwargs):
        return r'Heaviside(' + printer._print(self.args[0], **kwargs) + r')'


class Not(UniVarFunc):
    """
    Represents bitwise_not
    """

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '({op1})'.format(op1=printer._print(self.args[0]))

    def _numpycode(self, printer, **kwargs):
        return r'SolCF.Not(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'


# %% multi-variate func
@VarParser
class MulVarFunc(Function):
    arglength = 3
    is_real = True

    @classmethod
    def eval(cls, *args):
        if len(args) != cls.arglength:
            raise TypeError(f"{cls.__name__} takes {cls.arglength} positional arguments but {len(args)} were given!")

    def _numpycode(self, printer, **kwargs):
        return (f'{self.__class__.__name__}' + r'(' +
                ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')')

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class switch(MulVarFunc):
    def _eval_derivative(self, s):
        return switch(*[arg.diff(s) for arg in self.args[0:len(self.args) - 1]], self.args[-1])

    def _numpycode(self, printer, **kwargs):
        return r'SolCF.switch(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'


class Saturation(MulVarFunc):
    r"""
    The element-wise saturation a number

        .. math::

            \operatorname{Saturation}(v, v_\min, v_\max)=
            \begin{cases}
            v_\max&v> v_\max\\
            v& v_\min\leq v\leq v_\max\\
            v_\min&v< v_\min
            \end{cases}
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return In(*self.args)
        else:
            return Integer(0)

    def _numpycode(self, printer, **kwargs):
        return r'SolCF.Saturation(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'


class AntiWindUp(MulVarFunc):
    r"""
    For PI controller

        .. math::

            \begin{aligned}
                \dot{z}(t) & =e(t) \\
                u_{d e s}(t) & =K_p e(t)+K_i z(t)
            \end{aligned},

    we limit the integrator output by setting, in the $K_i>0$ case,

        .. math::

            \dot{z}(t)=
            \begin{cases}
                0 & \text { if } u_{\text {des }}(t) \geq u_{\max } \text { and } e(t) \geq 0 \\
                0 & \text { if } u_{\text {des }}(t) \leq u_{\min } \text { and } e(t) \leq 0 \\
                e(t) & \text { otherwise }
            \end{cases}.

    So the AntiWindUp function reads

        .. math::

            \dot{z}(t)=
            \operatorname{AntiWindUp}(u_{\text {des }}(t), u_{\min }, u_{\max }, e(t)).

    """
    arglength = 4

    @classmethod
    def eval(cls, u, umin, umax, e):
        return e * Not(Or(And(GreaterThan(u, umax),
                              GreaterThan(e, 0)),
                          And(LessThan(u, umin),
                              LessThan(e, 0))
                          ))


class Min(MulVarFunc):
    r"""
    The element-wise minimum of two numbers

    .. math::

        \begin{split}\min(x,y)=
        \begin{cases}
        x&x\leq y\\
        y& x>y
        \end{cases}\end{split}

    """
    arglength = 2

    @classmethod
    def eval(cls, x, y):
        return x * LessThan(x, y) + y * (1 - LessThan(x, y))


class In(MulVarFunc):
    """
    In(v, vmin, vmax)
        return True if vmin<=v<=vmax
    """
    arglength = 3

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})<=({op2})<=({op3}))'.format(op1=printer._print(self.args[1]),
                                                    op2=printer._print(self.args[0]),
                                                    op3=printer._print(self.args[2]))

    def _numpycode(self, printer, **kwargs):
        return r'SolCF.In(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'


class GreaterThan(MulVarFunc):
    """
    Represents > operator
    """
    arglength = 2

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})>({op2}))'.format(op1=printer._print(self.args[0]),
                                          op2=printer._print(self.args[1]))

    def _numpycode(self, printer, **kwargs):
        return r'SolCF.GreaterThan(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'


class LessThan(MulVarFunc):
    """
    Represents < operator
    """
    arglength = 2

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})<({op2}))'.format(op1=printer._print(self.args[0]),
                                          op2=printer._print(self.args[1]))

    def _numpycode(self, printer, **kwargs):
        return r'SolCF.LessThan(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'


class And(MulVarFunc):
    """
    Represents bitwise_and
    """
    arglength = 2

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})&({op2}))'.format(op1=printer._print(self.args[0]),
                                          op2=printer._print(self.args[1]))

    def _numpycode(self, printer, **kwargs):
        return r'SolCF.And(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'


class Or(MulVarFunc):
    """
    Represents bitwise_or
    """
    arglength = 2

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})|({op2}))'.format(op1=printer._print(self.args[0]),
                                          op2=printer._print(self.args[1]))

    def _numpycode(self, printer, **kwargs):
        return r'SolCF.Or(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'


# %% custom func of equation printer
class Slice(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) > 3:
            raise TypeError(f"minmod takes at most 3 positional arguments but {len(args)} were given!")


class CSC_array(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) != 1:
            raise TypeError(f"CSC_array takes 1 positional arguments but {len(args)} were given!")

    def _numpycode(self, printer, **kwargs):
        return r'scipy.sparse.csc_array(' + printer._print(self.args[0]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Arange(Function):
    """
    Solverz' arange function
    """

    @classmethod
    def eval(cls, *args):
        if any([not isinstance(arg, Integer) for arg in args]):
            raise ValueError(f"Solverz' arange object accepts only integer inputs.")
        if len(args) > 2:
            raise ValueError(f"Solverz' arange object takes 2 positional arguments but {len(args)} were given!")

    def _numpycode(self, printer, **kwargs):
        return r'np.arange(' + ', '.join([printer._print(arg) for arg in self.args]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Ones(Function):
    r"""
    The all-one vector to broadcast scalars to a vector
    For example, 2 -> 2*Ones(eqn_size)
    The derivative is d(Ones(eqn_size))/dx=0
    """

    @classmethod
    def eval(cls, x):
        if not isinstance(x, Integer):
            raise ValueError("The arg of Ones() should be integer.")

    def _eval_derivative(self, s):
        return 0

    def _numpycode(self, printer, **kwargs):
        x = self.args[0]
        return r'np.ones(' + printer._print(x, **kwargs) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class coo_array(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            raise ValueError(
                f"Solverz' coo_array object accepts only one inputs.")

    def _numpycode(self, printer, **kwargs):
        return f'sps.coo_array({printer._print(self.args[0])})'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class extend(Function):

    def _numpycode(self, printer, **kwargs):
        return f'{printer._print(self.args[0])}.extend({printer._print(self.args[1])})'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class zeros(Function):
    # print zeros(6,6) as zeros((6,6))
    # or zeros(6,) as zeros((6,))
    def _numpycode(self, printer, **kwargs):
        if len(self.args) == 2:
            temp1 = printer._print(self.args[0])
            temp2 = printer._print(self.args[1])
            return r'np.zeros((' + temp1 + ', ' + temp2 + r'))'
        elif len(self.args) == 1:
            temp = printer._print(self.args[0])
            return r'np.zeros((' + temp + ', ' + r'))'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)
