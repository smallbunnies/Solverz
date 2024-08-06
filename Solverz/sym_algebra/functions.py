from functools import reduce

from sympy import Symbol, Function, Number, S, Integer, sin as Symsin, cos as Symcos
from sympy.core.function import ArgumentIndexError

from Solverz.variable.ssymbol import sSym2Sym


# %% miscellaneous
class F(Function):
    """
    For the usage of denoting the function being differentiated in EqnDiff object only
    """
    pass


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

        return r'diagflat(' + printer._print(self.args[0], **kwargs) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _octave(self, printer, **kwargs):
        return r'diag(' + printer._print(self.args[0], **kwargs) + r')'


# %% Univariate func
@VarParser
class UniVarFunc(Function):
    @classmethod
    def eval(cls, *args):
        if len(args) != 1:
            raise TypeError(f'Supports one operand while {len(args)} input!')


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
        return r'abs(' + printer._print(self.args[0]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class exp(UniVarFunc):
    r"""
    The exponential function, $e^x$.
    """

    def fdiff(self, argindex=1):
        return exp(*self.args)

    def _numpycode(self, printer, **kwargs):
        return r'exp(' + printer._print(self.args[0]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class ln(UniVarFunc):
    r"""
    The ln function, $ln(x)$.
    """

    def fdiff(self, argindex=1):
        return 1 / self.args[0]

    def _numpycode(self, printer, **kwargs):
        return r'log(' + printer._print(self.args[0]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


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
        return r'sin(' + printer._print(self.args[0]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


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
        return r'cos(' + printer._print(self.args[0]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


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
        return r'sign(' + printer._print(self.args[0], **kwargs) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _octave(self, printer, **kwargs):
        return r'sign(' + printer._print(self.args[0], **kwargs) + r')'


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
        return r'Heaviside(' + printer._print(self.args[0], **kwargs) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _octave(self, printer, **kwargs):
        return r'Heaviside(' + printer._print(self.args[0], **kwargs) + r')'


# %% multi-variate func
@VarParser
class MulVarFunc(Function):
    pass


class minmod_flag(MulVarFunc):
    """
    Different from `minmod`, minmod function outputs the position of args instead of the values of args.
    """

    @classmethod
    def eval(cls, *args):
        if len(args) != 3:
            raise TypeError(f"minmod takes 3 positional arguments but {len(args)} were given!")


class switch(MulVarFunc):
    def _eval_derivative(self, s):
        return switch(*[arg.diff(s) for arg in self.args[0:len(self.args) - 1]], self.args[-1])

    def _numpycode(self, printer, **kwargs):
        return r'switch(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


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

    @classmethod
    def eval(cls, v, vmin, vmax):
        return v * In(v, vmin, vmax) + vmax * GreaterThan(v, vmax) + vmin * LessThan(v, vmin)


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

    @classmethod
    def eval(cls, x, y):
        return x * LessThan(x, y) + y * (1 - LessThan(x, y))


class In(MulVarFunc):
    """
    In(v, vmin, vmax)
        return True if vmin<=v<=vmax
    """

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})<=({op2})<=({op3}))'.format(op1=printer._print(self.args[1]),
                                                    op2=printer._print(self.args[0]),
                                                    op3=printer._print(self.args[2]))

    def _numpycode(self, printer, **kwargs):
        return r'SolIn(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class GreaterThan(MulVarFunc):
    """
    Represents > operator
    """

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})>({op2}))'.format(op1=printer._print(self.args[0]),
                                          op2=printer._print(self.args[1]))

    def _numpycode(self, printer, **kwargs):
        return r'SolGreaterThan(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class LessThan(MulVarFunc):
    """
    Represents < operator
    """

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})<({op2}))'.format(op1=printer._print(self.args[0]),
                                          op2=printer._print(self.args[1]))

    def _numpycode(self, printer, **kwargs):
        return r'SolLessThan(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class And(MulVarFunc):
    """
    Represents bitwise_and
    """

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})&({op2}))'.format(op1=printer._print(self.args[0]),
                                          op2=printer._print(self.args[1]))

    def _numpycode(self, printer, **kwargs):
        return r'And(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Or(MulVarFunc):
    """
    Represents bitwise_or
    """

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})|({op2}))'.format(op1=printer._print(self.args[0]),
                                          op2=printer._print(self.args[1]))

    def _numpycode(self, printer, **kwargs):
        return r'Or(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Not(MulVarFunc):
    """
    Represents bitwise_not
    """

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '({op1})'.format(op1=printer._print(self.args[0]))

    def _numpycode(self, printer, **kwargs):
        return r'Not(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


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
        return r'csc_array(' + printer._print(self.args[0]) + r')'

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
        return r'arange(' + ', '.join([printer._print(arg) for arg in self.args]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)
