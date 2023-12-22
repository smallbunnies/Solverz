from functools import reduce

from sympy import Symbol, Function, Number, S, sin, cos, Integer
from sympy import exp as spexp
from sympy.core.function import ArgumentIndexError


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


class Abs(Function):

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


class exp(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) != 1:
            raise TypeError(f'Supports one operand while {len(args)} input!')

    def fdiff(self, argindex=1):
        return exp(*self.args)

    def _numpycode(self, printer, **kwargs):
        return r'exp(' + printer._print(self.args[0]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


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

    def _numpycode(self, printer, **kwargs):

        return r'sign(' + printer._print(self.args[0], **kwargs) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

class zeros(Function):
    # print zeros(6,6) as zeros((6,6))
    # or zeros(6,) as zeros((6,))
    def _numpycode(self, printer, **kwargs):
        if len(self.args) == 2:
            temp1 = printer._print(self.args[0])
            temp2 = printer._print(self.args[1])
            return r'zeros((' + temp1 + ',' + temp2 + r'))'
        elif len(self.args) == 1:
            temp = printer._print(self.args[0])
            return r'zeros((' + temp + ',' + r'))'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class CSC_array(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) != 1:
            raise TypeError(f"CSC_array takes 1 positional arguments but {len(args)} were given!")

    def _numpycode(self, printer, **kwargs):
        return r'csc_array(' + printer._print(self.args[0]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class SolList(Function):

    @classmethod
    def eval(cls, *args):
        if any([not isinstance(arg, Integer) for arg in args]):
            raise ValueError(f"Solverz' list object accepts only integer inputs.")

    def _numpycode(self, printer, **kwargs):
        return r'[' + ','.join([printer._print(arg) for arg in self.args]) + r']'

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
        return r'arange(' + ','.join([printer._print(arg) for arg in self.args]) + r')'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)
