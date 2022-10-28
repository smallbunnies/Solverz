from sympy import Function, Mul, sign

Sympify_Mapping = {}


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
