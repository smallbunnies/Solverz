import numpy as np
from sympy import Function, sympify, Symbol, Mul, symbols, Matrix, lambdify, diag, Abs

Sympify_Mapping = {}


def implements_sympify(symbolic_func: str):
    """Register an __array_function__ implementation for PSArray objects."""

    def decorator(func):
        Sympify_Mapping[symbolic_func] = func
        return func

    return decorator


@implements_sympify('Mat_Mul')
class Mat_Mul(Function):

    def fdiff(self, argindex=1):

        if isinstance(self.args[argindex - 1], Diagonal):
            return Mul(*self.args[0:argindex - 1], Diagonal(Mul(*self.args[argindex:])))
        else:
            return Mul(*self.args[0:argindex - 1])


@implements_sympify('Diagonal')
class Diagonal(Function):

    def fdiff(self, argindex=1):
        return 1  # the 1 also means Identity matrix here


@implements_sympify('Transposes')
class Transposes(Function):
    pass


x, y, m, V, Ts = symbols('x,y, m, V, Ts', commutative=False)
# 使用locals参数指定sympify表达式中变量的namespace
# f = sympify('Mat_Mul(Diagonal(Ts),V,m)',
#             locals={'Mat_Mul': Mat_Mul, 'Diagonal': Diagonal, 'm': m,
#                     'V': V,
#                     'Ts': Ts})
# g = sympify('Mat_Mul(V,Diagonal(Touts),m)',
#             locals={'Mat_Mul': Mat_Mul, 'Diagonal': Diagonal, 'm': m,
#                     'V': V,
#                     'Touts': Symbol('Touts', commutative=False)})
# b = sympify('(g(x))', locals={'Matrix':Matrix,'Diagonal':Diagonal})

# fm = f.diff(m)
#
# f1 = lambdify([Ts, V, m], f, [Lambdify_Mapping, 'numpy'])
# fm1 = lambdify([Ts, V, m], fm, [{'Diagonal': np.diag}, 'numpy'])
# a = PSArray([1, 2, 3])
# a * a * a
# mat_multiply([1, 2, 3], [1, 2, 3])
