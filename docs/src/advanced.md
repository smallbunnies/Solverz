(advanced)=

# Advanced Usage

## Writing Custom Functions
Sometimes one may have functions that go beyond the Solverz built-in library. This guide will describe how to create 
such custom functions in Solverz, so that the functions can be incorporated into numerical simulations. The philosophy 
of function customization comes from Sympy, it helps to learn the [Sympy basics](https://docs.sympy.org/latest/index.html) 
and read the [Sympy tutorial of custom functions](https://docs.sympy.org/latest/guides/custom-functions.html) for an overview.

As a motivating example for this document, let's create a custom function class representing the $\min$ function. We 
want use $\min$ to determine the smaller one of two operands, which can be defined by

```{math}
\min(x,y)=
\begin{cases}
x&x\leq y\\
y& x>y
\end{cases}
```

We also want to extend the function to vector input, that is, capable of finding the element-wise minimum of two vectors.
To summarize, we shall implement $\min$ that

1. evaluates $\min(x,y)$ correctly
2. can be derived proper derivatives with respect to $x$ and $y$.

However, it is difficult to devise the analytical derivatives of $\min$. We should perform the trick that rewrites 
$\min(x,y)$ as

```{math}
x*\operatorname{lessthan}(x,y)+y*(1-\operatorname{lessthan}(x,y)).
```

where the $\operatorname{lessthan}(x,y)$ function mathematically denotes the $\leq$ operator and returns 1 if 
$x\leq y$ else 0. Since $\operatorname{lessthan}(x,y)$ can only be either 1 or 0, the above transformation holds. 

If the derivatives of $\operatorname{lessthan}(x,y)$ with respect to any argument are zero, then we have
```{math}
\frac{\partial}{\partial x}\min(x,y)=
\operatorname{lessthan}(x,y)
```
and
```{math}
\frac{\partial}{\partial y}\min(x,y)=
1-\operatorname{lessthan}(x,y).
```
Hence, it suffices to have a custom $\operatorname{lessthan}(x,y)$ function that

1. evaluates $\operatorname{lessthan}(x,y)$ correctly
2. has zero-derivative with respect to $x$ or $y$.

Let us start by subclassing `MulVarFunc`
```python
from Solverz.sym_algebra.functions import MulVarFunc
class Min(MulVarFunc):
    pass
class LessThan(MulVarFunc):
    pass
```
The `MulVarFunc` is the base class of multi-variate functions in Solverz. 
At this point, `Min` has no behaviors defined on it. To automatically evaluate the `Min` function, we ought to define 
the **_class method_** `eval()`. `eval()` should take the arguments of the function and return the value 
$x*\operatorname{lessthan}(x,y)+y*(1-\operatorname{lessthan}(x,y))$:
```python
class Min(MulVarFunc):
    @classmethod
    def eval(cls, x, y):
        return x * LessThan(x, y) + y * (1 - LessThan(x, y))
```
```python
>>> from Solverz import Var
>>> Min(Var('x',0),Var('y',0))
... x*((x)<=(y)) + y*(1 - ((x)<=(y)))
```
To define the differentiation of  `LessThan()`, we have
```python
from sympy import Integer
class LessThan(MulVarFunc):
    """
    Represents < operator
    """

    def _eval_derivative(self, s):
        return Integer(0)

    def _sympystr(self, printer, **kwargs):
        return '(({op1})<=({op2}))'.format(op1=printer._print(self.args[0]),
                                           op2=printer._print(self.args[1]))

    def _numpycode(self, printer, **kwargs):
        return r'SolLessThan(' + ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')'

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)
```
Here, the `_sympystr()` method defines its string representation:
```python
>>> LessThan(Var('x'), Var('y'))
... ((x)<=(y))
```
The `_eval_derivative()` method forces the derivatives of `LessThan()` to be zero:
```python
from Solverz import iVar
>>> Min(Var('x',0),Var('y',0)).diff(iVar('x'))
... ((x)<=(y))
```
where `iVar` is the internal variable type of Solverz, `diff()` is the method to derive derivatives.

The `_numpycode()` function defines what should `LessThan()` be printed to  in numerical codes. Here, we define the 
`SolLessThan()` as the numerical implementation of  `LessThan()`. Given array `[0,2,-1]` and `[1,2,3]`:
```python
>>> import numpy as np
>>> SolLessThan(np.array([0, 2, -1]), np.array([1,2,3]))
... array([1, 0, 1])
```
```{note}
In Solverz, the numerical computations are mainly dependent on the prevailing numerical libraries such as numpy and scipy. 
It is recommended that one first gets familiar with the [numpy](https://numpy.org/doc/stable/user/index.html) and 
[scipy](https://docs.scipy.org/doc/scipy/tutorial/index.html).
```
The implementation of `SolLessThan()` should be put in the `Solverz.num_api.custom_function` module:
```python
@implements_nfunc('SolLessThan')
@njit(cache=True)
def SolLessThan(x, y):
    x = np.asarray(x).reshape((-1,))
    return (x < y).astype(np.int32)
```
The `implements_nfunc()` cannot be omitted and the `njit()` decorator enables the numba-based dynamic compilation for efficiency.

