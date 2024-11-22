(advanced)=

# Advanced Usage

## Writing Custom Functions
Sometimes one may have functions that go beyond the Solverz built-in library. This guide will describe how to create 
such custom functions using Solverz and inform Solverz of their paths, so that the functions can be incorporated into 
numerical simulations. 

```{note}
Alternatively, one can directly contribute to the [SolMuseum](https://solmuseum.solverz.org/stable/) 
library so that 1) others can utilize your models/functions and 2) one avoids configuring the module paths.
```

```{note}
The philosophy of function customization comes from Sympy, it helps to learn the [Sympy basics](https://docs.sympy.org/latest/index.html) 
and read the [Sympy tutorial of custom functions](https://docs.sympy.org/latest/guides/custom-functions.html) for an overview.
```

```{note}
In Solverz, the numerical computations are mainly dependent on the prevailing numerical libraries such as numpy and scipy. 
It is recommended that one first gets familiar with the [numpy](https://numpy.org/doc/stable/user/index.html) and 
[scipy](https://docs.scipy.org/doc/scipy/tutorial/index.html).
```

### An Illustrative Example

As a motivating example for this document, let's create a custom function class representing the $\min$ function.
The $\min$ function is typical in controllers of many industrial applications, which can be defined by

```{math}
\min(x,y)=
\begin{cases}
x&x\leq y\\
y& x>y
\end{cases}.
```

To incorporate $\min$ in our simulation modelling, its symbolic and numerical implementations shall be defined. 
Specifically,

1. a symbolic function `min` can be called to represent the $\min$ function;
2. the symbolic derivatives of `min` are automatically derived for the Jacobian block parser;
3. the numerical interfaces are defined so that the `min` function and its derivatives can be correctly evaluated.

First, we define the numerical interfaces. The derivatives of $\min$ function are

```{math}
\pdv{\min(x,y)}{x}=
\begin{cases}
1&x\leq y\\
0& x>y
\end{cases},\quad
\pdv{\min(x,y)}{y}=
\begin{cases}
0&x\leq y\\
1& x>y
\end{cases}.
```

Let us create a `myfunc` directory and put the numerical codes in the `myfunc.py` file that looks like

```python
# myfunc.py
import numpy as np
from numba import njit


@njit(cache=True)
def Min(x, y):
    x = np.asarray(x).reshape((-1,))
    y = np.asarray(y).reshape((-1,))

    z = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] <= y[i]:
            z[i] = x[i]
        else:
            z[i] = y[i]
    return z


@njit(cache=True)
def dMindx(x, y):
    x = np.asarray(x).reshape((-1,))
    y = np.asarray(y).reshape((-1,))

    z = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] <= y[i]:
            z[i] = 1
        else:
            z[i] = 0
    return z


@njit(cache=True)
def dMindy(x, y):
    return 1-dMindx(x, y)
```

In `myfunc.py`, we use `Min` to avoid conflicting with the built-in `min` function. 
The `@njit(cache)` decorator is used to perform the jit-compilation and hence speed up the numerical codes.

Then let us install the `myfunc` module, so that Solverz can import the `myfunc` module. Use the terminal to switch
to the `myfunc` module directory. Add a `pyproject.toml` file there. 

```{note}
One can clone the `pyproject.toml` file from [example repo](https://github.com/smallbunnies/myfunc).
```

Use the following command to install the module.

```shell
pip install -e .
```

```{note}
Because `myfunc` module is installed in the editable mode, one can change the numerical implementations in `myfunc.py`
with great freedom.
```

As for the symbolic implementation, let us start by creating a `Min.py` file and subclassing `MulVarFunc` there with

```python
from Solverz import MulVarFunc

class Min(MulVarFunc):
    pass

class dMindx(MulVarFunc):
    pass

class dMindy(MulVarFunc):
    pass
```

The `MulVarFunc` is the base class of symbolic multi-variate functions in Solverz. 

At this point, `Min` and its derivatives have no behaviors defined on it. To instruct Solverz in the differentiation 
rule of `Min` and the numerical implementations, we shall add following lines
```python
class Min(MulVarFunc):
    arglength = 2

    def fdiff(self, argindex=1):
        if argindex == 1:
            return dMindx(*self.args)
        elif argindex == 2:
            return dMindy(*self.args)

    def _numpycode(self, printer, **kwargs):
        return (f'myfunc.Min' + r'(' +
                ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')')


class dMindx(MulVarFunc):
    arglength = 2

    def _numpycode(self, printer, **kwargs):
        return (f'myfunc.dMindx' + r'(' +
                ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')')


class dMindy(MulVarFunc):
    arglength = 2

    def _numpycode(self, printer, **kwargs):
        return (f'myfunc.dMindy' + r'(' +
                ', '.join([printer._print(arg, **kwargs) for arg in self.args]) + r')')

```

where the `fdiff` function should return the derivative of the function, without considering the chain rule, 
with respect to the `argindex`-th variable; the `_numpycode` functions define the numerical implementations of the 
functions. Since the `myfunc` module has been installed, the numerical implementations can be called by
`myfunc.Min`.

After finish the above procedures, we can finally use the `Min` function in our simulation modelling. An example is as 
follows.

```python
from Solverz import Model, Var, Eqn, made_numerical
from Min import Min

m = Model()
m.x = Var('x', [1, 2])
m.y = Var('y', [3, 4])
m.f = Eqn('f', Min(m.x, m.y))
sae, y0 = m.create_instance()
ae = made_numerical(sae, y0, sparse=True)
```

We will have the output

```shell
>>> ae.F(y0, ae.p)
array([1.0, 2.0])
```
