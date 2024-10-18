(gettingstarted)=

# Getting Started

## Overview
Solverz supports three types of equality constraints, which are respectively

1. Algebraic Equations (AEs) with general formula $0=F(y,p)$
2. Finite Difference Algebraic Equations (FDAEs) with general formula $0=F(t,y,p,y_0)$
3. Differential Algebraic Equations (DAEs) with general formula $M\dot{y}=F(t,y,p)$

where $y$ is the known variable, $F$ is the mapping, $p$ is the parameter set of your models, $y_0$ is the previous time node value of $y$, $M$is the singular mass matrix.

If your mathematical models belong to one of the equation types above, then Solverz should be your choice, because it 
supports an object-oriented design for the definition of these equations and can generate high-performance numerical codes. 

The basic steps of a simple modeling and solution process are:

1. Create model and declare components
2. Instantiate the model and generate numerical codes
3. Apply solver
4. Interrogate solver results

These steps and the philosophy behind Solverz are explained as follows.
## Symbolic Modelling
The modelling starts with declaring an empty Model with
```python
from Solverz import Model
m = Model()
```
### Variables
Variables represent unknown or changing parts of a model. The values taken by the variables are often referred to as a solution of the simulation. 
In Solverz, variables can be added to the model by
```python
from Solverz import Var
m.x = Var(name='x', value=[0, 1])
m.y = Var(name='y', value=0)
```
We should assign two attributes to each variable, which are `name` and `value` respectively. The `name` is how variable `m.x` is represented in expressions and the `value` is the initial conditions of the `m.x`. 

Solverz supports only one-dimensional variables and the length of the variables is implicitly dependent on the length of the `value` you give.
#### Index
Sometimes we want to access certain elements in a variable, for example, the first element of `x`, and the first to third elements of `y`. Solverz provides you with the flexibility by just calling
```python
>>> m.x[0]
x[0]
>>> m.y[0:3]
y[0:3]
```
Currently, only `int` and `slice` indices are recommended. We are working on more complex index types such as the list. 
Also, be careful not to exceed the maximum index that depends on the initial value you give. Otherwise, errors would be raised.
#### Operation
The symbolic mathematics of Solverz variables are based on [Sympy](https://www.sympy.org/en/index.html). That is, we can use the Solverz variables to perform the python operations `+`, `-`, `*`, `/`, `**`, just to name a few. 
For example, the expressions $x_0x_1+\sin(x_2)$ can be implemented by
```python
>>> from Solverz import Var, sin
>>> m.x = Var(name='x', value=[0, 0, 0])
>>> m.x[0]*m.x[1]+sin(m.x[2])
x[0]*x[1] + sin(x[2])
```
Currently, functions like `sin`, `cos`, `Abs`, `Sign`, `exp`, just to name a few, are supported. You can refer to [api reference](https://doc.solverz.org/reference/index.html) for all supported functions and their detailed implementations.
If you want to write your own functions, refer to [advanced usage](https://doc.solverz.org/advanced.html). And you can contact us so that we can make it a Solverz built-in funtion.

A special case of Variables is the `AliasVar`, that is, the alias variables. In Solverz, the `AliasVar` is used to denote the historical value of some variable, which is useful in finite difference equations. 
For example, one can use the following codes to declare `AliasVar`.
```python
from Solverz import AliasVar
m.p = Var(name='p', value=0)
m.p0 = AliasVar(name='p', init=m.p)
```
The `name` of `AliasVar` is used to denote that `m.p0` is the historical value of `m.p`. And it is initialized by `m.p`.  One can initialize the variables by assigning expressions to the `init` attribute, apart from giving the initial values.
### Parameters
Parameters represents the data that must be supplied to perform the simulation. Though we can use python expression `c*m.x`, where c is a number, to model the expression $cx$. But it is recommended to declare `c` as a new parameter if you want to change $c$ in simulations for different results.
The declaration of parameters below is the same as variable declaration.
```python
from Solverz import Param
m.c = Param(name='c', value=0)
```
The parameters can also be incorporated in operations. And it can be indexed just as the variables.
In Solverz, it is easy to alter the parameters in simulation, which will be shown later.
#### Parameters changing with time
Some parameters are time-varying. For example, when simulating differential equations and the finite difference algebraic equations, you want to know the results under given boundaries. In Solverz, this is realized by using `TimeSeriesParam`.
For example, we want the parameter `pb` to increase from 10 to 11 in the time interval [1,2]. We can do the following.
```python
from Solverz import TimeSeriesParam
m.pb = TimeSeriesParam('pb',
                       v_series=[10, 10, 11, 11],
                       time_series=[0, 1, 2, 100])
```
The `v_series` and `time_series` specify the values and time nodes of the Time-series parameter. Solverz performs linear interpolation for the value of `pb` at any given $t$. If $t>100$, `pb` is assumed to be 11.
### Equations
Equations describe the constraints of variables. In Solverz, equations are classified into two categories:
The basic equations 
$0=f(y,p),$
and the ordinary differential equation (ODE)
$\dot{y}=f(y,p)$
where $p$ denotes the parameters and $y$ denotes the variables.

The basic equations are imposed by the `Eqn` objects. Below is the example of a `Eqn` declaration.
```python
from Solverz import Var, Param, Eqn, made_numerical, sin, Model, nr_method

m = Model()
m.x = Var('x', 0)
m.t = Param('t', 0.3)
m.x1 = Param('x1', -0.1)
m.f = Eqn(name='f', eqn=m.x - sin(3.14 * m.x) * m.t - m.x1)
```
The `eqn` attribute should be the right hand size of $0=f(y,p)$.

The ODEs are imposed by the `Ode` object. Below is the example of an ODE declaration.
```python
from Solverz import Model, Var, Ode

m = Model()
m.h = Var('h', 0)
m.v = Var('v', 20)
m.f1 = Ode('f1', f=m.v, diff_var=m.h)
```
The `f` attribute of `Ode` objects denote the right hand side of $\dot{y}=f(y,p)$ and the `diff_var` attribute denotes the left hand side
variable.

### Create Model Instance
After adding all the elements to the `Model` object, we can create the symbolic model and the numerical variable 
instances by calling 
```python
eqs, y0 = m.create_instance()
```
#### Symbolic Equations
The `eqs` object is the symbolic equation, which can be either `AE`, `FDAE` or `DAE`. The type depends on the equations and variables you declare. The Solverz detects the equation and variable types, and then constructs `AE`, `FDAE` or `DAE` automatically.
The object stores all the symbolic equations, the derivatives and the equation addresses.  Also, it can be used to check if the equation number equals the variable number, which is critical for a model to be solved.
#### Numerical Variables
The `y0` object is the instance of numerical variables. In Solverz, it holds the meaning of the combination of all symbolic variables `Var`. And it is used directly in simulation solutions. 
For example, if `y0` contains variable `h` and `v`. Then we can access these variables by calling `y0['h']` and `y0['v']`. 
If you want to set different values of `h` and `v`, you can use `y0['h']=np.array([1,2,3])`. It should be noted that the 
new values must have the same shape compared with the original ones. 
This is because Solverz has already allocated the addresses for the variables when creating instances.
Moreover, by overloading operators, `y0` can be used for addition, subtraction, multiplication, and division.
## From Symbolic to Numerical
### Numerical Equations
To directly use the symbolic equations for computation is too slow. Alternatively, you should use the numerical equations 
derived by Solverz, which are optimized for efficient simulation. 
Specifically, Solverz prints all the symbolic expressions to well-organized python functions based on mature 
libraries such as numpy, scipy and numba. 

To derive numerical functions, just use the `made_numerical` function as follows.
We have equation 
$0=x-t\sin(\pi x)-x_1,$
where $x$ is the known variabe, $t$ and $x_1$ are parameters.
With Solverz, one has
```python
import numpy as np
from Solverz import Var, Param, Eqn, made_numerical, sin, Model, nr_method, module_printer

m = Model()
m.x = Var('x', 0)
m.t = Param('t', 0.3)
m.x1 = Param('x1', -0.1)
m.f = Eqn('f', m.x - sin(np.pi * m.x) * m.t - m.x1)
sae, y0 = m.create_instance()
ae, code = made_numerical(sae, y0, output_code=True, sparse=True)
```
The `ae` object is a instance of the `Solverz.nAE` class. It omit all the symbolic expressions and other redundant details, which has three attributes:

1. `ae.F` that returns $F(y,p)$ if one calls `ae.F(y,p)`
2. `ae.J` that returns  the jacobian$J(y,p)$ if one calls `ae.J(y,p)`
3. `ae.p` that is a python dict object storing all the parameter values.

In the above example, one has
```python
>>> ae.F(y0, ae.p)
array([0.1])
>>> ae.J(y0, ae.p).toarray()
array([[0.0575222]])
```

One can alter the parameter values, for example to set $t=1$, by
```python
>>> ae.p['t']=np.array([1.0])
```

You can also inspect the numerical codes generated by Solverz with
```python
>>> print(code['F'])
def F_(y_, p_):
    x = y_[0:1]
    t = p_["t"]
    x1 = p_["x1"]
    _F_ = zeros((1, ))
    _F_[0:1] = -t*sin(3.14159265358979*x) - x1 + x
    return _F_
>>> print(code['J'])
def J_(y_, p_):
    x = y_[0:1]
    t = p_["t"]
    x1 = p_["x1"]
    row = []
    col = []
    data = []
    row.extend([0])
    col.extend(arange(0, 1))
    data.extend(((-3.14159265358979*t*cos(3.14159265358979*x) + 1).tolist()))
    return coo_array((data, (row,col)), (1, 1)).tocsc()
```

Similarly, `Solverz.nFDAE` and `Solverz.nDAE` are respectively the numerical equation abstraction of FDAE and DAE, with the `F` and `J` attributes being the numerical interfaces. The `Solverz.nFDAE` instance has a `nstep` attribure to denote the number of historical time steps that is required. Currently, `nstep` can only be one.

Sometimes, one wants to use the second derivative information. Since Release/0.1, Solverz is able to derive the Hessian-vector product, with formula 


```{math}
\frac{\partial }{\partial y}\left(J(y)z\right)=H(y)\otimes z
```

where the Hessian tensor

```{math}
H(y)=\left(\frac{\partial \nabla g_i(y)^\mathrm{T}}{y_j}\right)_{ij}
```

and $J(y)$ is the original Jacobian.

One can call the `HVP()` method of numerical equations after generating numerical modules with `make_hvp=True`. A successful attempt of using `HVP` to improve the robustness of AE's solution can be found in [Solverz' cookbook](https://cook.solverz.org/ae/pf/pf.html).

#### Sparse matrix
In most cases, the Jacobian is a sparse matrix in which most of the elements are zero. 
The sparse matrices can be decomposed very efficiently using a sparse solver. 
It is suggested not to derive dense Jacobians because the decomposition and storage of zero elements in RAM can be very time-consuming.

However, Solverz still provides both sparse and dense Jacobian options. 
One can speficy whether to derive sparse Jacobian by setting the `sparse` argument in `made_numerical()`.
### The Module Printer
To model very complex systems can be slow. If one wants to avoid the repeated derivations of symbolic and numerical equations, a good way is to render an independent python module for your simulation model using `module_printer`.
An illustrative example is
```python
from Solverz import Model, Var, Ode, Opt, made_numerical, Rodas, module_printer

m = Model()
m.x = Var('x', [0, 20])
m.f1 = Ode('f1', m.x[1], m.x[0])
m.f2 = Ode('f2', -9.8, m.x[1])
bball, y0 = m.create_instance()

pyprinter = module_printer(mdl=bball,
                           variables=y0,
                           name='bounceball',
                           jit=True)
pyprinter.render()
```

One can import the module to the codes by
```python
from bounceball import mdl as nbball, y as y0
```


The modules rendered are better optimized compared with the ones generated by `made_numerical`. In efficiency-demanding scenarios, we recommend you using the module printer. 
### Dynamic Compilation
The dynamic nature of Python, an interpreted-type language, brings about the efficiency decreasing compared with 
languages like C/C++, especially in numerical computation. 
To resolve this, Solverz uses [Numba](https://numba.pydata.org/) to accelerate your codes based on just-in-time (JIT) 
compilation. Specifically, Numba translates Python functions to optimized machine code at runtime using the 
industry-standard LLVM compiler library. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN.

If one wants to take advantage of the dynamic compilation, just set the `jit` arg in `module_printer` to be `True`. 
Then the models will be compiled at the first time of function evaluation. 
Though the compilation time of complex models can be of tens of minutes, the compilation results are cached locally. 
Hence, the model should be compiled only once. 
It is recommended to first set `jit=False` to debug the models and then perform dynamic compilation.
Currently, dynamic compilation is not supported in `made_numerical`.
## Solvers
Solverz provides basic solvers for the solutions of AE, FDAE and DAE. 
We are working hard on implementing more mature solvers. Please feel free to contact us if you have any good idea~

Below is an overview of the built-in solvers.

### AE solvers

1. `nr_method()` the Newton-Raphson method 
2. `continuous_nr()` the continuous Newton method, which is more robust compared with the Newton-Raphson
3. `lm()` the Levenberg-Marquardt method provided by `scipy.optimize.root`. Only dense Jacobian is allowed, which may be time-consuming.
4. `sicnm()` the semi-implicit version of continuous Newton method. It possesses both the implicit stability and explicit computation overhead, which shows both robustness and efficiency. Please make Hessian-vector product if you want to use it.

### FDAE solver
`fdae_solver()`
### DAE solvers

1. `backward_euler()` the [backward Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method).
2. `implicit_trapezoid()` the [implicit trapezoidal method](https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)).
3. `Rodas()` the stiffly accurate Rosenbrock method with adaptive step size, dense output and event detection. One can use it the same as the Ode-series solvers in Matlab. This is the most stable solver in Solverz.

The detailed usage of these solvers can be found in [api reference](https://doc.solverz.org/reference/index.html).

It also a good idea to use solvers provided by scipy and other python packages since Solverz has derived the generic 
numerical interfaces.
