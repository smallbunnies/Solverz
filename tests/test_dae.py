import sympy as sp
import numpy as np
from core.var import *
from core.variables import TimeVars
from core.equations import AE, DAE
from core.eqn import Eqn, Ode, Pde
from core.param import Param
from core.solver import implicit_trapezoid_ode_nonautonomous
import matplotlib.pyplot as plt
import pandas as pd
from core.algebra import StateVar, AlgebraVar, AliasVar, ComputeParam, F, X, Y
from sympy import symbols, preorder_traversal, Expr, Symbol, Add

X0 = AliasVar(X, '0')
Y0 = AliasVar(Y, '0')
X_1 = AliasVar(X, '_1')
Y_1 = AliasVar(Y, '_1')
t = ComputeParam('t')
t0 = ComputeParam('t0')
dt = ComputeParam('dt')

scheme = X - X0 + dt / 2 * (F(X, Y, t) + F(X0, Y0, t0))
f0 = Ode(name='F', e_str='-2*(Y-cos(t))', diff_var='Y')
param, eqn = f0.discretize(scheme)

# locate F and its args

# ComputeParam 总是作为新的参数返回


# scheme = X - X0 + dt / 2 * (F(X, Y) + F(X0, Y0))

c = ComputeParam('c')
Xk1 = AliasVar(X, 'k1')
Yk1 = AliasVar(Y, 'k1')
scheme = F(X + 1 / 2 * Xk1, Y + 1 / 3 * Yk1, t + dt * c)
f1 = Ode(name='F', e_str='(Pm-D*omega)', diff_var='omega')
param0, eqn0 = f1.discretize(scheme)
param1, eqn1 = f1.discretize(scheme, param={'D': Param('D'),
                                            'Pm': Param('Pm')})
param2, eqn2 = f1.discretize(scheme, extra_diff_var=['D'],
                             param={'Pm': Param('Pm')})


# ComputeParam must be new Param

# scheme = k2- F(Y0+dt*1/2*k2, t0+1/4*dt)
# f1 = Ode(name='f1', e_str='X+Y+z', diff_var='X')
