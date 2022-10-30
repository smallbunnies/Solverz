import sympy as sp
import numpy as np
from core.var import *
from core.variables import TimeVars
from core.equations import AE, DAE
from core.eqn import Eqn, Ode, Pde
from core.param import Param

# solve equation dx/dt=-2(x-cos(t)) y(0)=0

x = TimeVar('x')
x.v = [1, 2, 3]
y = TimeVar('y')
y.v = [4, 5, 6]
f = Ode(name='f', e_str='-2*(x-cos(t))', diff_var='x')
g = Eqn(name='g', e_str='-2*(x-cos(t))')
# f_ = f.discretize(scheme='x-x0-dt/2*(f(x0)+f(x))')
scheme = 'x-x0-dt/2*(f(x0,t0)+f(x,t))'
dt = Param('dt')
dt.v = [0.1]
E = DAE([f, f], 'E')
E1 = AE(g, 'g')
d_f = f.discretize(scheme)[1]
E_ = DAE([f, g], 'f+g').discretize(scheme)
f_ = DAE([f], 'f_').discretize(scheme)
# TODO: Solve equation dx/dt=-2(x-cos(t)) y(0)=0
# TODO: Admit more complex schemes

x1 = TimeVars([x, y])
y = x1[0]
y.array[0:3] = [[100], [200], [300]]
x1[1] = y
