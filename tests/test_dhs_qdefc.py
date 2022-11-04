import sympy as sp
import numpy as np
from core.var import *
from core.equations import Equations
from core.eqn import Eqn
from core.algebra import StateVar, AlgebraVar, AliasVar, ComputeParam, F, G

# T = sp.Function('T')
# dx, dt, X, t, Tn1k1, Tn0k1, Tn1k0 = sp.symbols('dx, dt, X, t, Tn1k1, Tn0k1, Tn1k0')
# pde = sp.sympify('T(X,t).diff(t)+v*T(X,t).diff(X)+T(X,t)+b*T(X,t).diff(X)')
# pde = pde.subs({T(X, t).diff(t): (Tn1k1 - Tn1k0) / dt,
#                 T(X, t).diff(X): (Tn1k1 - Tn0k1) / dx,
#                 T(X, t): Tn1k1},
#                simultaneous=True)
#
# sp.collect(pde.expand(), (Tn1k1, Tn0k1))
#
# Ts = TimeVar(name='Ts', length=10)
# Ts.v = np.ones((10, 1))




# X, Y, xi1, xi0, yi1, yi0, h = sp.symbols('X, Y, xi1, xi0, yi1, yi0, h')
# F: sp.Expr = 3 * X + 2 * Y
# G = X + Y
# f_ = xi1 - xi0 - h / 2 * (F.subs([(X, xi0), (Y, yi0)]) + F.subs([(X, xi1), (Y, yi1)]))
# g_ = xi1 + yi1
#
# F = sp.sympify('h-f1(X,Y)')
# f1 = sp.Function('f1')
# f1x = sp.Function('f1x')
# fx = F.diff('X')
# fx = fx.replace(sp.Derivative(f1(X, Y), X), f1x(X, Y))
# E16 = Eqn(name='E16',
#           e_str='A_i*mq',
#           commutative=False)
#
# fx_num = sp.lambdify([X, Y], fx, {'f1x': E16.NUM_EQN})
