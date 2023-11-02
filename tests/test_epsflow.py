from Solverz import Eqn, idx, Var_, Const_, Sum_, sin, cos

g = Const_('g', dim=2)
b = Const_('b', dim=2)
h = idx('h')
k = idx('k')
B = idx('B')

v = Var_('v')
theta = Var_('theta')
p = Var_('p')

E1 = Eqn(name='Active power', eqn=v[h])

# from sympy import Sum
# from sympy.abc import m
#
a = v[h] * Sum_(v[k] * (g[h, k] * cos(theta[h] - theta[k]) + b[h, k] * sin(theta[h] - theta[k])), (k, B, ))
