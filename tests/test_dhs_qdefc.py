import sympy as sp
import numpy as np
from core.var import *
from core.equations import Equations
from core.eqn import Eqn

T = sp.Function('T')
dx, dt, x, t, Tn1k1, Tn0k1, Tn1k0 = sp.symbols('dx, dt, x, t, Tn1k1, Tn0k1, Tn1k0')
pde = sp.sympify('T(x,t).diff(t)+v*T(x,t).diff(x)+T(x,t)+b*T(x,t).diff(x)')
pde = pde.subs({T(x, t).diff(t): (Tn1k1 - Tn1k0) / dt,
                T(x, t).diff(x): (Tn1k1 - Tn0k1) / dx,
                T(x, t): Tn1k1},
               simultaneous=True)

sp.collect(pde.expand(), (Tn1k1, Tn0k1))

Ts = TimeVar(name='Ts', length=10)
Ts.v = np.ones((10, 1))


class MyVar(sp.Symbol):

    @property
    def v(self):
        return 1

    def _sympy_(self):
        return self.name


class MyFun(sp.Function, Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name=None):
        super(Equations).__init__(eqn)
        self.name = name


class MyExpr(sp.Expr):

    def _eval_nseries(self, x, n, logx, cdir):
        pass

    def __new__(cls, *args, **kwargs):
        pass


# x = sp.Function('x')
# f = x(t).diff(t) - 2 * (2 - sp.cos(x(t)))

x, y, xi1, xi0, yi1, yi0, h = sp.symbols('x, y, xi1, xi0, yi1, yi0, h')
f: sp.Expr = 3 * x + 2 * y
g = x + y
f_ = xi1 - xi0 - h / 2 * (f.subs([(x, xi0), (y, yi0)]) + f.subs([(x, xi1), (y, yi1)]))
g_ = xi1 + yi1

f = sp.sympify('h-f1(x,y)')
f1 = sp.Function('f1')
f1x = sp.Function('f1x')
fx = f.diff('x')
fx = fx.replace(sp.Derivative(f1(x, y), x), f1x(x, y))
E16 = Eqn(name='E16',
          e_str='A_i*mq',
          commutative=False)

fx_num = sp.lambdify([x, y], fx, {'f1x': E16.NUM_EQN})
