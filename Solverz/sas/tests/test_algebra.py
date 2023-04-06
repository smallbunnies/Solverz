import sympy as sp
from sympy import Derivative
from sympy.abc import x, y, z, t

from Solverz.sas.sas_alg import Index, Slice, DT, dDelta, dConv_s, dConv_v, dLinspace, dtify

k = Index('k')
k0 = Index('k', sequence=0)
Xk = DT('x', k)
X0 = DT('x', 0)
X1k = DT('x', Slice(1, k))
Yk = DT('y', k)
Y0 = DT('y', 0)
Zk = DT('z', k)
Z0 = DT('z', 0)
Y1k = DT('y', Slice(1, k))
Z1k = DT('z', Slice(1, k))
X0k = DT('x', Slice(0, k))
X0k1 = DT('x', Slice(0, k - 1))
Y0k = DT('y', Slice(0, k))
Y0k1 = DT('y', Slice(0, k - 1))
Z0k = DT('z', Slice(0, k))
Psi0k1 = DT('psi', Slice(0, k - 1))
Phi0k1 = DT('phi', Slice(0, k - 1))
Psi0k = DT('psi', Slice(0, k))
Phi0k = DT('phi', Slice(0, k))
Psi0 = DT('psi', 0)
Phi0 = DT('phi', 0)
Lin0k1 = dLinspace(0, k - 1)
Lin0k = dLinspace(0, k)
X03 = DT('x', Slice(0, 3))
Y03 = DT('y', Slice(0, 3))
Z03 = DT('z', Slice(0, 3))


def test_dt_algebra():
    assert dtify(x + y, k) == Xk + Yk
    assert dtify(x * y, k) == dConv_s(X0k, Y0k)
    assert dtify(x * y, Slice(0, k)) == dConv_v(X0k, Y0k)
    assert dtify(x * y, Slice(1, k)) == dConv_v(X0k1, Y1k) + X1k * Y0
    assert dtify(x * y * z, k) == dConv_s(X0k, Y0k, Z0k)
    assert dtify(x * y * z, k).expand(func=True, mul=False) == dConv_s(X0k, Y0k, Z0k)
    assert dtify(z - x * y, k) == DT('z', k) - dConv_s(X0k, Y0k)
    assert dtify(z * (x + y), k) == dConv_s(DT('z', Slice(0, k)), X0k + Y0k)
    assert dtify(x + y, 3) == DT('x', 3) + DT('y', 3)
    assert dtify(x * y, 3) == dConv_s(X03, Y03)
    assert dtify(x * y * z, 3) == dConv_s(X03, Y03, Z03)
    assert dtify(x * y * z, 3).expand(func=True, mul=False) == dConv_s(X03, Y03, Z03)
    assert dtify(z - x * y, 3) == DT('z', 3) - dConv_s(X03, Y03)
    assert dtify(z * (x + y), 3) == dConv_s(Z03, X03 + Y03)
    assert dtify(x + y, 0) == X0 + Y0
    assert dtify(x * y, 0) == X0 * Y0
    assert dtify(x * y * z, 0) == X0 * Y0 * Z0
    assert dtify(z - x * y, 0) == Z0 - X0 * Y0
    assert dtify(z * (x + y), 0) == Z0 * (X0 + Y0)
    assert dtify(sp.sin(x), k) == dConv_s((k - Lin0k1) * Psi0k1 / k, X1k)
    assert dtify(sp.sin(x), 2) == dConv_s((2 - dLinspace(0, 1)) * DT('psi', Slice(0, 1)) / 2, DT('x', Slice(1, 2)))
    assert dtify(sp.sin(x), 1) == Psi0 * DT('x', 1)
    assert dtify(sp.sin(x), 0) == Phi0
    assert dtify(sp.sin(x * y), k) == dConv_s((k - Lin0k1) * Psi0k1 / k, dConv_v(X0k1, Y1k) + X1k * Y0)
    expr1 = dtify(sp.sin(x * y), k)
    assert \
        expr1.expand(func=True, mul=False) == \
        dConv_s((k - Lin0k1) * Psi0k1 / k, X0k1, Y1k) + dConv_s((k - Lin0k1) * Psi0k1 / k, X1k * Y0)
    assert dtify(sp.sin(x * y), 2) == \
           dConv_s((2 - dLinspace(0, 1)) * DT('psi', Slice(0, 1)) / 2,
                   dConv_v(DT('x', Slice(0, 1)), DT('y', Slice(1, 2))) + DT('x', Slice(1, 2)) * Y0)
    assert dtify(sp.sin(x * y), 1) == Psi0 * dConv_s(DT('x', Slice(0, 1)), DT('y', Slice(0, 1)))
    assert dtify(sp.sin(x * y), 0) == Phi0
    assert dtify(sp.cos(x), k) == -dConv_s((k - Lin0k1) * Phi0k1 / k, X1k)
    assert dtify(sp.cos(x), 2) == -dConv_s((2 - dLinspace(0, 1)) * DT('phi', Slice(0, 1)) / 2, DT('x', Slice(1, 2)))
    assert dtify(sp.cos(x), 1) == -Phi0 * DT('x', 1)
    assert dtify(sp.cos(x), 0) == Psi0
    assert dtify(sp.cos(x * y), k) == -dConv_s((k - Lin0k1) * Phi0k1 / k, dConv_v(X0k1, Y1k) + X1k * Y0)
    assert dtify(sp.cos(x * y), k).expand(func=True, mul=False) == \
           (-1) * (dConv_s((k - Lin0k1) * Phi0k1 / k, X0k1, Y1k) +
                   dConv_s((k - Lin0k1) * Phi0k1 / k, X1k * Y0))
    assert dtify(sp.cos(x * y), 2) \
           == -dConv_s((2 - dLinspace(0, 1)) * DT('phi', Slice(0, 1)) / 2,
                       dConv_v(DT('x', Slice(0, 1)), DT('y', Slice(1, 2))) +
                       DT('x', Slice(1, 2)) * Y0)
    assert dtify(sp.cos(x * y), 1) == -Phi0 * dConv_s(DT('x', Slice(0, 1)), DT('y', Slice(0, 1)))
    assert dtify(sp.cos(x * y), 0) == Psi0
    expr2 = sp.cos(x * y + x * z)
    assert dtify(expr2, k) == \
           -dConv_s((k - Lin0k1) * Phi0k1 / k, dConv_v(X0k1, Y1k) + X1k * Y0 + dConv_v(X0k1, Z1k) + X1k * Z0)
    assert dtify(expr2, k).expand(func=True, mul=False) == \
           (-1) * (dConv_s((k - Lin0k1) * Phi0k1 / k, X0k1, Y1k) + dConv_s((k - Lin0k1) * Phi0k1 / k, X1k * Y0) +
                   dConv_s((k - Lin0k1) * Phi0k1 / k, X0k1, Z1k) + dConv_s((k - Lin0k1) * Phi0k1 / k, X1k * Z0))
    expr3 = x * sp.sin(y)
    assert dtify(expr3, k) == \
           dConv_s(X0k, dConv_v((k - Lin0k) * Psi0k / k, Y0k * (1 - dDelta(Slice(0, k)))) + dDelta(Slice(0, k)) * Phi0)
    expr4 = x * sp.sin(x * y)
    assert dtify(expr4, k) == \
           dConv_s(X0k, dConv_v((k - Lin0k) * Psi0k / k, dConv_v(X0k, Y0k * (1 - dDelta(Slice(0, k))))) + dDelta(
               Slice(0, k)) * Phi0)
    assert dtify(expr4, k).expand(func=True, mul=False) == \
           dConv_s(X0k, (k - Lin0k) * Psi0k / k, X0k, Y0k * (1 - dDelta(Slice(0, k)))) + \
           dConv_s(X0k, dDelta(Slice(0, k)) * Phi0)
    expr5 = (x + x * z) * (x + y * z)
    assert dtify(expr5, k) == dConv_s(X0k + dConv_v(X0k, Z0k), X0k + dConv_v(Y0k, Z0k))
    assert dtify(expr5, k).expand(func=True, mul=False) == \
           dConv_s(X0k, X0k) + dConv_s(X0k, Y0k, Z0k) + dConv_s(X0k, Z0k, X0k) + dConv_s(X0k, Z0k, Y0k, Z0k)
    assert dtify(sp.sin(t), Index('k')) == dConv_s((k - Lin0k1) * Psi0k1 / k, DT('t', Slice(1, k)))
    assert dtify(Derivative(y, t), Index('k')) == (k + 1) * DT('y', k + 1)
    assert dtify(sp.Integer(5), Index('k')) == 5 * dDelta(k)
    assert dtify(sp.Integer(5), 0) == 5
    assert dtify(sp.Integer(5), 1) == 0
    assert dtify(t, Index('k')) == DT('t', Index('k'))
    assert dtify(t, 0) == 0
    assert dtify(t, 1) == 1
    assert dtify(sp.sin(t), 0) == DT('phi', 0)
    assert dtify(sp.sin(t), Index('k')) == dConv_s((k - Lin0k1) * Psi0k1 / k, DT('t', Slice(1, k)))
