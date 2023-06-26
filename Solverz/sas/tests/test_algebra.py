import sympy as sp
from sympy import Derivative, sin, cos
from sympy.abc import x, y, z, t, a, b, c

from Solverz.sas.sas_alg import Index, Slice, DT, dDelta, dConv_s, dConv_v, dLinspace, dtify, _dtify, psi, phi, Constant

k = Index('k')
Xk = DT(x, k)
X0 = DT(x, 0)
X1k = DT(x, Slice(1, k))
Yk = DT(y, k)
Y0 = DT(y, 0)
Zk = DT(z, k)
Z0 = DT(z, 0)
Y1k = DT(y, Slice(1, k))
Z1k = DT(z, Slice(1, k))
X0k = DT(x, Slice(0, k))
X0k1 = DT(x, Slice(0, k - 1))
Y0k = DT(y, Slice(0, k))
Y0k1 = DT(y, Slice(0, k - 1))
Z0k = DT(z, Slice(0, k))
Lin0k1 = dLinspace(0, k - 1)
Lin0k = dLinspace(0, k)
X03 = DT(x, Slice(0, 3))
Y03 = DT(y, Slice(0, 3))
Z03 = DT(z, Slice(0, 3))
A0k = DT(a, Slice(0, k))


def test_dt_algebra():
    assert dtify(x + y).__repr__() == 'x[k] + y[k]=0'
    assert dtify(x * y).__repr__() == 'dConv_s(x[0:k], y[0:k])=0'
    assert dtify(x * y, Slice(0, k)).__repr__() == 'dConv_v(x[0:k], y[0:k])=0'
    assert dtify(x * y, Slice(1, k)).__repr__() == 'x[1:k]*y[0] + dConv_v(x[0:k - 1], y[1:k])=0'
    assert dtify(x * y * z).__repr__() == 'dConv_s(x[0:k], y[0:k], z[0:k])=0'
    assert dtify(z - x * y).__repr__() == 'z[k] - dConv_s(x[0:k], y[0:k])=0'
    assert dtify(z * (x + y)).__repr__() == 'dConv_s(z[0:k], x[0:k] + y[0:k])=0'
    assert dtify(x + y, 3).__repr__() == 'x[3] + y[3]=0'
    assert dtify(x * y, 3).__repr__() == 'dConv_s(x[0:3], y[0:3])=0'
    assert dtify(x * y * z, 3).__repr__() == 'dConv_s(x[0:3], y[0:3], z[0:3])=0'
    assert dtify(z - x * y, 3).__repr__() == 'z[3] - dConv_s(x[0:3], y[0:3])=0'
    assert dtify(z * (x + y), 3).__repr__() == 'dConv_s(z[0:3], x[0:3] + y[0:3])=0'
    assert dtify(x + y, 0).__repr__() == 'x[0] + y[0]=0'
    assert dtify(x * y, 0).__repr__() == 'x[0]*y[0]=0'
    assert dtify(x * y * z, 0).__repr__() == 'x[0]*y[0]*z[0]=0'
    assert dtify(z - x * y, 0).__repr__() == '-x[0]*y[0] + z[0]=0'
    assert dtify(z * (x + y), 0).__repr__() == 'z[0]*(x[0] + y[0])=0'
    assert _dtify(sin(x), k) == dConv_s((k - Lin0k1) * DT(psi(x), Slice(0, k - 1)) / k, X1k)
    assert _dtify(sin(x), 2) == dConv_s((2 - dLinspace(0, 1)) * DT(psi(x), Slice(0, 1)) / 2, DT(x, Slice(1, 2)))
    assert _dtify(sin(x), 1) == DT(psi(x), 0) * DT(x, 1)
    assert _dtify(sin(x), 0) == DT(phi(x), 0)
    assert _dtify(sin(x * y), k) == dConv_s((k - Lin0k1) * DT(psi(x * y), Slice(0, k - 1)) / k,
                                            dConv_v(X0k1, Y1k) + X1k * Y0)
    expr1 = _dtify(sin(x * y), k)
    assert \
        expr1.expand(func=True, mul=False) == \
        dConv_s((k - Lin0k1) * DT(psi(x * y), Slice(0, k - 1)) / k, X0k1, Y1k) + dConv_s(
            (k - Lin0k1) * DT(psi(x * y), Slice(0, k - 1)) / k, X1k * Y0)
    assert _dtify(sin(x * y), 2) == \
           dConv_s((2 - dLinspace(0, 1)) * DT(psi(x * y), Slice(0, 1)) / 2,
                   dConv_v(DT(x, Slice(0, 1)), DT(y, Slice(1, 2))) + DT(x, Slice(1, 2)) * Y0)
    assert _dtify(sin(x * y), 1) == DT(psi(x * y), 0) * dConv_s(DT(x, Slice(0, 1)), DT(y, Slice(0, 1)))
    assert _dtify(sin(x * y), 0) == DT(phi(x * y), 0)
    assert _dtify(cos(x), k) == -dConv_s((k - Lin0k1) * DT(phi(x), Slice(0, k - 1)) / k, X1k)
    assert _dtify(cos(x), 2) == -dConv_s((2 - dLinspace(0, 1)) * DT(phi(x), Slice(0, 1)) / 2, DT(x, Slice(1, 2)))
    assert _dtify(cos(x), 1) == -DT(phi(x), 0) * DT(x, 1)
    assert _dtify(cos(x), 0) == DT(psi(x), 0)
    assert _dtify(cos(x * y), k) == -dConv_s((k - Lin0k1) * DT(phi(x * y), Slice(0, k - 1)) / k,
                                             dConv_v(X0k1, Y1k) + X1k * Y0)
    assert _dtify(cos(x * y), k).expand(func=True, mul=False) == \
           (-1) * (dConv_s((k - Lin0k1) * DT(phi(x * y), Slice(0, k - 1)) / k, X0k1, Y1k) +
                   dConv_s((k - Lin0k1) * DT(phi(x * y), Slice(0, k - 1)) / k, X1k * Y0))
    assert _dtify(cos(x * y), 2) \
           == -dConv_s((2 - dLinspace(0, 1)) * DT(phi(x * y), Slice(0, 1)) / 2,
                       dConv_v(DT(x, Slice(0, 1)), DT(y, Slice(1, 2))) +
                       DT(x, Slice(1, 2)) * Y0)
    assert _dtify(cos(x * y), 1) == -DT(phi(x * y), 0) * dConv_s(DT(x, Slice(0, 1)), DT(y, Slice(0, 1)))
    assert _dtify(cos(x * y), 0) == DT(psi(x * y), 0)
    expr2 = cos(x * y + x * z)
    assert _dtify(expr2, k) == \
           -dConv_s((k - Lin0k1) * DT(phi(x * y + x * z), Slice(0, k - 1)) / k,
                    dConv_v(X0k1, Y1k) + X1k * Y0 + dConv_v(X0k1, Z1k) + X1k * Z0)
    assert _dtify(expr2, k).expand(func=True, mul=False) == \
           (-1) * (dConv_s((k - Lin0k1) * DT(phi(x * y + x * z), Slice(0, k - 1)) / k, X0k1, Y1k) +
                   dConv_s((k - Lin0k1) * DT(phi(x * y + x * z), Slice(0, k - 1)) / k, X1k * Y0) +
                   dConv_s((k - Lin0k1) * DT(phi(x * y + x * z), Slice(0, k - 1)) / k, X0k1, Z1k) +
                   dConv_s((k - Lin0k1) * DT(phi(x * y + x * z), Slice(0, k - 1)) / k, X1k * Z0))
    expr3 = x * sin(y)
    assert _dtify(expr3, k) == \
           dConv_s(X0k,
                   dConv_v((k - Lin0k) * DT(psi(y), Slice(0, k)) / k, Y0k * (1 - dDelta(Slice(0, k)))) +
                   dDelta(Slice(0, k)) * DT(phi(y), 0))
    assert [arg.__repr__() for arg in dtify(expr3, etf=True)] == \
           ['dConv_s(x[0:k], phi_y[0:k])=0',
            'phi_y[k] - dConv_s(psi_y[0:k - 1]*(k - dLinspace(0, k - 1))/k, y[1:k])=0',
            'psi_y[k] + dConv_s(phi_y[0:k - 1]*(k - dLinspace(0, k - 1))/k, y[1:k])=0']
    expr4 = x * sin(x * y)
    assert _dtify(expr4, k) == \
           dConv_s(X0k,
                   dConv_v((k - Lin0k) * DT(psi(x * y), Slice(0, k)) / k,
                           dConv_v(X0k, Y0k * (1 - dDelta(Slice(0, k))))) +
                   dDelta(Slice(0, k)) * DT(phi(x * y), 0))
    assert _dtify(expr4, k).expand(func=True, mul=False) == \
           dConv_s(X0k, (k - Lin0k) * DT(psi(x * y), Slice(0, k)) / k, X0k, Y0k * (1 - dDelta(Slice(0, k)))) + \
           dConv_s(X0k, dDelta(Slice(0, k)) * DT(phi(x * y), 0))
    expr5 = (x + x * z) * (x + y * z)
    assert _dtify(expr5, k) == dConv_s(X0k + dConv_v(X0k, Z0k), X0k + dConv_v(Y0k, Z0k))
    assert _dtify(expr5, k).expand(func=True, mul=False) == \
           dConv_s(X0k, X0k) + dConv_s(X0k, Y0k, Z0k) + dConv_s(X0k, Z0k, X0k) + dConv_s(X0k, Z0k, Y0k, Z0k)
    assert _dtify(sin(t), Index('k')) == dConv_s((k - Lin0k1) * DT(psi(t), Slice(0, k - 1)) / k, DT(t, Slice(1, k)))
    assert _dtify(sin(t), 0) == 0
    assert _dtify(cos(t), 0) == 1
    assert dtify(Derivative(y, t), Index('k')).__repr__() == 'y[k + 1]*(k + 1)=0'
    assert _dtify(sp.Integer(5), Index('k')) == 5 * dDelta(k)
    assert _dtify(sp.Integer(5), 0) == 5
    assert _dtify(sp.Integer(5), 1) == 0
    assert _dtify(t, Index('k')) == DT(t, Index('k'))
    assert _dtify(t, 0) == 0
    assert _dtify(t, 1) == 1
    assert dtify((x + a * z) * sp.cos(y), constants=['a']).__repr__() == 'dConv_s(a*z[0:k] + x[0:k], psi_y[0:k])=0'
    assert dtify((x + a * z) * sp.cos(y)).__repr__() == 'dConv_s(x[0:k] + dConv_v(a[0:k], z[0:k]), psi_y[0:k])=0'
    assert _dtify((a - x * y) * z, k).expand(func=True, mul=False) == \
           dConv_s(Z0k, DT(a, Slice(0, k))) + dConv_s(Z0k, -X0k, Y0k)
    assert dtify((a - x * y) * cos(z)).__repr__() == 'dConv_s(a[0:k] - dConv_v(x[0:k], y[0:k]), psi_z[0:k])=0'
    assert dtify((a - x * y) * cos(z), k).LHS.expand(func=True, mul=False) == \
           dConv_s(A0k, DT(psi(z), Slice(0, k))) + dConv_s(-X0k, Y0k, DT(psi(z), Slice(0, k)))
    assert dtify((a - 2 * x * y) * cos(z)).LHS.expand(func=True, mul=False) == \
           dConv_s(A0k, DT(psi(z), Slice(0, k))) + dConv_s(-2 * X0k, Y0k, DT(psi(z), Slice(0, k)))
    assert [arg.__repr__() for arg in dtify(Derivative(y, t) - 2 * (y - cos(t)), etf=True, k=k)] == \
           ['2*psi_t[k] + y[k + 1]*(k + 1) - 2*y[k]=0',
            'psi_t[k] + dConv_s(phi_t[0:k - 1]*(k - dLinspace(0, k - 1))/k, t[1:k])=0',
            'phi_t[k] - dConv_s(psi_t[0:k - 1]*(k - dLinspace(0, k - 1))/k, t[1:k])=0']
    assert dtify(Derivative(y, t) - 2 * (y - cos(t)), eut=True).__repr__() == \
           'y[k + 1]=2*(-psi_t[k] + y[k])/(k + 1)'
    assert [arg.__repr__() for arg in dtify(Derivative(y, t) - 2 * (y - cos(t)), etf=True, eut=True, k=k)] == \
           ['y[k + 1]=2*(-psi_t[k] + y[k])/(k + 1)',
            '-psi_t[k]=phi_t[0]*t[k] + dConv_s(phi_t[1:k - 1]*(k - dLinspace(1, k - 1))/k, t[1:k - 1])',
            '-phi_t[k]=-psi_t[0]*t[k] - dConv_s(psi_t[1:k - 1]*(k - dLinspace(1, k - 1))/k, t[1:k - 1])']
    test = dtify(Derivative(y, t) - 2*(y-cos(t)), etf=True, eut=True, k=Index('k'))
    assert test[0].__repr__() == "y[k + 1]=2*(-psi_t[k] + y[k])/(k + 1)"
    assert test[1].__repr__() == "-psi_t[k]=phi_t[0]*t[k] + dConv_s(phi_t[1:k - 1]*(k - dLinspace(1, k - 1))/k, t[1:k - 1])"
    assert test[2].__repr__() == "-phi_t[k]=-psi_t[0]*t[k] - dConv_s(psi_t[1:k - 1]*(k - dLinspace(1, k - 1))/k, t[1:k - 1])"
