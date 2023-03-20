import sympy as sp
from sympy import Derivative, Sum
from sympy.abc import x, y, z, t

from Solverz.algebra.sas_algebra import *
from Solverz.algebra.sas_algebra import dDelta

k = Index('k')
k0 = Index('k', sequence=0)
k1 = Index('k', sequence=1)


def test_dt_algebra():
    assert traverse_for_DT(x + y, Index('k')) == DT('X', k) + DT('Y', k)
    assert traverse_for_DT(3 * x, Index('k')) == 3 * DT('X', k)
    assert traverse_for_DT(x * y, Index('k')) == Sum(DT('X', k0) * DT('Y', k - k0), (k0, 1, k - 1)) + \
           + DT('X', 0) * DT('Y', k) + DT('X', k) * DT('Y', 0)
    assert traverse_for_DT(z, Index('k')) == DT('Z', k)
    assert traverse_for_DT(z - x, Index('k')) == DT('Z', k) - DT('X', k)
    assert traverse_for_DT(z - x * y, Index('k')) == DT('Z', k) + (-1) * \
           (DT('X', 0) * DT('Y', k) + DT('X', k) * DT('Y', 0) + Sum(DT('X', k0) * DT('Y', k - k0), (k0, 1, k - 1)))
    assert traverse_for_DT(z - 3 * x * y, Index('k')) == DT('Z', k) + (-3) * \
           (DT('X', 0) * DT('Y', k) + DT('X', k) * DT('Y', 0) + Sum(DT('X', k0) * DT('Y', k - k0), (k0, 1, k - 1)))
    assert traverse_for_DT(z - x * y * z, Index('k')) == DT('Z', k) + (-1)*(DT('X', 0)*(DT('Y', 0)*DT('Z', k) + DT('Y', k)*DT('Z', 0) + Sum(DT('Y', k0)*DT('Z', k - k0), (k0, 1, k - 1))) + DT('X', k)*DT('Y', 0)*DT('Z', 0) + Sum(DT('X', k0)*Sum(DT('Y', k1)*DT('Z', k - k0 - k1), (k1, 0, k - k0)), (k0, 1, k - 1)))
    assert traverse_for_DT(z - x * y * z, 3) == DT('Z', 3) + (-1) * (DT('X', 0)*(DT('Y', 0)*DT('Z', 3) + DT('Y', 3)*DT('Z', 0) + Sum(DT('Y', k0)*DT('Z', 3 - k0), (k0, 1, 2))) + DT('X', 3)*DT('Y', 0)*DT('Z', 0) + Sum(DT('X', k0)*Sum(DT('Y', k1)*DT('Z', -k0 - k1 + 3), (k1, 0, 3 - k0)), (k0, 1, 2)))

    assert traverse_for_DT(Derivative(y, t), Index('k')) == (k + 1) * DT('Y', k + 1)
    assert traverse_for_DT(sp.sin(x), Index('k')) == Sum((k - k0) / k * DT('Psi', k0) * DT('X', k - k0), (k0, 0, k - 1))
    assert traverse_for_DT(sp.sin(x), 0) == DT('Phi', 0)
    assert traverse_for_DT(sp.sin(x), 1) == DT('Psi', 0) * DT('X', 1)
    assert traverse_for_DT(sp.cos(x), Index('k')) == -Sum((k - k0) / k * DT('Phi', k0) * DT('X', k - k0),
                                                          (k0, 0, k - 1))
    assert traverse_for_DT(sp.cos(x), 0) == DT('Psi', 0)
    assert traverse_for_DT(sp.cos(x), 1) == -DT('Phi', 0) * DT('X', 1)

    assert traverse_for_DT(sp.sin(x * y + z), Index('k')) == Sum(
        (k - k0) / k * DT('Psi', k0) * (DT('Z', k - k0) + Sum(DT('X', k1) * DT('Y', k - k0 - k1), (k1, 0, k - k0))),
        (k0, 0, k - 1))
    assert traverse_for_DT(sp.Integer(5), Index('k')) == 5 * dDelta(k)
    assert traverse_for_DT(sp.Integer(5), 0) == 5
    assert traverse_for_DT(sp.Integer(5), 1) == 0
    assert traverse_for_DT(t, Index('k')) == DT('T', Index('k'))
    assert traverse_for_DT(t, 0) == 0
    assert traverse_for_DT(t, 1) == 1
    assert traverse_for_DT(sp.sin(t), 0) == DT('Phi', 0)
    assert traverse_for_DT(sp.sin(t), Index('k')) == Sum((k - k0) * DT('Psi', k0) * DT('T', k - k0) / k, (k0, 0, k - 1))



