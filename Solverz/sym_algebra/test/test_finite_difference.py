from sympy import pycode, Integer

from Solverz.sym_algebra.transform import finite_difference, semi_descritize
from Solverz.sym_algebra.symbols import iVar, Para

pi = iVar('pi')
qi = iVar('qi')
S = Para('S')
va = Para('va')
D = Para('D')
lam = Para('lam')


# finite difference of hyperbolic PDE, the gas equation
def test_fd():
    fde = finite_difference(qi,
                            S * pi,
                            -lam * va ** 2 * qi ** 2 / (2 * D * S * pi),
                            [pi, qi],
                            10,
                            'central diff')
    assert pycode(
        fde) == 'S*dt*(-pi_tag_0[0:10] + pi_tag_0[1:11] - pi[0:10] + pi[1:11]) + dx*(-qi_tag_0[0:10] - qi_tag_0[1:11] + qi[0:10] + qi[1:11]) + (1/4)*dt*dx*lam*va**2*(qi_tag_0[0:10] + qi_tag_0[1:11] + qi[0:10] + qi[1:11])**2/(D*S*(pi_tag_0[0:10] + pi_tag_0[1:11] + pi[0:10] + pi[1:11]))'

    fde = finite_difference(pi,
                            va ** 2 / S * qi,
                            Integer(0),
                            [pi, qi],
                            10,
                            'central diff')
    assert pycode(
        fde) == 'dx*(-pi_tag_0[0:10] - pi_tag_0[1:11] + pi[0:10] + pi[1:11]) + dt*va**2*(-qi_tag_0[0:10] + qi_tag_0[1:11] - qi[0:10] + qi[1:11])/S'

    fde = finite_difference(qi,
                            S * pi,
                            -lam * va ** 2 * qi ** 2 / (2 * D * S * pi),
                            [pi, qi],
                            10,
                            'euler',
                            direction=1)
    assert pycode(
        fde) == 'S*dt*(-pi[0:10] + pi[1:11]) + dx*(-qi_tag_0[1:11] + qi[1:11]) + (1/2)*qi[1:11]**2*dt*dx*lam*va**2/(pi[1:11]*D*S)'

    fde = finite_difference(pi,
                            va ** 2 / S * qi,
                            Integer(0),
                            [pi, qi],
                            10,
                            'euler',
                            direction=1)
    assert pycode(fde) == 'dx*(-pi_tag_0[1:11] + pi[1:11]) + dt*va**2*(-qi[0:10] + qi[1:11])/S'

    fde = finite_difference(qi,
                            S * pi,
                            -lam * va ** 2 * qi ** 2 / (2 * D * S * pi),
                            [pi, qi],
                            10,
                            'euler',
                            direction=-1)
    assert pycode(
        fde) == 'S*dt*(-pi[0:10] + pi[1:11]) + dx*(-qi_tag_0[0:10] + qi[0:10]) + (1/2)*qi[0:10]**2*dt*dx*lam*va**2/(pi[0:10]*D*S)'

    fde = finite_difference(pi,
                            va ** 2 / S * qi,
                            Integer(0),
                            [pi, qi],
                            10,
                            'euler',
                            direction=-1)
    assert pycode(fde) == 'dx*(-pi_tag_0[0:10] + pi[0:10]) + dt*va**2*(-qi[0:10] + qi[1:11])/S'


def test_sd():
    # semi-discretize
    eqn_dict = semi_descritize(qi,
                               S * pi,
                               -lam * va ** 2 * qi ** 2 / (2 * D * S * pi),
                               [pi, qi],
                               11,
                               'TVD1',
                               a0=va,
                               a1=va)

    rhs = eqn_dict['Ode'][0]
    diff_var = eqn_dict['Ode'][1]

    assert pycode(
        rhs) == '-1/2*S*(-pi[0:10] + pi[2:12])/dx + (1/2)*va*(qi[0:10] - 2*qi[1:11] + qi[2:12])/dx - 1/2*qi[1:11]**2*lam*va**2/(pi[1:11]*D*S)'
    assert pycode(diff_var) == 'qi[1:11]'

    eqn_dict = semi_descritize(pi,
                               va ** 2 / S * qi,
                               Integer(0),
                               [pi, qi],
                               11,
                               'TVD1',
                               a0=va,
                               a1=va)
    rhs = eqn_dict['Ode'][0]
    diff_var = eqn_dict['Ode'][1]

    assert pycode(rhs) == '(1/2)*va*(pi[0:10] - 2*pi[1:11] + pi[2:12])/dx - 1/2*va**2*(-qi[0:10] + qi[2:12])/(S*dx)'
    assert pycode(diff_var) == 'pi[1:11]'
