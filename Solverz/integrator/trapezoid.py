"""The trapezoidal rule written as its formula."""
from Solverz.integrator.algorithm import Algorithm

__all__ = ['Trapezoid']


class Trapezoid(Algorithm):
    r"""``M (y1 - y0) = h/2 (F(t0, y0) + F(t0 + h, y1))`` on the differential rows and
    ``0 = F(t0 + h, y1)`` on the algebraic rows, of order 2.

    ``f0 = D F(t0, y0)`` keeps the differential rows only. Averaging the algebraic
    equations as well would make their residual alternate in sign at a constant
    size, and an error estimate that carries it could never meet a tolerance below it.
    The error estimate ``M (y1 - y0) - h f0``, passed through ``W = M - h/2 J``,
    is the local error of the explicit Euler step, of order ``h**2``; it is
    conservative for the trapezoidal rule, whose own local error is of order ``h**3``.
    """

    scheme = 'trapezoid'
    order = 2
    error_order = 2
    adaptive = True

    def perform_step(self, s):
        f0 = s.D * s.F0
        y = s.implicit(s.t + s.h, 0.5, s.M @ s.y0 + 0.5 * s.h * f0)
        return y, s.W(0.5).solve(s.M @ (y - s.y0) - s.h * f0)
