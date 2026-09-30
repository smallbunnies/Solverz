"""Backward Euler written as its formula; the template of docs/src/integrator_adding_algorithms.md."""
from Solverz.integrator.algorithm import Algorithm

__all__ = ['ImplicitEuler']


class ImplicitEuler(Algorithm):
    r"""Backward Euler, ``M (y1 - y0) = h F(t0 + h, y1)``, of order 1.

    The algebraic equations hold at ``t0 + h`` by construction. The error estimate
    is the local error ``h**2 y''/2`` of the differential rows, written as
    ``(M (y1 - y0) - h D F(t0, y0)) / 2`` and passed through ``W = M - h J`` so that
    stiff and algebraic components are scaled. ``D`` removes the algebraic rows of
    ``F(t0, y0)``, where a residual that the consistent initialization left below
    its threshold would otherwise enter the estimate at every step size.
    """

    scheme = 'implicit_euler'
    order = 1
    error_order = 2
    adaptive = True

    def perform_step(self, s):
        y = s.implicit(s.t + s.h, 1.0, s.M @ s.y0)
        err = s.W(1.0).solve(0.5 * (s.M @ (y - s.y0) - s.h * (s.D * s.F0)))
        return y, err
