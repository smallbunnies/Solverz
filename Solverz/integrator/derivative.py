"""The partial derivative of the residual with respect to time.

A Rosenbrock step of a non-autonomous model needs ``dF/dt`` at the start of
the step. Both policies take a forward difference, since a
``TimeSeriesParam`` cannot be evaluated before its first time stamp, and
both write into buffers the Integrator owns: ``F(t + delta, y, out=scratch)``
is the one residual they evaluate, and the quotient is written into ``ft``.
``f0`` is ``F(t, y)``, which the step has already evaluated.

``'ode23s'``, the default, scales the increment with the step, so the
product ``dt * ft`` that enters the stages carries a rounding error of about
``SQRT_EPS`` times the terms of ``F`` at every ``t`` and every ``dt``, and
the probe never leaves the step. ``'legacy'`` is the difference quotient of
legacy Rodas, whose increment at ``t = 0`` is ``1.49e-16`` whatever the
step.
"""
import numpy as np

__all__ = []

# sqrt(spacing(1.0)), which is 2**-26 exactly
SQRT_EPS = float(np.sqrt(np.spacing(1.0)))


def dfdt_ode23s(F, t, dt, y, f0, ft, scratch):
    """``dF/dt`` at ``(t, y)`` into ``ft`` with the increment of MATLAB's ode23s.

    ``dt`` is the step of the first attempt of the step, after bounds and
    truncation. The increment ``tdel`` is rounded to a representable
    difference of times, and ``ft`` is zero when it vanishes, which happens
    only when ``t + dt`` rounds to ``t``.
    """
    delt = SQRT_EPS * max(abs(t), abs(t + dt))
    tdel = (t + min(delt, abs(dt))) - t
    if tdel == 0.0:
        ft.fill(0.0)
        return ft
    F(t + tdel, y, out=scratch)
    np.subtract(scratch, f0, out=ft)
    np.divide(ft, tdel, out=ft)
    return ft


def dfdt_legacy(F, t, dt, y, f0, ft, scratch):
    """``dF/dt`` at ``(t, y)`` into ``ft`` with the expressions of legacy Rodas.

    The values are those of ``Solverz.solvers.daesolver.rodas.rodas.dfdt``
    bit for bit; ``dt`` is not read.
    """
    tscale = np.maximum(0.1 * np.abs(t), 1e-8)
    ddt = t + np.sqrt(np.spacing(1)) * tscale - t
    F(t + ddt, y, out=scratch)
    np.subtract(scratch, f0, out=ft)
    np.divide(ft, ddt, out=ft)
    return ft


DFDT_POLICIES = {'ode23s': dfdt_ode23s, 'legacy': dfdt_legacy}
