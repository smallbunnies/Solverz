"""The simplified Newton iteration behind ``implicit``.

It solves ``G(y) = M y - h*gamma*F(t, y) - rhs = 0`` with the Newton matrix
``W = M - h*gamma*J0`` of the attempt, which the Integrator factorizes once
per ``gamma`` and keeps across the iterations and across the calls of the
attempt. The stopping test is Hairer's, with ``KAPPA`` applied to the RMS
norm of the correction weighted by the tolerances of the run, as in SciML's
``nlsolve``: the error of the returned iterate is estimated at one percent
of the tolerance. A failure raises ``StepFailure``, which rejects the
attempt, and the loop retries with half the step.
"""
import math

import numpy as np

from Solverz.integrator.algorithm import StepFailure

__all__ = []

KAPPA = 0.01
MAXIT = 10
_EPS = float(np.spacing(1.0))


def implicit(integ, t, gamma, rhs, y_start=None, out=None, slope=False):
    """``y`` with ``M y - h*gamma*F(t, y) = rhs``, from ``y_start`` or ``uprev``.

    The iteration runs in the Integrator's buffers ``_nl_y``, ``_nl_G``,
    ``_nl_dz``, ``_nl_w`` and ``_nl_w2``, so it allocates only ``M @ y`` per
    iteration and the backend's result under SuperLU. Each iteration costs
    one counted residual and one counted solve. The result is copied out of
    ``_nl_y``, which the next call overwrites: into ``out`` when it is
    given, else into a new array. With ``slope=True`` the result is ``(y,
    k)`` with ``k = (M y - rhs) / (h*gamma)``, which equals ``F(t, y)`` to the
    Newton tolerance at the cost of one matrix-vector product; ``out`` is
    then a pair, either of whose entries may be ``None``.

    The rate estimate ``eta`` of the last successful call starts the next
    one, so a call whose first correction is already small stops after one
    residual.
    """
    M, opts = integ.M, integ.opts
    rtol, atol = opts.rtol, opts.atol
    W = integ.W(gamma)
    hgamma = integ.dt * gamma
    y, G, dz, w, w2 = integ._nl_y, integ._nl_G, integ._nl_dz, integ._nl_w, integ._nl_w2
    uprev = integ.uprev
    np.copyto(y, uprev if y_start is None else y_start)
    eta = max(integ._nl_eta, _EPS) ** 0.8
    ndz_prev = None
    for k in range(1, MAXIT + 1):
        integ.F(t, y, out=G)
        np.multiply(G, hgamma, out=G)
        np.subtract(M @ y, G, out=G)
        np.subtract(G, rhs, out=G)
        W.solve(G, out=dz)
        np.subtract(y, dz, out=y)
        np.abs(y, out=w)
        np.abs(uprev, out=w2)
        np.maximum(w, w2, out=w)
        np.multiply(w, rtol, out=w)
        np.add(w, atol, out=w)
        np.divide(dz, w, out=w)
        ndz = np.sqrt(np.mean(np.square(w, out=w)))
        if not math.isfinite(ndz):
            raise StepFailure('the Newton iteration produced a non-finite value')
        if ndz == 0.0:
            break
        if ndz_prev is not None:
            theta = ndz / ndz_prev
            if theta >= 1.0:
                raise StepFailure('the Newton iteration diverged')
            eta = theta / (1.0 - theta)
            # the error predicted after the iterations that remain
            if ndz * theta ** (MAXIT - k) / (1.0 - theta) > KAPPA:
                raise StepFailure('the Newton iteration converges too slowly')
        if eta * ndz < KAPPA:
            break
        ndz_prev = ndz
    else:
        raise StepFailure('the Newton iteration did not converge')
    integ._nl_eta = eta

    y_out, k_out = out if (slope and out is not None) else (out, None)
    if slope:
        # before y is written, since the caller may pass rhs as a target
        k_out = np.empty(integ.n) if k_out is None else k_out
        np.subtract(M @ y, rhs, out=k_out)
        np.divide(k_out, hgamma, out=k_out)
    if y_out is None:
        y_out = y.copy()
    else:
        np.copyto(y_out, y)
    return (y_out, k_out) if slope else y_out
