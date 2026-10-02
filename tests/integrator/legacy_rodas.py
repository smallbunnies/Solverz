"""A transcription of legacy Rodas that exposes every attempt.

``legacy_step`` is one attempt, ``rodas.py:156-212`` line by line with the
Jacobian passed in, and ``legacy_run`` is the loop, ``rodas.py:65-358``,
without the event branch, the progress bar and the counters, none of which
changes a state. ``test_legacy_transcription.py`` proves that ``legacy_run``
reproduces whole legacy trajectories bit for bit, which makes the attempts it
records a validated reference for the integrator's own step. ``dfdt`` and
``ntrp`` are imported from the legacy module, since they are the legacy code.

The transcription keeps legacy's side effects on ``opt``: ``hmax`` is written
when it is ``None``, and ``facmax``, the growth cap of the step-size
controller, is written after every attempt. Like legacy Rodas, a call
therefore needs an ``Opt`` of its own to be reproducible.

Every ``rodas.py:N`` refers to legacy Rodas at commit ``056e87a``, before its
deprecation warning moved its lines.
"""
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
from scipy.sparse import diags_array

from Solverz.solvers.daesolver.daeic import DaeIc
from Solverz.solvers.daesolver.rodas.param import Rodas_param
from Solverz.solvers.daesolver.rodas.rodas import dfdt, ntrp
from Solverz.solvers.klu_backend import KLUCache
from Solverz.solvers.laesolver import lu_decomposition, resolve_backend
from Solverz.solvers.option import Opt


class LegacyStop(Exception):
    """The factorization failed, where legacy Rodas silently leaves its loop."""


class Attempt(NamedTuple):
    """One attempt of ``legacy_run``.

    ``t`` and ``dt`` are stored unconverted: ``t`` is ``np.int64`` on the
    first attempt of an integer ``tspan``, and ``dt`` can be ``np.int64`` when
    the stretch to ``tend`` fires on that attempt. ``dt`` is the step after
    the stretch, the one the attempt uses. ``err_raw`` is the error after the
    ``1e6`` override and before the floor of ``1e-6``, and ``J`` is the
    Jacobian the attempt used, evaluated on the first attempt of the step.
    """
    t: Any
    y0: np.ndarray
    dt: Any
    reject: int
    ynew: np.ndarray
    err_raw: Any
    J: Any


def legacy_iteration_matrix(M, J, dt, gamma, state):
    """``(Miter, rscale)``: the row-scaled iteration matrix of ``rodas.py:173-185``."""
    Miter = M - dt * gamma * J
    row_max = np.max(np.abs(Miter), axis=1)
    if state.rscale_to_dense is None:
        state.rscale_to_dense = hasattr(row_max, 'toarray')
    if state.rscale_to_dense:
        row_max = row_max.toarray()
    rscale = (1.0 / np.asarray(row_max)).ravel()
    Miter = diags_array(rscale, format='csc') @ Miter
    return Miter, rscale


def legacy_step(dae, M, p, t, y0, dt, J, rparam, opt, linsolver, klu_cache, state):
    """One attempt of legacy Rodas: ``(ynew, err_raw, K)``.

    ``state`` carries ``rscale_to_dense``, which legacy decides on the first
    assembly of a call. A failed factorization raises ``LegacyStop``; an
    error of a solve propagates, as it escapes legacy Rodas.
    """
    vsize = y0.shape[0]
    K = np.zeros((vsize, rparam.s))

    dfdt0 = dt * dfdt(dae, t, y0)
    rhs = dae.F(t, y0, p) + rparam.g[0] * dfdt0

    Miter, rscale = legacy_iteration_matrix(M, J, dt, rparam.gamma, state)
    try:
        lu = lu_decomposition(Miter, backend=linsolver, cache=klu_cache)
    except RuntimeError as e:
        raise LegacyStop(str(e)) from e
    K[:, 0] = lu.solve(rscale * rhs)

    for j in range(1, rparam.s):
        sum_1 = K @ rparam.alpha[:, j]
        sum_2 = K @ rparam.gammatilde[:, j]
        y1 = y0 + dt * sum_1

        rhs = dae.F(t + dt * rparam.a[j], y1, p) + M @ sum_2 + rparam.g[j] * dfdt0
        sol = lu.solve(rscale * rhs)
        K[:, j] = sol - sum_2

    sum_1 = K @ (dt * rparam.b)
    ynew = y0 + sum_1
    if not opt.fix_h:
        sum_2 = K @ (dt * rparam.bd)
        SK = (opt.atol + opt.rtol * np.abs(ynew)).reshape((-1,))
        err = np.max(np.abs((sum_1 - sum_2) / SK))
        if np.any(np.isinf(ynew)) or np.any(np.isnan(ynew)):
            err = 1.0e6
            print('Warning Rodas: NaN or Inf occurs.')
    else:
        err = 1.0
    return ynew, err, K


def legacy_run(dae, tspan, y0, opt, attempts=None):
    """``(T, Y)`` of legacy ``Rodas(dae, tspan, y0, opt)`` on an event-free run.

    ``y0`` is an ndarray. With a list ``attempts``, an ``Attempt`` is appended
    for every attempt, rejected ones included.
    """
    if opt is None:
        opt = Opt()
    if opt.event is not None:
        raise ValueError("legacy_run transcribes the event-free path of Rodas only")

    rparam = Rodas_param(opt.scheme)
    vsize = y0.shape[0]
    tspan = np.array(tspan)
    tend = tspan[-1]
    t0 = tspan[0]
    if t0 > tend:
        raise ValueError(f't0: {t0} > tend: {tend}')
    if opt.hmax is None:
        opt.hmax = np.abs(tend - t0)
    nt = 0
    t = t0
    hmin = 16 * np.spacing(t0)
    uround = np.spacing(1.0)
    T = np.zeros((10001,))
    T[nt] = t0
    Y = np.zeros((10001, vsize))
    y0 = DaeIc(dae, y0, t0, opt.rtol)
    Y[0, :] = y0

    dense_output = False
    n_tspan = len(tspan)
    told = t0
    if n_tspan > 2:
        dense_output = True
        inext = 1
        tnext = tspan[inext]

    if opt.hinit is None:
        dt = 1e-6 * (tend - t0)
    else:
        dt = opt.hinit

    dt = np.maximum(dt, hmin)
    dt = np.minimum(dt, opt.hmax)

    M = dae.M
    p = dae.p
    linsolver = resolve_backend(getattr(opt, 'linsolver', None))
    klu_cache = KLUCache()
    state = SimpleNamespace(rscale_to_dense=None)
    done = False
    reject = 0
    while not done:
        if np.abs(dt) < uround:
            print(f"Error exit of RODAS at time = {t}: step size too small h = {dt}.\n")
            break

        if reject > 100:
            print(f"Step rejected over 100 times at time = {t}.\n")
            break

        if t + dt >= tend:
            dt = tend - t
        else:
            dt = np.minimum(dt, 0.5 * (tend - t))

        if opt.fix_h:
            dt = opt.hinit

        if reject == 0:
            J = dae.J(t, y0, p)

        try:
            ynew, err, K = legacy_step(dae, M, p, t, y0, dt, J, rparam, opt, linsolver, klu_cache, state)
        except LegacyStop:
            break
        if attempts is not None:
            attempts.append(Attempt(t, y0.copy(), dt, reject, ynew, err, J))

        if not opt.fix_h:
            err = np.maximum(err, 1.0e-6)
            fac = opt.f_savety / (err ** (1 / rparam.pord))
            fac = np.minimum(opt.facmax, np.maximum(opt.fac1, fac))
            dtnew = dt * fac
        else:
            dtnew = dt

        if err <= 1.0:
            reject = 0
            told = t
            t = t + dt

            if dense_output:
                while t >= tnext > told:
                    tau = (tnext - told) / dt
                    ynext = ntrp(y0, ynew, told, dt, opt.scheme, rparam, K, tau, dae)
                    nt = nt + 1
                    T[nt] = tnext
                    Y[nt] = ynext

                    inext = inext + 1
                    if inext <= n_tspan - 1:
                        tnext = tspan[inext]
                    else:
                        tnext = tend + dt
            else:
                nt = nt + 1
                T[nt] = t
                Y[nt] = ynew

            if nt == T.shape[0] - 1:
                T = np.concatenate([T, np.zeros(1000)])
                Y = np.concatenate([Y, np.zeros((1000, vsize))])

            if np.abs(tend - t) < uround:
                done = True
            y0 = ynew
            opt.facmax = opt.fac2

        else:
            reject = reject + 1
            opt.facmax = 1
        dt = np.min([opt.hmax, np.max([hmin, dtnew])])

    T = T[0:nt + 1]
    Y = Y[0:nt + 1]
    return T, Y
