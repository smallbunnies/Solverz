"""
Radau IIA(5) — 3-stage 5th-order fully-implicit Runge-Kutta DAE solver.

Direct port of SciML OrdinaryDiffEqFIRK ``RadauIIA5ConstantCache``
``perform_step!`` (firk_perform_step.jl, lines ~486-685), adapted
to Solverz's `nDAE` API.

For DAEs M y' = F(t, y) the implicit stage system

    M Z_i = h F(t + c_i h, y_n + Z_i),   i = 1..3

(with Z_i ≈ h * y'_i) is solved by simplified Newton iteration
on the W-coordinate system (W = TI @ Z) which decouples the
Newton update into one real and one complex (n × n) linear solve
per iteration. Stiffly accurate: y_{n+1} = y_n + Z_3.

Adaptive step size uses the SciML/Hairer error estimator: the
embedded vector ``utilde = f_n + M (E1 z1 + E2 z2 + E3 z3) / h``
is filtered through the real factorisation when ``smooth_est=True``.

References
----------
- E. Hairer, G. Wanner. *Solving Ordinary Differential Equations II*,
  2nd ed., Springer, 1996, §IV.8.
- SciML/OrdinaryDiffEq.jl, OrdinaryDiffEqFIRK module, RadauIIA5.
"""

from typing import List
import numpy as np
from scipy.sparse.linalg import splu

from Solverz.solvers.daesolver.utilities import (
    nDAE, Opt, dae_io_parser, Stats, daesol, DaeIc, getyp0, tqdm,
)

from Solverz.solvers.daesolver.radau import param as P


# Lagrange basis through (0, c1, c2, 1) for the Radau IIA(5)
# collocation polynomial. The polynomial p(theta) of degree 3 satisfies
# p(0) = 0, p(c1) = z1, p(c2) = z2, p(1) = z3, so
# y(t_n + theta * h) = y_n + p(theta), the standard SciML / Hairer
# `RADAU5` continuous extension (firk_perform_step.jl /
# CONTR5 in radau5.f). Order-4 accurate, suitable for both dense
# output and event-time refinement on accepted steps.
_C1 = P.c1
_C2 = P.c2
_DEN1 = _C1 * (_C1 - _C2) * (_C1 - 1.0)
_DEN2 = _C2 * (_C2 - _C1) * (_C2 - 1.0)
_DEN3 = (1.0 - _C1) * (1.0 - _C2)


def _radau_interp(theta, z1, z2, z3):
    """Collocation-polynomial interpolant for the Radau IIA(5) step.

    Returns ``y(t_n + theta * h) - y_n`` for ``theta`` in ``[0, 1]``.
    """
    L1 = theta * (theta - _C2) * (theta - 1.0) / _DEN1
    L2 = theta * (theta - _C1) * (theta - 1.0) / _DEN2
    L3 = theta * (theta - _C1) * (theta - _C2) / _DEN3
    return L1 * z1 + L2 * z2 + L3 * z3


def _refine_event_theta(events, t_old, h, y_old, z1, z2, z3, ie,
                        v_lo, v_hi, max_iter=50, theta_atol=1e-12):
    """Bisect to find theta in (0, 1) where event component ``ie``
    crosses zero on the accepted-step's continuous extension.

    Brackets ``v_lo`` (= event(t_old, y_old)[ie]) and ``v_hi``
    (= event(t_new, y_new)[ie]) must straddle zero on entry.
    """
    theta_lo, theta_hi = 0.0, 1.0
    for _ in range(max_iter):
        if (theta_hi - theta_lo) < theta_atol:
            break
        theta_mid = 0.5 * (theta_lo + theta_hi)
        y_mid = y_old + _radau_interp(theta_mid, z1, z2, z3)
        v_mid = events(t_old + theta_mid * h, y_mid)[0][ie]
        if v_mid == 0.0:
            return theta_mid, y_mid
        if v_lo * v_mid < 0:
            theta_hi = theta_mid
            v_hi = v_mid
        else:
            theta_lo = theta_mid
            v_lo = v_mid
    theta_event = 0.5 * (theta_lo + theta_hi)
    y_event = y_old + _radau_interp(theta_event, z1, z2, z3)
    return theta_event, y_event


@dae_io_parser
def Radau(dae: nDAE,
          tspan: List | np.ndarray,
          y0: np.ndarray,
          opt: Opt = None):
    r"""
    Radau IIA(5) DAE solver — 3-stage 5th-order fully-implicit
    Runge-Kutta method, ported from SciML's ``RadauIIA5``.

    Parameters
    ==========

    dae : nDAE
        Numerical DAE object.

    tspan : List | np.ndarray
        Either
        - a list specifying [t0, tend], or
        - a `np.ndarray` specifying the time nodes that you are concerned about

    y0 : np.ndarray
        The initial values of variables

    opt : Opt
        The solver options, including:

        - rtol: 1e-3(default)|float
            The relative error tolerance
        - atol: 1e-6(default)|float
            The absolute error tolerance
        - event: Callable
            The simulation events, with $t$ and $y$ being args,
            and `value`, `is_terminal` and `direction` being outputs
        - hinit: None(default)|float
            Initial step size
        - hmax: None(default)|float
            Maximum step size
        - max_it: 100(default)|int
            The simplified Newton maximum iteration count per stage
            attempt. SciML's Radau5 default is 10; the lower bound is
            applied automatically.
        - pbar: False(default)|bool
            To display progress bar

    Returns
    =======

    sol : daesol
        The daesol object.

    Notes
    =====

    Stiffly accurate: ``y_{n+1} = y_n + Z_3``. The Newton update is
    performed in the W-coordinate system (``W = TI @ Z``), which
    decouples the linear solves into one real and one complex sparse
    LU factorisation per step attempt. Step-size adaptation uses the
    SciML/Hairer predictive (Gustafsson) controller.

    References
    ==========

    .. [1] E. Hairer and G. Wanner, *Solving Ordinary Differential
       Equations II: Stiff and Differential-Algebraic Problems*,
       2nd ed. Berlin, Heidelberg: Springer-Verlag, 1996, §IV.8.
    .. [2] SciML/OrdinaryDiffEq.jl, OrdinaryDiffEqFIRK module,
       RadauIIA5.
    """
    if opt is None:
        opt = Opt()
    stats = Stats('Radau5')

    # SciML uses transformed tolerances for Radau5:
    rtol_user = opt.rtol
    atol_user = opt.atol
    rtol = rtol_user ** (2.0 / 3.0) / 10.0
    atol = rtol * (atol_user / rtol_user)

    # Newton maxiter — SciML's RadauIIA5 default is 10 (alg.max_iter).
    # Opt.max_it defaults to 100, which is also the value used here when
    # the caller does not override it.
    nmax_newton = getattr(opt, 'max_it', None) or 10
    kappa = 1e-2                                       # SciML κ default

    # Predictive (Gustafsson) step controller constants — SciML defaults.
    GAMMA_SAFETY = 0.9
    QMIN, QMAX = 0.2, 6.0      # max growth 5×, max shrink 1/6
    EXPO_ADAPT = 1.0 / 5.0     # 1 / (adaptive_order + 1) for Radau5
    dt_acc = 1.0
    err_acc = 1.0

    vsize = y0.shape[0]
    tspan = np.array(tspan)
    t0 = float(tspan[0])
    tend = float(tspan[-1])
    if t0 > tend:
        raise ValueError(f't0: {t0} > tend: {tend}')
    hmax = opt.hmax if opt.hmax is not None else abs(tend - t0)

    # IC consistency.
    y0 = DaeIc(dae, y0, t0, rtol_user)
    yp0 = getyp0(dae, y0, t0)

    # Initial step.
    if opt.hinit is None:
        wt = np.maximum(np.abs(y0), atol_user / rtol_user)
        rh = 1.25 * np.linalg.norm(yp0 / wt, np.inf) / rtol_user ** 0.5
        h = min(hmax, tend - t0)
        if h * rh > 1:
            h = 1 / rh
        h = max(h, 16 * np.spacing(t0))
    else:
        h = min(hmax, max(16 * np.spacing(t0), opt.hinit))

    # Output buffers.
    dense_output = len(tspan) > 2
    if dense_output:
        n_buf = max(len(tspan), 1024)
        inext = 1
    else:
        n_buf = 10001
    T_out = np.zeros(n_buf)
    Y_out = np.zeros((n_buf, vsize))
    nt = 0
    T_out[0] = t0
    Y_out[0] = y0

    # Event handling.
    events = opt.event
    have_event = events is not None
    if have_event:
        value_prev, isterm_prev, dir_prev = events(t0, y0)
        n_event = -1
        te_buf = np.zeros(10001)
        ye_buf = np.zeros((10001, vsize))
        ie_buf = np.zeros(10001, dtype=np.int64)

    pbar = tqdm(total=tend - t0) if opt.pbar else None

    M = dae.M.tocsc() if hasattr(dae.M, 'tocsc') else dae.M
    p = dae.p

    # Constants from SciML tableau.
    GAMMA = P.GAMMA
    ALPHA = P.ALPHA
    BETA = P.BETA
    T11, T12, T13 = P.T11, P.T12, P.T13
    T21, T22, T23 = P.T21, P.T22, P.T23
    T31, T32, T33 = P.T31, P.T32, P.T33
    TI11, TI12, TI13 = P.TI11, P.TI12, P.TI13
    TI21, TI22, TI23 = P.TI21, P.TI22, P.TI23
    TI31, TI32, TI33 = P.TI31, P.TI32, P.TI33
    c1, c2, c3 = P.c1, P.c2, P.c3
    E1, E2, E3 = P.E1, P.E2, P.E3

    t = t0
    y = y0.copy()
    f_n = dae.F(t, y, p)   # SciML "fsalfirst" — derivative at step start
    stats.nfeval += 1

    z1 = np.zeros(vsize)
    z2 = np.zeros(vsize)
    z3 = np.zeros(vsize)
    # Cache the last accepted-step's z values for the polynomial
    # extrapolation predictor. Without this cache, a Newton failure
    # that leaves z1/z2/z3 corrupted (e.g. NaN from a negative-sqrt
    # in F during the search) propagates into the next step's predictor.
    z1_last = np.zeros(vsize)
    z2_last = np.zeros(vsize)
    z3_last = np.zeros(vsize)
    eta_old = 1e-6
    success_iter = 0
    dt_prev = 1.0

    # SciML's RadauIIA5ConstantCache refactors J + LU on every step
    # attempt (firk_perform_step.jl line 509-516). Match that exactly
    # — no J caching.
    # Watchdog: abort if h shrinks below this for too many consecutive
    # rejections (model has a singularity Newton can't navigate).
    H_FLOOR = 1e-9
    floor_strikes = 0
    FLOOR_STRIKES_MAX = 20

    done = False
    while not done:
        if t + 1.1 * h >= tend:
            h = tend - t
            done = True
        h = min(h, hmax)
        h = max(h, 16 * np.spacing(t))
        if h <= H_FLOOR:
            floor_strikes += 1
            if floor_strikes >= FLOOR_STRIKES_MAX:
                # Give up — return whatever we have so far instead of hanging.
                if pbar is not None:
                    pbar.close()
                T_out = T_out[: nt + 1]
                Y_out = Y_out[: nt + 1]
                stats.nreject += 1
                if have_event and n_event >= 0:
                    te_buf = te_buf[: n_event + 1]
                    ye_buf = ye_buf[: n_event + 1]
                    ie_buf = ie_buf[: n_event + 1]
                    return daesol(T_out, Y_out, te_buf, ye_buf, ie_buf, stats)
                return daesol(T_out, Y_out, stats=stats)
        else:
            floor_strikes = 0

        J_cache = dae.J(t, y, p)
        stats.nJeval += 1
        gamma_dt = GAMMA / h
        alpha_dt = ALPHA / h
        beta_dt = BETA / h
        try:
            LU1 = splu((J_cache - gamma_dt * M).tocsc())
            cscale = alpha_dt + 1j * beta_dt
            Wc = (J_cache.astype(complex) - cscale * M.astype(complex)).tocsc()
            LU2 = splu(Wc)
            stats.ndecomp += 2
        except Exception:
            h *= 0.25
            done = False
            stats.nreject += 1
            continue

        # Initial Z guess. Use the cached z values from the last
        # ACCEPTED step (`z*_last`) — never the working `z*` because
        # those may have been corrupted by a NaN-producing Newton.
        if success_iter == 0:
            z1[:] = 0.0
            z2[:] = 0.0
            z3[:] = 0.0
            w1 = np.zeros(vsize)
            w2 = np.zeros(vsize)
            w3 = np.zeros(vsize)
        else:
            # Polynomial extrapolation from the LAST ACCEPTED step's z's.
            c3p = h / dt_prev
            c1p = c1 * c3p
            c2p = c2 * c3p
            c1m1 = c1 - 1
            c2m1 = c2 - 1
            c1mc2 = c1 - c2
            k1 = (z2_last - z3_last) / c2m1
            tmp = (z1_last - z2_last) / c1mc2
            k2 = (tmp - k1) / c1m1
            k3 = k2 - (tmp - z1_last / c1) / c2
            z1 = c1p * (k1 + (c1p - c2m1) * (k2 + (c1p - c1m1) * k3))
            z2 = c2p * (k1 + (c2p - c2m1) * (k2 + (c2p - c1m1) * k3))
            z3 = c3p * (k1 + (c3p - c2m1) * (k2 + (c3p - c1m1) * k3))
            w1 = TI11 * z1 + TI12 * z2 + TI13 * z3
            w2 = TI21 * z1 + TI22 * z2 + TI23 * z3
            w3 = TI31 * z1 + TI32 * z2 + TI33 * z3

        # Newton iteration.
        eta = max(eta_old, np.finfo(float).eps) ** 0.8
        ndw = 1.0
        fail_convergence = True
        k = 0
        for k in range(nmax_newton):
            ff1 = dae.F(t + c1 * h, y + z1, p)
            ff2 = dae.F(t + c2 * h, y + z2, p)
            ff3 = dae.F(t + c3 * h, y + z3, p)
            stats.nfeval += 3

            fw1 = TI11 * ff1 + TI12 * ff2 + TI13 * ff3
            fw2 = TI21 * ff1 + TI22 * ff2 + TI23 * ff3
            fw3 = TI31 * ff1 + TI32 * ff2 + TI33 * ff3

            Mw1 = M @ w1
            Mw2 = M @ w2
            Mw3 = M @ w3

            rhs1 = fw1 - gamma_dt * Mw1
            rhs2 = fw2 - alpha_dt * Mw2 + beta_dt * Mw3
            rhs3 = fw3 - beta_dt * Mw2 - alpha_dt * Mw3

            dw1 = LU1.solve(rhs1)
            dw23 = LU2.solve(rhs2 + 1j * rhs3)
            stats.nsolve += 2
            dw2 = dw23.real
            dw3 = dw23.imag

            ndw_prev = ndw
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y + z3))
            ndw = (np.linalg.norm(dw1 / scale) +
                   np.linalg.norm(dw2 / scale) +
                   np.linalg.norm(dw3 / scale)) / np.sqrt(vsize)
            # NaN-safe rejection — F can return NaN inside the Newton
            # search (e.g. negative argument to sqrt in physical
            # equations). Treat as immediate failure so the controller
            # halves h and we restart with a fresh predictor.
            if not np.isfinite(ndw):
                break

            if k > 0:
                theta = ndw / ndw_prev
                if theta > 1.0:
                    break
                if ndw * theta ** (nmax_newton - k - 1) > kappa * (1 - theta):
                    break
                eta = theta / (1 - theta)

            w1 = w1 - dw1
            w2 = w2 - dw2
            w3 = w3 - dw3
            z1 = T11 * w1 + T12 * w2 + T13 * w3
            z2 = T21 * w1 + T22 * w2 + T23 * w3
            z3 = T31 * w1 + T32 * w2 + T33 * w3      # T32=1, T33=0

            if eta * ndw < kappa and (k > 0 or ndw == 0 or success_iter > 0):
                fail_convergence = False
                break

        if fail_convergence:
            h *= 0.5
            stats.nreject += 1
            done = False
            continue

        eta_old = eta

        # Stiffly accurate update.
        y_new = y + z3
        t_new = t + h

        # Embedded error estimator (SciML formula).
        tmp = (E1 * z1 + E2 * z2 + E3 * z3) / h
        utilde = f_n + M @ tmp
        # smooth_est: filter through real LU factorisation.
        utilde = LU1.solve(utilde)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        EEst = np.linalg.norm(utilde / scale) / np.sqrt(vsize)

        # SciML retries the error estimator with a refreshed f0 if the
        # first estimate is too large on the very first step.
        if EEst >= 1.0 and stats.nstep == 0:
            f0 = dae.F(t, y + utilde, p)
            stats.nfeval += 1
            utilde = f0 + M @ tmp
            utilde = LU1.solve(utilde)
            EEst = np.linalg.norm(utilde / scale) / np.sqrt(vsize)

        if EEst <= 1.0:
            # Accept. Old-state aliases for the dense interpolant and
            # event-time bisection.
            t_old = t
            y_old = y.copy()
            t_attempted = t + h
            y_attempted = y + z3

            # Event refinement on the accepted step. The collocation
            # interpolant `_radau_interp` is order-4 accurate over
            # [t_old, t_old + h]; bisecting it locates the event time
            # to machine precision regardless of how large h is, which
            # is essential because Radau IIA(5) integrates polynomials
            # up to degree 5 exactly and the controller can therefore
            # accept very large h on smooth trajectories.
            event_truncated = False
            if have_event:
                value_attempted, isterm_attempted, dir_attempted = events(
                    t_attempted, y_attempted)
                cross = (value_prev * value_attempted < 0) & (
                    (dir_attempted == 0) |
                    ((dir_attempted > 0) & (value_attempted > value_prev)) |
                    ((dir_attempted < 0) & (value_attempted < value_prev))
                )
                if cross.any():
                    # Refine every crossing and pick the chronologically
                    # earliest one — that's the event the trajectory
                    # actually hits first.
                    crossing_idx = np.where(cross)[0]
                    best_theta = np.inf
                    best_ie = -1
                    best_y = None
                    for i in crossing_idx:
                        theta_i, y_i = _refine_event_theta(
                            events, t_old, h, y_old, z1, z2, z3,
                            int(i),
                            float(value_prev[i]),
                            float(value_attempted[i]),
                        )
                        if theta_i < best_theta:
                            best_theta = theta_i
                            best_ie = int(i)
                            best_y = y_i
                    t_event = t_old + best_theta * h
                    y_event = best_y
                    n_event += 1
                    te_buf[n_event] = t_event
                    ye_buf[n_event] = y_event
                    ie_buf[n_event] = best_ie
                    # Truncate the step at the event so subsequent
                    # integration starts from the event state.
                    t_new = t_event
                    y_new = y_event
                    event_truncated = True
                    if isterm_attempted[best_ie]:
                        done = True
                    # Re-evaluate value_prev at the truncated state for
                    # the next step's crossing check.
                    value_prev, _, _ = events(t_event, y_event)
                else:
                    t_new = t_attempted
                    y_new = y_attempted
                    value_prev = value_attempted
            else:
                t_new = t_attempted
                y_new = y_attempted

            t = t_new
            y = y_new
            f_n = dae.F(t, y, p)
            stats.nfeval += 1
            stats.nstep += 1
            if event_truncated:
                # Discard predictor warmth — the cached z values were
                # for the full step h, not the truncated theta_event*h.
                # Forcing success_iter back to 0 makes the next step
                # use a zero initial Z guess.
                success_iter = 0
            else:
                success_iter += 1
                dt_prev = h
                z1_last = z1.copy()
                z2_last = z2.copy()
                z3_last = z3.copy()

            if dense_output:
                while inext < len(tspan) and tspan[inext] <= t:
                    nt += 1
                    if nt >= n_buf:
                        T_out = np.concatenate([T_out, np.zeros(1024)])
                        Y_out = np.concatenate([Y_out, np.zeros((1024, vsize))])
                        n_buf += 1024
                    theta_dense = (tspan[inext] - t_old) / h
                    delta_dense = _radau_interp(theta_dense, z1, z2, z3)
                    T_out[nt] = tspan[inext]
                    Y_out[nt] = y_old + delta_dense
                    inext += 1
                if event_truncated and (nt < 0 or T_out[nt] != t):
                    # Make sure the event state is the last entry —
                    # the event time will rarely coincide with a tspan
                    # grid point, but the caller needs (T[-1], Y[-1])
                    # to be the bounce state for restart logic.
                    nt += 1
                    if nt >= n_buf:
                        T_out = np.concatenate([T_out, np.zeros(1024)])
                        Y_out = np.concatenate([Y_out, np.zeros((1024, vsize))])
                        n_buf += 1024
                    T_out[nt] = t
                    Y_out[nt] = y
            else:
                nt += 1
                if nt >= n_buf:
                    T_out = np.concatenate([T_out, np.zeros(1024)])
                    Y_out = np.concatenate([Y_out, np.zeros((1024, vsize))])
                    n_buf += 1024
                T_out[nt] = t
                Y_out[nt] = y

            if pbar is not None:
                pbar.update(t - t_old)

            # Predictive (Gustafsson) step controller — direct port of
            # SciML OrdinaryDiffEqCore's `PredictiveController` for FIRK
            # methods (controllers.jl).
            niter = k + 1
            fac = min(GAMMA_SAFETY,
                      (1.0 + 2.0 * nmax_newton) * GAMMA_SAFETY
                      / (niter + 2.0 * nmax_newton))
            q_tmp = max(EEst, 1e-10) ** EXPO_ADAPT / fac
            q = max(1.0 / QMAX, min(1.0 / QMIN, q_tmp))
            if success_iter > 1:
                q_gus = (dt_acc / h) * ((EEst ** 2) / err_acc) ** EXPO_ADAPT
                q_gus = max(1.0 / QMAX,
                            min(1.0 / QMIN, q_gus / GAMMA_SAFETY))
                q_acc = max(q, q_gus)
            else:
                q_acc = q
            # Save state of this accepted step BEFORE recomputing h.
            dt_acc = h
            err_acc = max(0.01, EEst)
            h = h / q_acc
        else:
            stats.nreject += 1
            # On rejection SciML also uses the Predictive controller's
            # result: reject uses the same q from stepsize_controller!.
            niter = k + 1
            fac = min(GAMMA_SAFETY,
                      (1.0 + 2.0 * nmax_newton) * GAMMA_SAFETY
                      / (niter + 2.0 * nmax_newton))
            q_tmp = max(EEst, 1e-10) ** EXPO_ADAPT / fac
            q = max(1.0, min(1.0 / QMIN, q_tmp))  # reject → shrink only
            h = h / q
            done = False

    if pbar is not None:
        pbar.close()

    T_out = T_out[: nt + 1]
    Y_out = Y_out[: nt + 1]
    if have_event and n_event >= 0:
        te_buf = te_buf[: n_event + 1]
        ye_buf = ye_buf[: n_event + 1]
        ie_buf = ie_buf[: n_event + 1]
        return daesol(T_out, Y_out, te_buf, ye_buf, ie_buf, stats)
    return daesol(T_out, Y_out, stats=stats)
