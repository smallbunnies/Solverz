"""
Partitioned-explicit (PE) DAE solver, with a modified-Euler variant.

This implements the partitioned-explicit integration scheme of Sauer, Pai
and Chow, *Power System Dynamics and Stability* (2nd ed., Wiley/IEEE Press,
2018), Chapter 7, Section 7.7.5. The differential-algebraic model is written
in the Solverz semi-explicit form

    M y' = F(t, y),      M singular (diagonal),

so the rows where ``M`` has a nonzero diagonal are the *differential*
equations and their paired columns are the *differential* (state) variables
``x``; the rows where ``M`` is zero are the *algebraic* (network/interface)
equations and their paired columns are the *algebraic* variables ``z``. The
PE method advances the two blocks in an alternating, partitioned fashion:

  1. Hold the algebraic variables fixed and step the differential states
     *explicitly* (no implicit coupling of x_{n+1} to its own RHS).
  2. Hold the freshly stepped states fixed and *solve the algebraic block*
     0 = F_a(t_{n+1}, x_{n+1}, z_{n+1}) for z_{n+1} by Newton iteration
     (the "network/interface solution").

This is the book's PE loop (7.156)-(7.158): integrate ``x' = f(x, I_dq, V)``
to obtain ``x(n+1)``, then re-solve the algebraic equations for the new
``I_dq(n+1), V(n+1)``.

Two explicit integrators are provided through ``Opt(scheme=...)``:

  * ``scheme='euler'``           : forward (explicit) Euler.  The basic PE.
  * ``scheme='modified_euler'``  : Heun's improved/modified-Euler predictor
    (default).  This is the "improved-Euler version" of the PE method:

        x_p      = x_n + h f(t_n, x_n, z_n)          (predictor)
        z_p      : solve  F_a(t_{n+1}, x_p, z_p) = 0 (network solve)
        x_{n+1}  = x_n + (h/2)[f(t_n, x_n, z_n) + f(t_{n+1}, x_p, z_p)]
        z_{n+1}  : solve  F_a(t_{n+1}, x_{n+1}, z_{n+1}) = 0

The PE method is a *fixed-step* method: the step is ``opt.step_size`` (it has
no embedded error estimate, hence no per-step tolerance). Its maximum stable
step is set by the fastest mode of the explicit differential block; on a
method-of-lines gas network plus electromechanical machine dynamics this is a
CFL-type limit, so a very small fixed step is required across the whole
horizon. That is precisely the property this baseline is meant to expose.

Robustness for very long horizons (the cascade horizon is tens of thousands
of seconds, so a small fixed step yields 1e6-1e8 internal steps):

  * Memory is bounded. The solution is reported on the requested ``tspan``
    nodes when ``len(tspan) > 2`` (dense-output style), or on a uniform grid
    of ``opt`` -capped size when ``len(tspan) == 2``. The internal fixed-step
    march is never stored in full. Within-step reporting uses linear
    interpolation, consistent with the method's first/second order.
  * Divergence is detected. A non-finite state, or a failure of the
    algebraic Newton to converge, terminates the run with ``stats.ret =
    'failed'`` and returns the trajectory accumulated so far rather than
    raising.
  * The algebraic Newton uses a held-Jacobian factorization of the
    algebraic Jacobian block J_aa = (dF_a/dz), refreshed only when the inner
    iteration stalls or fails, following Section 7.8.1 of the reference. This
    keeps the per-step cost dominated by F evaluations and back-substitutions
    rather than by a fresh sparse LU at every micro-step.

Event handling mirrors the other Solverz solvers: an optional ``opt.event``
callable ``g(t, y) -> (value, isterminal, direction)`` is checked after every
accepted step; a sign change is located within the step by a secant +
bisection root finder on a *linear* interpolant of the step, and a terminal
event stops the integration and is reported through ``daesol.te/ye/ie``.
"""

import warnings

import numpy as np
from numpy import linalg
from scipy.sparse import csc_matrix, issparse
from tqdm import tqdm

from Solverz.solvers.daesolver.utilities import *


# Default number of uniform report nodes when tspan has only [t0, tend]
# (a 2-point, event-terminated phase). Bounds memory regardless of how many
# internal fixed steps the PE march takes.
_DEFAULT_NREPORT = 10001
# Inner algebraic-Newton controls.
_ALG_MAXIT = 30          # max Newton iterations for the algebraic block
_ALG_REFRESH_ITERS = 4   # refresh the held J_aa if the inner solve needs more


@dae_io_parser
def PE(dae: nDAE,
       tspan,
       y0,
       opt: Opt = None):
    """Partitioned-explicit DAE solver. See the module docstring.

    Parameters
    ----------
    dae : nDAE
        Semi-explicit DAE ``M y' = F(t, y)`` with singular diagonal ``M``.
    tspan : array-like
        ``[t0, tend]`` for raw reporting on a capped uniform grid, or a
        length >= 3 array of report instants for dense-output reporting.
    y0 : ndarray
        Initial state (made consistent with the algebraic constraints by
        ``DaeIc`` before the march starts).
    opt : Opt
        ``step_size`` sets the fixed step. ``scheme`` selects ``'euler'`` or
        ``'modified_euler'`` (default). ``event``, ``pbar`` as usual.
    """
    if opt is None:
        opt = Opt()
    scheme = (getattr(opt, 'scheme', None) or 'modified_euler').lower()
    if scheme in ('me', 'meuler', 'heun', 'improved_euler', 'modified-euler'):
        scheme = 'modified_euler'
    if scheme in ('fe', 'feuler', 'forward_euler'):
        scheme = 'euler'
    if scheme not in ('euler', 'modified_euler'):
        # Any unrelated scheme name (e.g. a rodas/NDF default) maps to the
        # improved-Euler PE so a shared Opt across solvers still runs.
        scheme = 'modified_euler'
    stats = Stats(scheme=f'PE/{scheme}')

    vsize = y0.shape[0]
    tspan = np.asarray(tspan, dtype=float)
    t0 = float(tspan[0])
    tend = float(tspan[-1])
    if t0 > tend:
        raise ValueError(f't0 {t0} > tend {tend}')

    h = float(getattr(opt, 'step_size', None) or 1e-3)
    if h <= 0:
        raise ValueError(f'PE needs a positive step_size, got {h}')

    # --- differential / algebraic partition from the mass matrix ---------
    # The Solverz mass matrix M is in general NOT diagonal: each
    # differential equation is one off-diagonal entry M[r, c] coupling
    # equation row r to its state-variable column c, i.e. the differential
    # equation reads  M[r, c] * d(y[c])/dt = F[r].  The differential
    # equation ROWS (set of r) and the differential variable COLUMNS (set
    # of c) are therefore distinct index sets, and the algebraic block must
    # be sliced with the distinct masks J[alg_eqn_rows, alg_var_cols] -- the
    # index-matched J[alg_var, alg_var] block is singular for this model.
    M = dae.M if issparse(dae.M) else csc_matrix(dae.M)
    Mcoo = M.tocoo()
    diff_rows = Mcoo.row.astype(int)   # differential equation indices (r)
    diff_cols = Mcoo.col.astype(int)   # paired state-variable indices (c)
    diff_vals = Mcoo.data.astype(float)  # mass coefficients M[r, c]
    if np.any(np.abs(diff_vals) < 1e-30):
        diff_vals = np.where(np.abs(diff_vals) < 1e-30, 1.0, diff_vals)
    n_diff = diff_cols.size

    DiffVarMask = np.zeros(vsize, dtype=bool)
    DiffVarMask[diff_cols] = True
    AlgVarMask = ~DiffVarMask
    DiffEqnMask = np.zeros(vsize, dtype=bool)
    DiffEqnMask[diff_rows] = True
    AlgEqnMask = ~DiffEqnMask
    alg_eqn_rows = np.where(AlgEqnMask)[0]
    alg_var_cols = np.where(AlgVarMask)[0]
    n_alg = alg_var_cols.size
    if n_diff == 0:
        warnings.warn('PE: no differential rows; the model is purely algebraic.')

    def f_diff(F_full):
        """Explicit differential RHS dy[c]/dt = F[r] / M[r, c], returned in
        the order of ``diff_cols`` (the differential state variables)."""
        return F_full[diff_rows] / diff_vals

    # --- consistent initial state ---------------------------------------
    y0 = DaeIc(dae, y0, t0, opt.rtol)

    # --- output buffers --------------------------------------------------
    dense_output = len(tspan) > 2
    if dense_output:
        report_t = tspan.copy()
    else:
        report_t = np.linspace(t0, tend, _DEFAULT_NREPORT)
    n_report = report_t.size
    T = np.zeros(n_report)
    Y = np.zeros((n_report, vsize))
    T[0] = t0
    Y[0] = y0
    nt = 0
    inext = 1
    tnext = report_t[inext] if n_report > 1 else tend + h

    threshold = opt.atol / opt.rtol
    uround = np.spacing(1.0)

    # --- event support ---------------------------------------------------
    events_fn = getattr(opt, 'event', None)
    have_events = events_fn is not None
    if have_events:
        prev_vals, isterminal, direction = events_fn(t0, y0)
        prev_vals = np.asarray(prev_vals, dtype=float)
        direction = np.asarray(direction, dtype=float)
        te = np.zeros(1001)
        ye = np.zeros((1001, vsize))
        ie = np.zeros(1001, dtype=int)
        nevent = -1

    # --- held algebraic Jacobian factorization (VDHN) --------------------
    alg_state = {'lu': None}

    def refresh_alg_lu(t, y):
        J = dae.J(t, y, dae.p)
        stats.nJeval += 1
        J = (J if issparse(J) else csc_matrix(J)).tocsc()
        Jaa = J[alg_eqn_rows][:, alg_var_cols].tocsc()
        alg_state['lu'] = lu_decomposition(Jaa)
        stats.ndecomp += 1

    def solve_alg(t, y_with_diff_fixed):
        """Solve F_a(t, y) = 0 over the algebraic variables y[alg_var_cols]
        by held-Jacobian Newton, with the differential variables held at the
        values already written into ``y_with_diff_fixed``.

        Returns (y, F_full, ok). On non-convergence ok=False (caller aborts).
        The full residual at the converged point is returned so the caller
        can extract the differential RHS without a re-evaluation.
        """
        y = y_with_diff_fixed  # modified in place on the algebraic entries
        if n_alg == 0:
            F_full = dae.F(t, y, dae.p)
            stats.nfeval += 1
            return y, F_full, True
        if alg_state['lu'] is None:
            refresh_alg_lu(t, y)
        refreshed_here = False
        F_full = None
        for it in range(1, _ALG_MAXIT + 1):
            F_full = dae.F(t, y, dae.p)
            stats.nfeval += 1
            res = F_full[alg_eqn_rows]
            wt = np.maximum(np.abs(y[alg_var_cols]), threshold)
            del_z = alg_state['lu'].solve(-res)
            stats.nsolve += 1
            y[alg_var_cols] += del_z
            relnorm = linalg.norm(del_z / wt, np.inf)
            if not np.all(np.isfinite(y[alg_var_cols])):
                return y, F_full, False
            if relnorm < opt.ite_tol:
                # Refresh the held factorization if convergence was slow, so
                # the next step starts from a current Jaa (VDHN heuristic).
                if it > _ALG_REFRESH_ITERS:
                    refresh_alg_lu(t, y)
                return y, F_full, True
            # Stalled: refresh the Jacobian once and keep iterating.
            if it == _ALG_REFRESH_ITERS and not refreshed_here:
                refresh_alg_lu(t, y)
                refreshed_here = True
        return y, F_full, False

    # --- initial split + first RHS --------------------------------------
    refresh_alg_lu(t0, y0)
    F_n = dae.F(t0, y0, dae.p)
    stats.nfeval += 1
    f_n = f_diff(F_n)

    t = t0
    y_n = y0.copy()
    pbar = tqdm(total=tend - t0) if opt.pbar else None
    done = False
    stopped_by_event = False

    while not done:
        # Final partial step lands exactly on tend.
        h_step = h
        if t + h_step >= tend - 1e-12 * max(abs(tend), 1.0):
            h_step = tend - t
            done = True
        if h_step <= 0:
            break

        t_new = t + h_step

        # ---- explicit differential update ------------------------------
        # Start the new state from the previous one: algebraic entries seed
        # the network Newton, differential entries are overwritten below.
        if scheme == 'euler':
            y_new = y_n.copy()
            y_new[diff_cols] = y_n[diff_cols] + h_step * f_n
            y_new, F_new, ok = solve_alg(t_new, y_new)
            if not ok:
                _fail(stats, t_new, 'algebraic Newton diverged (euler step)')
                break
        else:  # modified_euler (Heun)
            y_p = y_n.copy()
            y_p[diff_cols] = y_n[diff_cols] + h_step * f_n
            y_p, F_p, ok = solve_alg(t_new, y_p)
            if not ok:
                _fail(stats, t_new, 'algebraic Newton diverged (predictor)')
                break
            f_p = f_diff(F_p)
            y_new = y_p.copy()
            y_new[diff_cols] = y_n[diff_cols] + 0.5 * h_step * (f_n + f_p)
            y_new, F_new, ok = solve_alg(t_new, y_new)
            if not ok:
                _fail(stats, t_new, 'algebraic Newton diverged (corrector)')
                break

        if not np.all(np.isfinite(y_new)):
            _fail(stats, t_new, 'non-finite state')
            break

        stats.nstep += 1

        # ---- reporting on the requested grid (linear within step) ------
        while inext < n_report and t < tnext <= t_new + 1e-12:
            s = (tnext - t) / h_step if h_step > 0 else 1.0
            y_at = y_n + s * (y_new - y_n)
            nt += 1
            T[nt] = tnext
            Y[nt] = y_at
            inext += 1
            tnext = report_t[inext] if inext < n_report else tend + h
        if pbar:
            pbar.update(h_step)

        # ---- event detection -------------------------------------------
        if have_events:
            new_vals, new_isterm, new_dir, hits = _locate_events_linear(
                events_fn, t, t_new, y_n, y_new, prev_vals, direction,
                uround, opt.event_duration)
            if hits:
                for (te_i, ye_i, ie_i, term_i) in hits:
                    nevent += 1
                    if nevent == te.size:
                        te = np.concatenate([te, np.zeros(1000)])
                        ye = np.concatenate([ye, np.zeros((1000, vsize))])
                        ie = np.concatenate([ie, np.zeros(1000, dtype=int)])
                    te[nevent] = te_i
                    ye[nevent] = ye_i
                    ie[nevent] = ie_i
                last_te, last_ye, _, last_term = hits[-1]
                if last_term:
                    # Land the final reported sample on the event.
                    nt += 1
                    if nt >= T.size:
                        T = np.concatenate([T, np.zeros(1000)])
                        Y = np.concatenate([Y, np.zeros((1000, vsize))])
                    T[nt] = last_te
                    Y[nt] = last_ye
                    t = last_te
                    y_n = last_ye
                    stopped_by_event = True
                    done = True
            prev_vals = new_vals
            direction = new_dir
            if stopped_by_event:
                break

        # ---- roll state ------------------------------------------------
        t = t_new
        y_n = y_new
        F_n = F_new
        f_n = f_diff(F_n)

    if pbar:
        pbar.close()

    # Trim the unused dense-output tail (the run may have stopped early at
    # an event or a divergence before filling the whole report grid).
    T = T[:nt + 1]
    Y = Y[:nt + 1]
    if stats.ret != 'failed':
        stats.succeed = True

    if have_events and nevent >= 0:
        te = te[:nevent + 1]
        ye = ye[:nevent + 1]
        ie = ie[:nevent + 1]
        return daesol(T, Y, te=te, ye=ye, ie=ie, stats=stats)
    return daesol(T, Y, stats=stats)


def _fail(stats, t, reason):
    stats.ret = 'failed'
    stats.succeed = False
    print(f'PE: integration failed at t = {t:.6g}: {reason}.')


def _locate_events_linear(events_fn, t_n, t_new, y_n, y_new, prev_vals,
                          direction, uround, event_dur):
    """Locate sign changes of events(t, y) on (t_n, t_new] using a *linear*
    interpolant y(s) = y_n + s (y_new - y_n), s in [0, 1]. Returns
    (new_vals, new_isterminal, new_direction, hits) where hits is a
    time-ordered list of (t_e, y_e, idx, isterminal).
    """
    new_vals_raw, new_isterm, new_dir = events_fn(t_new, y_new)
    new_vals = np.asarray(new_vals_raw, dtype=float)
    new_isterm = np.asarray(new_isterm)
    new_dir = np.asarray(new_dir, dtype=float)
    h = t_new - t_n
    hits = []

    cross = np.where(prev_vals * new_vals < 0)[0]
    for i in cross:
        v0 = prev_vals[i]
        v1 = new_vals[i]
        if direction[i] < 0 and v0 <= v1:
            continue
        if direction[i] > 0 and v0 >= v1:
            continue
        tL, tR = t_n, t_new
        if abs(v1 - v0) > uround:
            t_e = t_n - v0 * h / (v1 - v0)
        else:
            t_e = t_new
        tol = max(128.0 * uround, event_dur)
        y_e = y_n + ((t_e - t_n) / h) * (y_new - y_n)
        for _it in range(60):
            s = (t_e - t_n) / h
            y_e = y_n + s * (y_new - y_n)
            v_now = np.asarray(events_fn(t_e, y_e)[0], dtype=float)[i]
            if v1 * v_now < 0:
                tL = t_e
                v0 = v_now
                t_e = 0.5 * (t_e + tR)
            elif v0 * v_now < 0:
                tR = t_e
                v1 = v_now
                t_e = 0.5 * (tL + t_e)
            else:
                break
            if (tR - tL) < tol:
                break
        if t_e - t_n < event_dur:
            continue
        hits.append((t_e, y_e, int(i), bool(new_isterm[i])))

    hits.sort(key=lambda r: r[0])
    return new_vals, new_isterm, new_dir, hits
