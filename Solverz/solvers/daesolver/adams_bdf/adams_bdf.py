"""
Mixed Adams-BDF variable-step variable-order solver, Nordsieck form
(Astic, Bihain, Jerosolimski, IEEE TPWRS 9(2):929-935, 1994).

Paper-faithful implementation: it carries the Nordsieck vector and realizes
the paper's defining "mixed" treatment, in which the Adams method governs the
differential state variables and the BDF method governs the algebraic state
variables. The two enter through the Nordsieck l-vector used in the corrector
update and, critically, in the local-error estimate: the differential-variable
error uses the Adams l-coefficient and the algebraic-variable error uses the
BDF l-coefficient, both fed into the error norm. This restores the
algebraic-variable error control that the earlier direct-history
implementation omitted.

Nordsieck convention z_j = h^j y^{(j)} / j!, so for order q the array is

    Z = [ y , h y' , (h^2/2) y'' ]   (rows 0..q used).

Predictor: Z_pred = P Z, P the Pascal (binomial) upper-triangular matrix.
Step change by alpha = h_new/h_old: rescale Z_j <- alpha^j Z_j. Corrector,
by Newton:

    differential equation rows r (paired variable column c via M[r,c]):
        h F_r(y_{n+1}) = M[r,c]( Z_pred[1,c] + l1 (y_{n+1,c} - Z_pred[0,c]) )
    algebraic equation rows r:
        F_r(y_{n+1}) = 0                                   (paper eq. 3.b)

l1 is the Adams leading coefficient (1 at order 1, 2 at order 2 = trapezoidal).
After convergence with correction e = y_{n+1} - Z_pred[0], the Nordsieck array
is updated component-wise with the Adams l-vector on differential columns and
the BDF l-vector on algebraic columns.

l-vectors, convention z_j = h^j y^{(j)}/j! (l0 = 1, l1 = 1/beta0):
    order 1 (Adams1 = BDF1 = implicit Euler):  l = [1, 1]
    order 2 Adams (trapezoidal, differential): l = [1, 2,   1  ]
    order 2 BDF  (algebraic):                  l = [1, 3/2, 1/2]

Local error at order q: E_i = l_q^{method(i)} e_i, Adams l_q on differential,
BDF l_q on algebraic, then a max norm (the paper's anti-hyper-stability norm).
At order 2 l_2^{BDF}=1/2 < l_2^{Adams}=1, so the algebraic-variable error is
less restrictive, exactly the paper's reason for BDF on algebraic variables.

Retained paper features: order capped at 2; Very-Dishonest Newton; step-and-
order hold for >= q+1 steps; 10% step-increase acceptance hysteresis; self-
start at order 1; re-init to order 1 with the minimum step on any event;
optional ADAMSBDF_TRACE per-step trace.
"""

import os
import warnings

import numpy as np
from numpy import linalg
from scipy.sparse import diags_array, csc_matrix, issparse
from tqdm import tqdm

from Solverz.solvers.daesolver.utilities import *

_TRACE = []
_TRACE_ON = os.environ.get('ADAMSBDF_TRACE', '0') == '1'

# Nordsieck l-vectors, convention z_j = h^j y^{(j)}/j!, l0 = 1, l1 = 1/beta0.
_L_EULER = np.array([1.0, 1.0, 0.0])
_L_ADAMS2 = np.array([1.0, 2.0, 1.0])
_L_BDF2 = np.array([1.0, 1.5, 0.5])


@dae_io_parser
def AdamsBDF(dae: nDAE, tspan, y0, opt: Opt = None):
    """Nordsieck-form mixed Adams-BDF solver. See module docstring."""
    if opt is None:
        opt = Opt()
    stats = Stats(scheme='AdamsBDF-Nordsieck')
    if _TRACE_ON:
        _TRACE.clear()

    vsize = y0.shape[0]
    tspan = np.asarray(tspan, dtype=float)
    t0, tend = float(tspan[0]), float(tspan[-1])
    if t0 > tend:
        raise ValueError(f't0 {t0} > tend {tend}')
    if opt.hmax is None:
        opt.hmax = abs(tend - t0)

    M = (dae.M if issparse(dae.M) else csc_matrix(dae.M)).tocsc()
    Mcoo = M.tocoo()
    diff_rows = Mcoo.row.astype(int)
    diff_cols = Mcoo.col.astype(int)
    diff_vals = np.where(np.abs(Mcoo.data) < 1e-30, 1.0, Mcoo.data.astype(float))
    DiffVar = np.zeros(vsize, dtype=bool)
    DiffVar[diff_cols] = True
    AlgVar = ~DiffVar
    DiffEqn = np.zeros(vsize, dtype=bool)
    DiffEqn[diff_rows] = True
    AlgEqn = ~DiffEqn
    n_diff = int(DiffVar.sum())
    has_alg = bool(AlgVar.any())
    if n_diff == 0:
        warnings.warn('AdamsBDFns: no differential rows; degenerates to Newton.')

    dense_output = len(tspan) > 2
    n_buf = len(tspan) if dense_output else 10001
    if dense_output:
        inext = 1
        tnext = float(tspan[inext])
    T = np.zeros(n_buf)
    Y = np.zeros((n_buf, vsize))
    stats.order_variation = np.zeros(n_buf)

    y0 = DaeIc(dae, y0, t0, opt.rtol)
    yp0 = getyp0(dae, y0, t0)
    T[0], Y[0] = t0, y0
    nt = 0

    threshold = opt.atol / opt.rtol
    uround = np.spacing(1.0)
    hmin = 16.0 * np.spacing(t0)

    if opt.hinit is None:
        wt0 = np.maximum(np.abs(y0), threshold)
        rh = 1.25 * linalg.norm(yp0 / wt0, np.inf) / opt.rtol ** 0.5
        absh = min(opt.hmax, tend - t0)
        if absh * rh > 1.0:
            absh = 1.0 / rh
        absh = max(absh, hmin)
    else:
        absh = min(opt.hmax, max(hmin, opt.hinit))

    Z = np.zeros((3, vsize))
    Z[0] = y0
    Z[1] = absh * yp0

    t = t0
    order = 1
    stats.order_variation[0] = order
    h = absh
    n_at_curr = 0

    J_cached = _eval_J(dae, t, y0, vsize, getattr(opt, 'numJac', False))
    stats.nJeval += 1
    Jcurrent = True
    lu = rscale = None
    cached_h = cached_order = None

    maxit = max(getattr(opt, 'ndfmaxit', 4) or 4, 8)
    pbar = tqdm(total=tend - t0) if opt.pbar else None

    events_fn = getattr(opt, 'event', None)
    have_events = events_fn is not None
    if have_events:
        prev_ev, _, direction = events_fn(t0, y0)
        prev_ev = np.asarray(prev_ev, dtype=float)
        direction = np.asarray(direction, dtype=float)
        te = np.zeros(1001)
        ye = np.zeros((1001, vsize))
        ie = np.zeros(1001, dtype=int)
        nevent = -1
        force_reinit = False

    done = stopped_by_event = False
    while not done:
        hmin = 16.0 * np.spacing(t)
        absh = max(hmin, min(opt.hmax, absh))

        # Rescale the Nordsieck array if the step changed.
        if abs(absh - h) > 1e-13 * max(h, 1.0):
            alpha = absh / h
            Z[1] *= alpha
            Z[2] *= alpha * alpha
            h = absh

        if 1.1 * absh >= abs(tend - t):
            new_h = tend - t
            if new_h > 0 and abs(new_h - h) > 1e-13 * max(h, 1.0):
                alpha = new_h / h
                Z[1] *= alpha
                Z[2] *= alpha * alpha
            h = absh = new_h
            done = True

        K = order
        if have_events and force_reinit:
            K = order = 1
            absh = max(hmin, min(absh, 1e-3 * max(opt.hmax, 1e-3)))
            h = absh
            Z[1] = h * _yp(dae.F(t, Z[0], dae.p), diff_rows, diff_cols, diff_vals)
            stats.nfeval += 1
            Z[2] = 0.0
            n_at_curr = 0
            force_reinit = False
            cached_h = None
            done = (1.1 * absh >= abs(tend - t))

        l1 = _L_ADAMS2[1] if K == 2 else _L_EULER[1]

        if (cached_h is None or abs(cached_h - h) > 1e-12 * max(h, 1.0)
                or cached_order != K):
            lu, rscale = _build_lu(M, h, J_cached, l1, DiffEqn)
            stats.ndecomp += 1
            cached_h, cached_order = h, K

        y_n = Z[0].copy()
        yp_n = Z[1] / h if h > 0 else np.zeros(vsize)
        Zp = _predict(Z, K)
        y_pred = Zp[0].copy()
        z1_pred = Zp[1]

        # Newton.
        y_new = y_pred.copy()
        success = False
        oldnrm = None
        for _ in range(maxit):
            F_new = dae.F(t + h, y_new, dae.p)
            stats.nfeval += 1
            e = y_new - y_pred
            G = np.empty(vsize)
            G[DiffEqn] = (h * F_new[DiffEqn] - (M @ (z1_pred + l1 * e))[DiffEqn])
            G[AlgEqn] = F_new[AlgEqn]
            del_y = lu.solve(rscale * (-G))
            stats.nsolve += 1
            y_new += del_y
            wt = np.maximum(np.maximum(np.abs(y_n), np.abs(y_new)), threshold)
            newnrm = linalg.norm(del_y / wt, np.inf)
            if newnrm < 1e-10:
                success = True
                break
            if oldnrm is not None:
                rate = newnrm / max(oldnrm, 1e-30)
                if rate >= 0.9:
                    break
                if newnrm * rate / max(1.0 - rate, 1e-30) < 0.1 * opt.rtol:
                    success = True
                    break
            oldnrm = newnrm

        if not success:
            stats.nreject += 1
            if not Jcurrent:
                J_cached = _eval_J(dae, t, y_n, vsize, getattr(opt, 'numJac', False))
                stats.nJeval += 1
                Jcurrent = True
                cached_h = None
                done = False
                continue
            absh = max(0.3 * absh, hmin)
            if absh <= hmin:
                print(f'AdamsBDFns: step too small at t={t}.')
                stats.ret = 'failed'
                break
            cached_h = None
            n_at_curr = 0
            done = False
            continue

        e = y_new - y_pred
        wt = np.maximum(np.maximum(np.abs(y_n), np.abs(y_new)), threshold)
        lq_A = (_L_ADAMS2 if K == 2 else _L_EULER)[K]
        lq_B = (_L_BDF2 if K == 2 else _L_EULER)[K]
        err_vec = np.empty(vsize)
        err_vec[DiffVar] = lq_A * e[DiffVar]
        err_vec[AlgVar] = lq_B * e[AlgVar]
        err = linalg.norm(err_vec / wt, np.inf)

        if _TRACE_ON:
            _TRACE.append((float(t), float(h), int(K), float(err),
                           bool(err <= opt.rtol)))

        if err > opt.rtol:
            stats.nreject += 1
            if K == 2:
                order = 1
                n_at_curr = 0
            absh = max(0.1 * absh,
                       0.8 * absh * (opt.rtol / max(err, 1e-30)) ** (1.0 / (K + 1)))
            if absh <= hmin:
                print(f'AdamsBDFns: step too small after rejection at t={t}.')
                stats.ret = 'failed'
                break
            cached_h = None
            Jcurrent = False
            n_at_curr = 0
            done = False
            continue

        # Accept: Nordsieck update with mixed l-vectors.
        lA = _L_ADAMS2 if K == 2 else _L_EULER
        lB = _L_BDF2 if K == 2 else _L_EULER
        z1_old = Z[1].copy()
        Znew = Zp.copy()
        Znew[0] = y_new
        Znew[1, DiffVar] = Zp[1, DiffVar] + lA[1] * e[DiffVar]
        Znew[1, AlgVar] = Zp[1, AlgVar] + lB[1] * e[AlgVar]
        if K == 2:
            Znew[2, DiffVar] = Zp[2, DiffVar] + lA[2] * e[DiffVar]
            Znew[2, AlgVar] = Zp[2, AlgVar] + lB[2] * e[AlgVar]
        else:
            Znew[2] = 0.5 * (Znew[1] - z1_old)
        Z = Znew

        stats.nstep += 1
        t_new = t + h
        n_at_curr += 1
        yp_new = _yp(dae.F(t_new, y_new, dae.p), diff_rows, diff_cols, diff_vals)
        stats.nfeval += 1

        if dense_output:
            while inext < len(tspan) and t_new >= tnext > t:
                s = (tnext - t) / h
                nt += 1
                if nt >= T.shape[0]:
                    T = np.concatenate([T, np.zeros(1000)])
                    Y = np.concatenate([Y, np.zeros((1000, vsize))])
                    stats.order_variation = np.concatenate(
                        [stats.order_variation, np.zeros(1000)])
                T[nt] = tnext
                Y[nt] = _hermite(s, y_n, y_new, yp_n, yp_new, h)
                stats.order_variation[nt] = K
                if pbar:
                    pbar.update(T[nt] - T[nt - 1])
                inext += 1
                tnext = float(tspan[inext]) if inext < len(tspan) else tend + h
        else:
            nt += 1
            if nt >= T.shape[0]:
                T = np.concatenate([T, np.zeros(1000)])
                Y = np.concatenate([Y, np.zeros((1000, vsize))])
                stats.order_variation = np.concatenate(
                    [stats.order_variation, np.zeros(1000)])
            T[nt], Y[nt] = t_new, y_new
            stats.order_variation[nt] = K
            if pbar:
                pbar.update(T[nt] - T[nt - 1])

        if have_events:
            new_ev, _, new_dir, hits = _locate_events(
                events_fn, t, t_new, y_n, y_new, yp_n, yp_new, h,
                prev_ev, direction, uround, opt.event_duration)
            if hits:
                for (te_i, ye_i, ie_i, term_i) in hits:
                    nevent += 1
                    if nevent == te.size:
                        te = np.concatenate([te, np.zeros(1000)])
                        ye = np.concatenate([ye, np.zeros((1000, vsize))])
                        ie = np.concatenate([ie, np.zeros(1000, dtype=int)])
                    te[nevent], ye[nevent], ie[nevent] = te_i, ye_i, ie_i
                force_reinit = True
                last_te, last_ye, _, last_term = hits[-1]
                if last_term:
                    t_new, y_new = last_te, last_ye.copy()
                    done = stopped_by_event = True
                    if dense_output:
                        # Drop grid points reported beyond the event, then
                        # APPEND the exact event point so T[-1] is the true
                        # crossing time (not the last grid point before it).
                        while nt > 0 and T[nt] > t_new:
                            nt -= 1
                        nt += 1
                        if nt >= T.shape[0]:
                            T = np.concatenate([T, np.zeros(1000)])
                            Y = np.concatenate([Y, np.zeros((1000, vsize))])
                            stats.order_variation = np.concatenate(
                                [stats.order_variation, np.zeros(1000)])
                        T[nt], Y[nt] = t_new, y_new
                        stats.order_variation[nt] = K
                    else:
                        T[nt], Y[nt] = t_new, y_new
            prev_ev, direction = new_ev, new_dir

        # Order + step selection (order <= 2, 1-vs-2 by projected step).
        z2 = Z[2]
        err_o1 = 0.0
        if n_diff:
            err_o1 = max(err_o1, linalg.norm(z2[DiffVar] / wt[DiffVar], np.inf))
        if has_alg:
            err_o1 = max(err_o1, linalg.norm(z2[AlgVar] / wt[AlgVar], np.inf))
        h_o1 = absh * max(0.1, 0.9 * (opt.rtol / max(err_o1, 1e-30)) ** 0.5)
        if K == 2:
            h_o2 = absh * max(0.1, 0.9 * (opt.rtol / max(err, 1e-30)) ** (1.0 / 3.0))
        else:
            h_o2 = 1.5 * h_o1  # bias toward order 2 after self-start
        new_order = 2 if h_o2 >= h_o1 else 1
        absh_cand = max(0.5 * absh, min(2.0 * absh, max(h_o1, h_o2)))

        if n_at_curr >= K + 1:
            order_changed = new_order != order
            h_changed = abs(absh_cand - absh) >= 0.1 * absh
            if order_changed:
                if new_order == 1:
                    Z[2] = 0.5 * (Z[1] - z1_old)
                order = new_order
                n_at_curr = 0
            if h_changed:
                absh = absh_cand
                if not order_changed:
                    n_at_curr = 0

        t = t_new
        Jcurrent = False
        if stopped_by_event:
            break

    T, Y = T[:nt + 1], Y[:nt + 1]
    stats.order_variation = stats.order_variation[:nt + 1]
    if pbar:
        pbar.close()
    if stats.ret != 'failed':
        stats.succeed = True
    if have_events and nevent >= 0:
        return daesol(T, Y, te=te[:nevent + 1], ye=ye[:nevent + 1],
                      ie=ie[:nevent + 1], stats=stats)
    return daesol(T, Y, stats=stats)


def _predict(Z, K):
    Zp = Z.copy()
    if K == 1:
        Zp[0] = Z[0] + Z[1]
        Zp[1] = Z[1]
    else:
        Zp[0] = Z[0] + Z[1] + Z[2]
        Zp[1] = Z[1] + 2.0 * Z[2]
        Zp[2] = Z[2]
    return Zp


def _eval_J(dae, t, y, vsize, eval_numjac=False):
    if eval_numjac:
        from Solverz.num_api.numjac import numjac
        _, dFdyt, _ = numjac(lambda t_, y_: dae.F(t_, y_, dae.p), t, y,
                             dae.F(t, y, dae.p), 1e-5 * np.ones_like(y), 0, 0, 0, 0)
        return dFdyt[:, 0:vsize]
    return dae.J(t, y, dae.p)


def _build_lu(M, h, J, l1, DiffEqn):
    M_csc = M if issparse(M) else csc_matrix(M)
    J_csc = J if issparse(J) else csc_matrix(J)
    row_scale_J = np.where(DiffEqn, h, 1.0)
    W = diags_array(row_scale_J, format='csc') @ J_csc - l1 * M_csc
    row_max = np.max(np.abs(W), axis=1).toarray().ravel()
    rscale = 1.0 / np.maximum(row_max, 1e-30)
    W = diags_array(rscale, format='csc') @ W
    return lu_decomposition(W), rscale


def _yp(F_val, diff_rows, diff_cols, diff_vals):
    yp = np.zeros_like(F_val)
    yp[diff_cols] = F_val[diff_rows] / diff_vals
    return yp


def _hermite(s, y_n, y_new, yp_n, yp_new, h):
    h0 = (1.0 - s) ** 2 * (1.0 + 2.0 * s)
    h1 = s * s * (3.0 - 2.0 * s)
    h2 = s * (1.0 - s) ** 2
    h3 = -(s * s) * (1.0 - s)
    return h0 * y_n + h1 * y_new + h * h2 * yp_n + h * h3 * yp_new


def _locate_events(events_fn, t_n, t_new, y_n, y_new, yp_n, yp_new, h,
                   prev_vals, direction, uround, event_dur):
    nv, new_isterm, new_dir = events_fn(t_new, y_new)
    new_vals = np.asarray(nv, dtype=float)
    new_isterm = np.asarray(new_isterm)
    new_dir = np.asarray(new_dir, dtype=float)
    hits = []
    for i in np.where(prev_vals * new_vals < 0)[0]:
        v0, v1 = prev_vals[i], new_vals[i]
        if direction[i] < 0 and v0 <= v1:
            continue
        if direction[i] > 0 and v0 >= v1:
            continue
        tL, tR = t_n, t_new
        t_e = t_n - v0 * h / (v1 - v0) if abs(v1 - v0) > uround else t_new
        tol = max(128.0 * uround, event_dur)
        for _ in range(60):
            s = (t_e - t_n) / h
            y_e = _hermite(s, y_n, y_new, yp_n, yp_new, h)
            v_now = np.asarray(events_fn(t_e, y_e)[0], dtype=float)[i]
            if v1 * v_now < 0:
                tL, v0, t_e = t_e, v_now, 0.5 * (t_e + tR)
            elif v0 * v_now < 0:
                tR, v1, t_e = t_e, v_now, 0.5 * (tL + t_e)
            else:
                break
            if (tR - tL) < tol:
                break
        if t_e - t_n < event_dur:
            continue
        y_e = _hermite((t_e - t_n) / h, y_n, y_new, yp_n, yp_new, h)
        hits.append((t_e, y_e, int(i), bool(new_isterm[i])))
    hits.sort(key=lambda r: r[0])
    return new_vals, new_isterm, new_dir, hits
