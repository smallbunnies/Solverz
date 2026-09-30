"""The two configurations of an integration, selected once per call.

``LegacyRodasPolicy`` reproduces legacy Rodas and ``DefaultPolicy`` follows
the rules of the integrator core. The Integrator builds the one that
``alg.legacy_compat`` selects, so that its loop contains no branch on the
configuration. Each policy names its ``dF/dt`` policy of ``derivative.py``
and holds its error norm, a function ``error_norm(integ, e)`` that works in
the Integrator's buffers ``_w``, ``_w2`` and ``_fin`` and returns an
``np.float64``.

The loop asks the policy, in this order: ``check_error`` whether the step
proposed for an attempt ends the run, and why; ``fix_dt_at_bounds`` and
``modify_dt_for_tstops`` for the step of the attempt; after the attempt,
``sanitize_EEst`` for the error the controller reads and ``dt_propose`` for
the step the controller proposes; ``savevalues`` for the rows of an accepted
step; ``handle_tstop`` and ``at_end`` whether the run is over.
``initial_dt`` bounds the algorithm's first step.
"""
import heapq
import math

import numpy as np

__all__ = []


def legacy_error_norm(integ, e):
    """``max |e / (atol + rtol |u|)|``, the error of legacy Rodas, and
    ``1e6`` when ``u`` is not finite.

    A ``NaN`` from ``0/0``, which ``atol = 0`` and a zero component give with
    a finite ``u``, stays ``NaN`` and rejects the attempt, as in legacy.
    """
    u, w, opts = integ.u, integ._w, integ.opts
    np.abs(u, out=w)
    np.multiply(w, opts.rtol, out=w)
    np.add(w, opts.atol, out=w)
    np.divide(e, w, out=w)
    np.abs(w, out=w)
    err = np.max(w)
    fin = integ._fin
    np.isfinite(u, out=fin)
    # for float64, "not every entry is finite" is legacy's "some entry is
    # inf or NaN", without its two temporary arrays
    if not fin.all():
        err = np.float64(1.0e6)
    return err


def _scaled_error(integ, e):
    """``e / (atol + rtol max(|u|, |uprev|))`` into ``integ._w``."""
    w, w2, opts = integ._w, integ._w2, integ.opts
    np.abs(integ.u, out=w)
    np.abs(integ.uprev, out=w2)
    np.maximum(w, w2, out=w)
    np.multiply(w, opts.rtol, out=w)
    np.add(w, opts.atol, out=w)
    np.divide(e, w, out=w)
    return w


def max_error_norm(integ, e):
    """The largest scaled error component."""
    w = _scaled_error(integ, e)
    return np.max(np.abs(w, out=w))


def rms_error_norm(integ, e):
    """The root mean square of the scaled error components."""
    w = _scaled_error(integ, e)
    return np.sqrt(np.mean(np.square(w, out=w)))


class LegacyRodasPolicy:
    """The legacy-compatible configuration: the step control, the saving and
    the end test of legacy Rodas, ``rodas.py:65-358`` without its events, and
    its ``dF/dt`` and error norm, the latter for every algorithm.

    Every expression is legacy's with the same NumPy functions, operand order
    and types: ``t0``, ``tend``, the nodes and ``dtmax`` keep the types legacy
    computes them with, so that an integer ``tspan`` gives the integer times
    and steps it gives there. ``t`` advances by ``t + dt``, the step is
    stretched to ``tend`` when it reaches it and otherwise capped at half the
    remaining span, and the run ends when ``|tend - t|`` falls below
    ``spacing(1)``. There are no ``tstops``.
    """

    dfdt = 'legacy'

    def __init__(self, opts, alg):
        self.error_norm = legacy_error_norm
        self.opts = opts
        self.tend = opts.tend
        self.saveat = opts.saveat
        # rodas.py:80-81: fixed for the call, from t0 in its tspan dtype
        self.hmin = 16 * np.spacing(opts.t0)
        self.uround = np.spacing(1.0)

    def initial_dt(self, integ, dt):
        """The algorithm's first step clamped to ``[hmin, dtmax]``, ``rodas.py:116-117``."""
        dt = np.maximum(dt, self.hmin)
        return np.minimum(dt, self.opts.dtmax)

    def check_error(self, integ):
        """Why the step proposed for the next attempt ends the run, or ``None``;
        the tests of ``rodas.py:135-143`` in their order, then a non-finite step."""
        dt = integ.dt
        if np.abs(dt) < self.uround:
            return f"the step size {float(dt)!r} is too small"
        if integ.nconsecutive_reject > self.opts.max_consecutive_reject:
            return f"more than {self.opts.max_consecutive_reject} consecutive attempts were rejected"
        if not np.isfinite(dt):
            return f"the step size {float(dt)!r} is not finite"
        return None

    def fix_dt_at_bounds(self, integ):
        """The step of the attempt, ``rodas.py:146-149``: stretched to ``tend``
        when it reaches it, otherwise at most half the remaining span. A run
        with a fixed step takes ``dt0`` and only the stretch."""
        t, tend = integ.t, self.tend
        if not self.opts.adaptive:
            dt = self.opts.dt0
            integ.dt = tend - t if t + dt >= tend else dt
        elif t + integ.dt >= tend:
            integ.dt = tend - t
        else:
            integ.dt = np.minimum(integ.dt, 0.5 * (tend - t))

    def modify_dt_for_tstops(self, integ):
        pass

    def sanitize_EEst(self, integ):
        """The ``1e6`` of a non-finite state is already inside the legacy norm."""

    def dt_propose(self, integ, dtnew):
        return dtnew

    def savevalues(self, integ):
        """The rows of the accepted step, as ``rodas.py:301-335`` saves them.

        With two entries in ``tspan`` the step's end. With more, every node in
        ``(tprev, t]``, each from the interpolant, ``tend`` included, and the
        algorithm's ``addsteps`` once per step before the first node. The
        condition is legacy's, ``t >= node > tprev``, so on a grid that does
        not increase saving ends at the first node that lies at or before the
        start of the step that reaches it. A node at the time ``te`` of an
        event handled in the step is the state ``u`` there, so that the last
        row of a terminal event is the recorded state; an event-free run
        never takes this branch.
        """
        if self.saveat is None:
            integ.sol.push(integ.t, integ.u.copy())
            return
        nodes, idx = self.saveat, integ.saveat_idx
        t, tprev, dt, te = integ.t, integ.tprev, integ.dt_step, integ._te
        while idx < len(nodes) and t >= nodes[idx] > tprev:
            tq = nodes[idx]
            if te is not None and tq == te:
                integ.sol.push(tq, integ.u.copy())
            else:
                integ.sol.push(tq, integ._interpolate((tq - tprev) / dt, np.empty(integ.n)))
            idx += 1
        integ.saveat_idx = idx

    def handle_tstop(self, integ):
        pass

    def at_end(self, integ):
        """``rodas.py:346``: an absolute test, which ``t = t + dt`` meets
        unless the sum misses ``tend`` by more than ``spacing(1)``."""
        return np.abs(self.tend - integ.t) < self.uround


def dtmin(t):
    """The smallest step at ``t``, ``16 * ulp(|t|)``, relative to ``t``.

    ``math.ulp`` equals ``np.spacing`` for every finite non-negative float
    and costs a plain C call.
    """
    return 16 * math.ulp(abs(t))


class DefaultPolicy:
    """The default configuration: the ``dF/dt`` of ode23s and the error scaled
    by ``max(|u|, |uprev|)``, in the norm the algorithm declares.

    ``t0``, ``tend`` and ``dtmax`` are Python floats. A step is bounded by
    ``dtmax`` and by ``dtmin(t)``, relative to the current time. The
    Integrator's heap of stop times holds ``tend`` and the ``tstops`` of the
    call; a step that would reach the next stop time, or end within 100
    units in the last place before it, is shortened or stretched to end on it
    exactly, and the run ends when the heap is empty. With more than two
    entries in ``tspan`` the nodes are saved from the interpolant and never
    change the steps.
    """

    dfdt = 'ode23s'

    def __init__(self, opts, alg):
        self.opts = opts
        self.dtmax = opts.dtmax
        self.max_consecutive_reject = opts.max_consecutive_reject
        # Python floats with the bits of the float64 nodes, for scalar comparisons
        self.nodes = None if opts.saveat is None else opts.saveat.tolist()
        if alg.norm == 'max':
            self.error_norm = max_error_norm
        elif alg.norm == 'rms':
            self.error_norm = rms_error_norm
        else:
            raise ValueError(f"{alg.scheme}.norm is {alg.norm!r}; it must be 'rms' or 'max'")

    def initial_dt(self, integ, dt):
        """The algorithm's first step clamped to ``[dtmin(t0), dtmax]``."""
        opts = self.opts
        return min(opts.dtmax, max(dtmin(opts.t0), dt))

    def check_error(self, integ):
        """Why the step proposed for the next attempt ends the run, or ``None``.

        A step below ``dtmin(t)`` ends the run only after a rejected or
        failed attempt: a step proposed after an acceptance is bounded
        below by ``dtmin`` already, and ``fix_dt_at_bounds`` raises any
        other step to it.
        """
        dt = integ.dt
        if not math.isfinite(dt):
            return f"the step size {float(dt)!r} is not finite"
        if integ.nconsecutive_reject > self.max_consecutive_reject:
            return f"more than {self.max_consecutive_reject} consecutive attempts were rejected"
        if integ.iter > 0 and not integ.accept_step and dt < dtmin(integ.t):
            return f"the step size {float(dt)!r} is too small"
        return None

    def fix_dt_at_bounds(self, integ):
        """The step of the attempt within ``[dtmin(t), dtmax]``."""
        integ.dt = max(min(integ.dt, self.dtmax), dtmin(integ.t))

    def modify_dt_for_tstops(self, integ):
        """Truncate the attempt to end on the next stop time, or stretch it
        there when it would end within ``100 ulp`` before it.

        The step becomes the distance itself, so the step the algorithm
        computes ends on the stop time up to one rounding of ``t + dt``;
        on acceptance ``t`` is assigned the stop time exactly. The step
        before truncation is kept for ``dt_propose``.
        """
        t, dt = integ.t, integ.dt
        tstop = integ.tstops[0]
        distance = tstop - t
        integ.dt_untruncated = dt
        if dt + 100 * math.ulp(max(abs(t), abs(tstop))) < distance:
            integ.next_step_tstop = False
        else:
            integ.next_step_tstop = True
            integ.tstop_target = tstop
            integ.dt = distance

    def sanitize_EEst(self, integ):
        """``EEst = inf`` unless the error and the state are finite, so that
        the controller shrinks the step by its smallest factor."""
        if not (math.isfinite(integ.EEst) and integ._all_finite(integ.u)):
            integ.EEst = np.inf

    def dt_propose(self, integ, dtnew):
        """The controller's step within ``[dtmin(t), dtmax]``; after a step
        shortened only to meet a stop time, at least the step it replaced."""
        if integ.next_step_tstop and integ.dt_untruncated > integ.dt:
            dtnew = max(dtnew, integ.dt_untruncated)
        return min(self.dtmax, max(dtmin(integ.t), dtnew))

    def savevalues(self, integ):
        """The rows of the accepted step.

        With two entries in ``tspan`` the step's end, unless a callback has
        saved a row at ``t`` already. With more, every node up to ``t``: a
        node equal to ``t`` is ``u`` itself, and every other node comes from
        the interpolant, so ``tend``, the last stop time, is saved exactly.
        """
        t = integ.t
        nodes = self.nodes
        if nodes is None:
            if integ.sol.last_t != t:
                integ.sol.push(t, integ.u.copy())
            return
        idx = integ.saveat_idx
        while idx < len(nodes) and nodes[idx] <= t:
            tq = nodes[idx]
            integ.sol.push(tq, integ.interp(tq))
            idx += 1
        integ.saveat_idx = idx

    def handle_tstop(self, integ):
        """Pop the stop time the accepted step landed on."""
        tstops, t = integ.tstops, integ.t
        while tstops and tstops[0] == t:
            heapq.heappop(tstops)

    def at_end(self, integ):
        return not integ.tstops
