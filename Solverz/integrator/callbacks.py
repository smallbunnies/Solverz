"""Callbacks: conditions checked on every accepted step, their crossings
located on the step's interpolant, and the adapter of legacy ``opt.event``.

A crossing is found after the step that contains it was accepted, from the
bottom values of the condition at the start of the step, its values at the
end and, for the components that may cross, at interior points of the
interpolant. The step itself is never shortened to find it. ``locate``
returns the earliest crossing that acts on the run and every crossing up to
it; the Integrator then handles them at that one time.
"""
import math

import numpy as np

__all__ = ['ContinuousCallback']

_ROOTFIND = ('left', 'right')


def _same_sign(a, b):
    """Whether ``a`` and ``b`` have the same sign; ``False`` when either is
    zero or ``NaN``, as ``np.sign(a) == np.sign(b)`` for nonzero ``b``."""
    return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)


def find_root(g, tl, tr, gl, gr, side, tstart):
    """The crossing of the scalar function ``g`` in ``[tl, tr]``.

    ``gl != 0`` and ``gr`` is zero or of the opposite sign. Regula falsi with
    the Illinois modification, and a bisection every third iteration and as
    fallback, shrinks the bracket to two adjacent floats; there is no
    tolerance. ``'right'`` returns the first float at which ``g`` has crossed
    or is zero, and ``'left'`` the last float before it, unless that float
    is ``tstart``, the start of the step, which is never an event time; then
    it returns the first float after it. An exact zero found on the way is
    returned whatever the side.
    """
    if gr == 0.0:
        return tr
    moved = 0
    for it in range(500):
        if math.nextafter(tl, math.inf) >= tr:
            break
        tm = tr - gr * (tr - tl) / (gr - gl)
        # the bisection bounds the count near that of plain bisection when
        # regula falsi stalls on one end of the bracket
        if it % 3 == 2 or not (tl < tm < tr):
            tm = tl + 0.5 * (tr - tl)
            if not (tl < tm < tr):
                tm = math.nextafter(tl, math.inf)
        gm = g(tm)
        if gm == 0.0:
            return tm
        if _same_sign(gm, gl):
            tl, gl = tm, gm
            if moved == 1:
                gr *= 0.5
            moved = 1
        else:
            tr, gr = tm, gm
            if moved == -1:
                gl *= 0.5
            moved = -1
    if side == 'left' and tl > tstart:
        return tl
    return tr


class ContinuousCallback:
    """An event located where a condition crosses zero inside an accepted step.

    ``condition(t, y, integ)`` returns a float or a 1-D array of a fixed
    length ``m``, one value per component. A component crosses when its
    value goes from nonzero to zero or to the other sign. A component that
    is exactly zero at the start of a step does not cross there, so nothing
    is reported at the initial point of a call; its next crossing is, also
    inside the same step. ``direction`` is -1 for crossings from positive
    to non-positive only, +1 for crossings from negative to non-negative
    only, and 0 for both. ``direction`` and ``terminal`` are a scalar or an
    array of length ``m``.

    A terminal component ends the run at its crossing, and ``record=True``
    logs every crossing as ``(te, ye, ie)`` in the result; at most one
    callback of a run records. ``affect(integ, idx)`` receives the indices of
    the components that cross at the event time. A component acts on the
    run if it is terminal or the callback has an ``affect``; the earliest
    acting crossing ``te`` is one event instant for all callbacks, and every
    component of any callback that crosses at ``te`` is handled there.

    The crossing is located on the step's interpolant, at the samples of
    ``interp_points`` points spread over the step and then to adjacent
    floats. ``rootfind='left'`` returns the last float before the crossing,
    but never the start of the step, and ``'right'`` the first float at
    which the component has crossed or is zero. ``save_positions`` saves the
    state before and after the event at ``te``. After an event at the start
    of a step, a re-crossing by the same component within ``repeat_nudge``
    of the step is the event already reported.
    """

    def __init__(self, condition, affect=None, *, direction=0, terminal=False, record=False,
                 rootfind='left', save_positions=(True, True), interp_points=10, repeat_nudge=0.01):
        if not callable(condition):
            raise TypeError("condition must be callable as condition(t, y, integ)")
        if affect is not None and not callable(affect):
            raise TypeError("affect must be None or callable as affect(integ, idx)")
        d = np.array(direction, dtype=np.int64)
        if d.ndim > 1 or not np.isin(d, (-1, 0, 1)).all() or not np.array_equal(d, direction):
            raise ValueError(f"direction must be -1, 0 or +1, or a 1-D array of them, not {direction!r}")
        term = np.array(terminal, dtype=bool)
        if term.ndim > 1:
            raise ValueError(f"terminal must be a bool or a 1-D array of bools, not {terminal!r}")
        if rootfind not in _ROOTFIND:
            raise ValueError(f"rootfind must be 'left' or 'right', not {rootfind!r}")
        save_positions = tuple(save_positions)
        if len(save_positions) != 2:
            raise ValueError(f"save_positions must be a pair of bools, not {save_positions!r}")
        if isinstance(interp_points, bool) or int(interp_points) != interp_points or interp_points < 0:
            raise ValueError(f"interp_points must be a non-negative int, not {interp_points!r}")
        if not 0 <= repeat_nudge < 1:
            raise ValueError(f"repeat_nudge must lie in [0, 1), not {repeat_nudge!r}")
        d.flags.writeable = False
        term.flags.writeable = False
        self.condition = condition
        self.affect = affect
        self.direction = d
        self.terminal = term
        self.record = bool(record)
        self.rootfind = rootfind
        self.save_positions = (bool(save_positions[0]), bool(save_positions[1]))
        self.interp_points = int(interp_points)
        self.repeat_nudge = repeat_nudge

    def _call(self, t, y, integ):
        """``(value, direction, terminal)`` at ``(t, y)``; ``None`` for the
        traits, which are fixed for the callback."""
        return self.condition(t, y, integ), None, None


class _LegacyEventCallback(ContinuousCallback):
    """Legacy ``opt.event(t, y) -> (value, isterminal, direction)`` as a
    recording callback located with ``rootfind='right'``.

    ``isterminal`` and ``direction`` are read from the evaluation at the end
    of the step being examined, since legacy events may change them with
    the state. With ``'right'`` the state returned at a terminal event has
    crossed or lies on the surface, so a new call from it does not detect
    the same crossing again. No row is saved for the event other than the
    step's own.
    """

    def __init__(self, event):
        super().__init__(lambda t, y, integ: event(t, y)[0], record=True, rootfind='right',
                         save_positions=(False, False), interp_points=10)
        self.event = event

    def _call(self, t, y, integ):
        value, isterminal, direction = self.event(t, y)
        return value, direction, isterminal


class _ContinuousState:
    """What one integration keeps of one continuous callback.

    A callback object may serve several calls, so its state lives here: the
    traits resolved for its ``m`` components, the bottom values ``g0`` of the
    condition at the start of the current step and ``g1`` at its end, the
    condition's values at the times evaluated in the current step, and the
    components that fired at the last event with its time, which the
    modification protocol stores for the repeat nudge.
    """

    __slots__ = ('cb', 'order', 'm', 'g0', 'g1', 'down', 'up', 'terminal', 'acting',
                 'record_only', 'relevant', 'values', 'fired', 'fired_t')

    def __init__(self, cb, order, integ):
        self.cb = cb
        self.order = order
        self.m = None
        g, direction, terminal = self.evaluate(integ, integ.t, integ.u)
        self.m = g.shape[0]
        self._traits(cb.direction if direction is None else direction,
                     cb.terminal if terminal is None else terminal)
        self.g0 = g
        self.g1 = None
        self.values = {}
        self.fired = None
        self.fired_t = None

    def evaluate(self, integ, t, y):
        """``(g, direction, terminal)`` of one counted call of the condition;
        ``g`` is a new float64 vector."""
        integ.stats.ncondition += 1
        value, direction, terminal = self.cb._call(t, y, integ)
        g = np.array(value, dtype=np.float64, ndmin=1)
        if g.ndim != 1 or (self.m is not None and g.shape[0] != self.m):
            raise ValueError(f"the condition returned a value of shape {g.shape} at t = {float(t)!r}; "
                             f"it must be a float or a 1-D array of fixed length"
                             + ('' if self.m is None else f" {self.m}"))
        return g, direction, terminal

    def _traits(self, direction, terminal):
        """The direction and terminal masks of the ``m`` components, and
        which components act, are recorded, or matter at all."""
        m = self.m
        d = np.asarray(direction)
        term = np.asarray(terminal) != 0
        for name, a in (('direction', d), ('terminal', term)):
            if a.ndim > 1 or (a.ndim == 1 and a.shape[0] != m):
                raise ValueError(f"{name} has shape {a.shape}; the condition has {m} components")
        self.down = np.broadcast_to(d < 0, (m,))
        self.up = np.broadcast_to(d > 0, (m,))
        self.terminal = np.broadcast_to(term, (m,))
        self.acting = self.terminal | (self.cb.affect is not None)
        self.record_only = ~self.acting & self.cb.record
        self.relevant = self.acting | self.record_only

    def value(self, integ, tau):
        """The condition at ``tau`` on the step's interpolant, evaluated once
        per time and step."""
        g = self.values.get(tau)
        if g is None:
            g = self.values[tau] = self.evaluate(integ, tau, integ.interp(tau, out=integ._cb_y))[0]
        return g

    def end(self, integ):
        """Evaluate the condition at the end of the accepted step into ``g1``;
        the adapter takes its traits from this evaluation."""
        t = integ.t
        g1, direction, terminal = self.evaluate(integ, t, integ.u)
        if direction is not None:
            self._traits(direction, terminal)
        self.g1 = g1
        self.values = {integ.tprev: self.g0, t: g1}

    def allowed(self, idx, g):
        """Whether the directions of the components ``idx`` allow a crossing
        from the values ``g``: -1 needs a positive value, +1 a negative one."""
        return ~((self.down[idx] & ~(g > 0)) | (self.up[idx] & ~(g < 0)))

    def eligible(self):
        """The components that may cross in the step from its start: acting
        or recorded, nonzero at the start, and in an allowed direction."""
        g0 = self.g0
        return self.relevant & (g0 != 0) & self.allowed(slice(None), g0)


class _Crossing:
    """A bracket ``[bottom, top]`` of component ``i`` of a callback, the
    condition's values at its ends, and the root located in it."""

    __slots__ = ('st', 'i', 'bottom', 'top', 'gl', 'gr', 'root')

    def __init__(self, st, i, bottom, top, gl, gr):
        self.st = st
        self.i = int(i)
        self.bottom = bottom
        self.top = top
        self.gl = float(gl)
        self.gr = float(gr)
        self.root = None

    def key(self):
        return self.bottom, self.st.order, self.i

    def time_key(self):
        return self.root, self.st.order, self.i


def _sample_times(integ, npoints):
    """The sample points of the accepted step after its start: the
    ``npoints - 2`` interior points and the end ``t`` itself."""
    tl, span = float(integ.tprev), float(integ.t) - float(integ.tprev)
    for k in range(1, npoints - 1):
        yield tl + k * span / (npoints - 1)
    yield integ.t


def _brackets(st, integ):
    """The brackets of the components of ``st`` that cross in the accepted step.

    For every eligible component the bracket is the first subinterval of the
    sample points at whose top it has crossed or is zero, so a crossing that
    enters and leaves within the step is found, and the first of several. A
    component that is exactly zero at the start of the step is not crossing
    there; its reference is the first sample at which it is nonzero, if its
    direction allows that sign, so that its next crossing is found even
    inside the same step.
    """
    tprev, t = integ.tprev, integ.t
    st.end(integ)
    npoints = st.cb.interp_points
    g0, g1 = st.g0, st.g1
    pending = np.flatnonzero(st.eligible())
    # without interior samples a component that leaves zero cannot cross
    # again before the end of the step
    waiting = np.flatnonzero(st.relevant & (g0 == 0)) if npoints >= 3 else pending[:0]
    if pending.size == 0 and waiting.size == 0:
        return []
    found = []
    if npoints < 3:
        for i in pending[g1[pending] * g0[pending] <= 0]:
            found.append(_Crossing(st, i, tprev, t, g0[i], g1[i]))
    else:
        ref = g0 if waiting.size == 0 else g0.copy()
        prev_tau, prev_g = tprev, g0
        for tau in _sample_times(integ, npoints):
            gk = st.value(integ, tau)
            if pending.size:
                hit = gk[pending] * ref[pending] <= 0
                if hit.any():
                    for i in pending[hit]:
                        found.append(_Crossing(st, i, prev_tau, tau, prev_g[i], gk[i]))
                    pending = pending[~hit]
            if waiting.size:
                left = gk[waiting] != 0
                if left.any():
                    w = waiting[left]
                    w = w[st.allowed(w, gk[w])]
                    ref[w] = gk[w]
                    pending = np.concatenate((pending, w))
                    waiting = waiting[~left]
            if pending.size == 0 and waiting.size == 0:
                break
            prev_tau, prev_g = tau, gk
    if st.cb.rootfind == 'left' and st.fired is not None and st.fired_t == tprev:
        found = _nudge(st, integ, found)
    return found


def _nudge(st, integ, found):
    """Move the bottom of a bracket that starts at an event reported at the
    start of the step, of a component that fired there, to ``tn``, by
    ``repeat_nudge`` of the step; a component that has crossed by ``tn``
    crosses the surface it was reported on, and has no event in this step."""
    tprev = integ.tprev
    tn = tprev + st.cb.repeat_nudge * (integ.t - tprev)
    kept = []
    for c in found:
        if c.bottom != tprev or c.i not in st.fired:
            kept.append(c)
            continue
        gn = float(st.value(integ, tn)[c.i])
        if gn * c.gl <= 0:
            continue
        if tn < c.top:
            c.bottom, c.gl = tn, gn
            kept.append(c)
            continue
        # the bracket lies inside the nudge interval, and the component is
        # back on its side by tn: a crossing after tn is a new event
        c = _bracket_after(st, integ, c.i, tn, gn)
        if c is not None:
            kept.append(c)
    return kept


def _bracket_after(st, integ, i, tb, gb):
    """The first bracket of component ``i`` after ``tb``, with ``gb != 0`` its
    value there, whose top is a sample point; ``None`` without one."""
    for tau in _sample_times(integ, st.cb.interp_points):
        if tau <= tb:
            continue
        gk = float(st.value(integ, tau)[i])
        if gk * gb <= 0:
            return _Crossing(st, i, tb, tau, gb, gk)
        tb, gb = tau, gk
    return None


def _root(c, te, integ):
    """The root of the crossing ``c`` if it may lie at or before ``te``, else ``None``.

    A bracket that starts after ``te`` holds no such root. When ``te`` lies
    inside the bracket, the condition is evaluated once at the probe time,
    shared by every component of the callback, and a component that has not
    crossed by then is skipped. The probe is ``te`` for ``'right'``; for
    ``'left'`` it is the float after ``te``, since a ``'left'`` root is the
    last float before the crossing and equals ``te`` when the crossing lies
    between ``te`` and the next float.
    """
    st, i = c.st, c.i
    side = st.cb.rootfind
    bottom, top, gl, gr = c.bottom, c.top, c.gl, c.gr
    if te is not None:
        if bottom > te or (bottom == te and side == 'right'):
            return None
        if te < top:
            probe = te if side == 'right' else math.nextafter(te, math.inf)
            gp = float(st.value(integ, probe)[i])
            if _same_sign(gp, gl):
                return None
            top, gr = probe, gp

    def g(tau):
        return float(st.value(integ, tau)[i])

    return find_root(g, bottom, top, gl, gr, side, integ.tprev)


def locate(integ, states):
    """The crossings of the accepted step ``[tprev, t]`` of every continuous callback.

    Returns ``(te, found)``. ``te`` is the earliest root of an acting
    component, or ``None``. ``found`` holds the crossings, with their
    ``root``, of every acting or recorded component whose root lies at or
    before ``te``, or of every recorded component when no acting one
    crosses, in increasing time, ties by callback and then by index. The
    acting components are located first, in increasing order of their
    bracket bottoms; a component whose root cannot lie at or before the
    earliest root found so far costs at most one evaluation instead of a
    search. The recorded ones follow against the final ``te``.
    """
    crossings = []
    for st in states:
        crossings += _brackets(st, integ)
    if not crossings:
        return None, []
    acting = sorted((c for c in crossings if c.st.acting[c.i]), key=_Crossing.key)
    recorded = sorted((c for c in crossings if not c.st.acting[c.i]), key=_Crossing.key)
    te = None
    found = []
    for c in acting:
        root = _root(c, te, integ)
        if root is None:
            continue
        c.root = root
        found.append(c)
        if te is None or root < te:
            te = root
    for c in recorded:
        root = _root(c, te, integ)
        if root is None:
            continue
        c.root = root
        found.append(c)
    if te is not None:
        found = [c for c in found if c.root <= te]
    found.sort(key=_Crossing.time_key)
    return te, found
