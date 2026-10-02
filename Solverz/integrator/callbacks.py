"""Callbacks: conditions checked on every accepted step, their crossings
located on the step's interpolant, the adapter of legacy ``opt.event``, and
discrete callbacks that change the run at the end of a step.

A crossing is found after the step that contains it was accepted, from the
bottom values of the condition at the start of the step, its values at
interior points of the interpolant and at the end. The step itself is never
shortened to find it. ``locate`` returns the earliest crossing that acts on
the run and every crossing up to it; the Integrator then handles them at
that one time.
"""
import math

import numpy as np

__all__ = ['ContinuousCallback', 'DiscreteCallback', 'preset_time_callback']

_ROOTFIND = ('left', 'right')
_EPS = float(np.spacing(1.0))


def _same_sign(a, b):
    """Whether ``a`` and ``b`` have the same sign; ``False`` when either is
    zero or ``NaN``, as ``np.sign(a) == np.sign(b)`` for nonzero ``b``.

    The signs are compared, not the product ``a * b``, which underflows to
    zero when both values lie below about ``1e-162``.
    """
    return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)


def _crossed(g, ref):
    """Whether ``g`` is zero or of the sign opposite to the nonzero ``ref``,
    elementwise; ``False`` where either is ``NaN``, as ``g * ref <= 0``
    without its underflow."""
    return (g == 0) | ((g > 0) & (ref < 0)) | ((g < 0) & (ref > 0))


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
        # gr - gl is zero only when the Illinois halving has taken both to
        # zero; the bisection below then takes over
        d = gr - gl
        tm = tr - gr * (tr - tl) / d if d != 0.0 else tl
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
    callback of a run records. ``affect(integ, idx)`` receives the int array
    of the components that cross at the event time; it may change
    ``integ.u``, entries of ``integ.dae.p`` and ``integ.dae.M.data``, and may
    call ``integ.terminate()``, and the core then makes the state consistent
    again with ``DaeIc``. A component acts on the run if it is terminal or
    the callback has an ``affect``; the earliest acting crossing ``te`` is
    one event instant for all callbacks, and every component of any callback
    that crosses at ``te`` is handled there.

    The crossing is located on the step's interpolant, at the samples of
    ``interp_points`` points spread over the step and then to adjacent
    floats. ``rootfind='left'`` returns the last float before the crossing,
    but never the start of the step, and ``'right'`` the first float at
    which the component has crossed or is zero. ``save_positions`` saves the
    state before and after the event at ``te``. After an event at the start
    of a step, a re-crossing by the same component within ``repeat_nudge``
    of the step is the event already reported, unless a change at the event
    moved the component off its value there.
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


class DiscreteCallback:
    """A change of the run at the end of an accepted step where a condition holds.

    ``condition(t, y, integ)`` returns a bool and is evaluated after every
    accepted step, after the continuous callbacks, unless the run has been
    terminated. Where it holds,
    ``affect(integ)`` runs; it may change ``integ.u``, entries of
    ``integ.dae.p`` and ``integ.dae.M.data``, and may call
    ``integ.terminate()``, and the core then makes the state consistent
    again with ``DaeIc``. ``save_positions`` saves the state before and
    after the affect at ``t``; when every step is saved, the step's own row
    at ``t`` is the state before it. ``tstops`` are stop times of the run,
    so that a step ends exactly on each.
    """

    def __init__(self, condition, affect, *, save_positions=(True, True), tstops=()):
        if not callable(condition):
            raise TypeError("condition must be callable as condition(t, y, integ)")
        if not callable(affect):
            raise TypeError("affect must be callable as affect(integ)")
        save_positions = tuple(save_positions)
        if len(save_positions) != 2:
            raise ValueError(f"save_positions must be a pair of bools, not {save_positions!r}")
        self.condition = condition
        self.affect = affect
        self.save_positions = (bool(save_positions[0]), bool(save_positions[1]))
        self.tstops = tuple(tstops)


def preset_time_callback(times, affect, *, save_positions=(True, True)):
    """A ``DiscreteCallback`` whose ``affect(integ)`` runs at each of ``times``.

    The times are stop times of the run, so a step ends exactly on each of
    them and the test of a time against the set is exact. A time outside
    ``(t0, tend]`` is never reached, so its affect never runs.
    """
    times = tuple(times)
    timeset = frozenset(float(x) for x in times)
    return DiscreteCallback(lambda t, y, integ: t in timeset, affect, save_positions=save_positions,
                            tstops=times)


class _ContinuousState:
    """What one integration keeps of one continuous callback.

    A callback object may serve several calls, so its state lives here: the
    traits resolved for its ``m`` components, the bottom values ``g0`` of the
    condition at the start of the current step and ``g1`` at its end, the
    condition's values at the times evaluated in the current step, and the
    components that fired at the last event with its time, which the
    modification protocol stores for the repeat nudge, with their values at
    the event before any change.
    """

    __slots__ = ('cb', 'order', 'm', 'g0', 'g1', 'down', 'up', 'terminal', 'acting',
                 'record_only', 'relevant', 'values', 'fired', 'fired_t', 'fired_g')

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
        self.fired_g = None

    def restart(self, integ):
        """The bottom values at ``(t, u)``, after the state or the model changed."""
        g, direction, terminal = self.evaluate(integ, integ.t, integ.u)
        if direction is not None:
            self._traits(direction, terminal)
        self.g0 = g

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

    def allows(self, i, g):
        """``allowed`` for the one component ``i`` and the nonzero float ``g``."""
        return not ((self.down[i] and g < 0) or (self.up[i] and g > 0))


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
    """The brackets of the crossings of the components of ``st`` in the accepted step.

    Every acting or recorded component is followed from the start of the
    step through the sample points, against a reference, its last nonzero
    value. At a sample at which it is zero or of the other sign it has
    crossed: the subinterval that ends there is its bracket if its direction
    allows a crossing from the sign of the reference, and the sample becomes
    the reference either way, so that a crossing against the direction is
    followed by the next one in it. A component that is zero, at the start
    of the step or at a sample, is not crossing there, and its next nonzero
    sample is its reference. An acting component needs only its first
    crossing, since the run continues from the earliest acting one. A
    component that only records never shortens the step, so every crossing
    of it is bracketed. A crossing that enters and leaves between two
    samples is not seen.
    """
    tprev = integ.tprev
    st.end(integ)
    tracked = np.flatnonzero(st.relevant)
    if tracked.size == 0:
        return []
    acting = st.acting
    fired = None
    if st.cb.rootfind == 'left' and st.fired is not None and st.fired_t == tprev:
        # a component that a change at the event moved off its value there
        # has left the surface, so a crossing near the start is a new one
        fired = st.fired[_unmoved(st.g0[st.fired], st.fired_g)]
    ref = st.g0.copy()
    found = []
    first = True
    prev_tau, prev_g = tprev, st.g0
    for tau in _sample_times(integ, st.cb.interp_points):
        gk = st.value(integ, tau)
        g, r = gk[tracked], ref[tracked]
        left = (r == 0) & (g != 0)
        crossed = (r != 0) & _crossed(g, r)
        if left.any():
            w = tracked[left]
            ref[w] = gk[w]
        if crossed.any():
            pos = np.flatnonzero(crossed)
            c = tracked[pos]
            ok = st.allowed(c, ref[c])
            ref[c] = gk[c]
            done = np.zeros(tracked.size, dtype=bool)
            for p, i in zip(pos[ok], c[ok]):
                if first and fired is not None and i in fired:
                    found += _nudge(st, integ, i, tau, prev_g[i], gk[i])
                    done[p] = True
                else:
                    found.append(_Crossing(st, i, prev_tau, tau, prev_g[i], gk[i]))
                    done[p] = acting[i]
            if done.any():
                tracked = tracked[~done]
                if tracked.size == 0:
                    break
        first = False
        prev_tau, prev_g = tau, gk
    return found


def _unmoved(g, ge):
    """Whether each value ``g`` at the start of the step is the value ``ge``
    of the event, within ten units of rounding of the larger of the two."""
    return np.abs(g - ge) <= 10 * _EPS * np.maximum(np.abs(g), np.abs(ge))


def _nudge(st, integ, i, top, gl, gr):
    """The crossings of component ``i`` of a ``'left'`` callback that fired at
    the start of the step, and that no change at the event moved off its
    value there, whose first crossing lies in the bracket from the start to
    ``top``, with the values ``gl`` and ``gr`` at its ends.

    The condition is evaluated at ``tn``, ``repeat_nudge`` of the step after
    its start. If the component has not crossed by ``tn``, which lies before
    ``top``, its bracket starts at ``tn``. Otherwise its crossing up to ``tn``
    is the event already reported: it has crossed the surface it was
    reported on, or crossed it and returned within the nudge interval, and
    its crossings are those after ``tn``.
    """
    tprev = integ.tprev
    tn = tprev + st.cb.repeat_nudge * (integ.t - tprev)
    gn = float(st.value(integ, tn)[i])
    if _same_sign(gn, gl) and tn < top:
        found = [_Crossing(st, i, tn, top, gn, gr)]
        return found if st.acting[i] else found + _scan(st, integ, i, top, gr)
    return _scan(st, integ, i, tn, gn)


def _scan(st, integ, i, tb, gb):
    """The crossings of component ``i`` after the time ``tb``, at which its
    value is ``gb``, over the sample points after ``tb``, by the rules of
    ``_brackets``."""
    found = []
    ref = gb
    for tau in _sample_times(integ, st.cb.interp_points):
        if tau <= tb:
            continue
        gk = float(st.value(integ, tau)[i])
        if ref == 0.0:
            ref = gk
        elif _crossed(gk, ref):
            if st.allows(i, ref):
                found.append(_Crossing(st, i, tb, tau, gb, gk))
                if st.acting[i]:
                    break
            ref = gk
        tb, gb = tau, gk
    return found


def _root(c, te, integ):
    """The root of the crossing ``c`` if it may lie at or before ``te``, else ``None``.

    A bracket that starts after ``te`` holds no such root. When ``te`` lies
    inside the bracket, the condition is evaluated once at the probe time,
    shared by every component of the callback, and a component that has not
    crossed by then is skipped. The probe is ``te`` for ``'right'``; for
    ``'left'`` it is the float after ``te``, since a ``'left'`` root is the
    last float before the crossing and equals ``te`` when the crossing lies
    between ``te`` and the next float.

    A component that has crossed by the probe but not at the float before it
    changes sign exactly where the event does, and its root is ``te`` without
    a search. Rounding can make the interpolant change sign several times
    within a few floats, and ``find_root`` ends on whichever sign change its
    bracket leads to, so a search could separate two identical components.
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
            before = math.nextafter(probe, -math.inf)
            gb = gl if before <= bottom else float(st.value(integ, before)[i])
            if _same_sign(gb, gl):
                return te
            top, gr = probe, gp

    def g(tau):
        return float(st.value(integ, tau)[i])

    return find_root(g, bottom, top, gl, gr, side, integ.tprev)


def locate(integ, states):
    """The crossings of the accepted step ``[tprev, t]`` of every continuous callback.

    Returns ``(te, found)``. ``te`` is the earliest root of an acting
    component, or ``None``. ``found`` holds the crossings, with their
    ``root``, whose root lies at or before ``te``, or every crossing when no
    acting component crosses: the first of each acting component and all of
    each recorded one, in increasing time, ties by callback and then by
    index. The acting components are located first, in increasing order of
    their bracket bottoms; a component whose root cannot lie at or before the
    earliest root found so far costs at most one evaluation instead of a
    search. An acting component whose root lies after an earlier root found
    later is located again against it. The recorded ones follow against the
    final ``te``.
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
    # a component located before te fell below its root may have crossed by
    # te already, on another sign change of the rounded interpolant; it is
    # located again against te, which can lower te once more
    lowered = True
    while lowered:
        lowered = False
        for c in found:
            if c.root > te:
                root = _root(c, te, integ)
                if root is not None:
                    c.root = root
                    if root < te:
                        te, lowered = root, True
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
