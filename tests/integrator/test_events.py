"""Continuous callbacks and the adapter of legacy ``opt.event``.

A crossing is found after the step that contains it was accepted and is
located on the step's interpolant, to adjacent floats. A component that is
exactly zero at the start of a step is not a crossing, so nothing is
reported at the initial point of a call, a zero at a step end is reported
once, and a component that leaves zero is reported at its next crossing if
its direction allows it. Samples of the interpolant find a crossing that
enters and leaves within one step, every crossing of a component that only
records, and the first crossing in its direction of an acting one, also
after a crossing against it in the same step. The earliest crossing of a
terminal component ends the run with ``T[-1] == te`` and the last row equal
to the recorded state, on every output, in both configurations, with the
rows at ``te`` that ``save_positions`` selects. Every component that crosses
at that time is reported with it, and crossings of recorded components
before it are logged without shortening the step. The adapter of
``opt.event`` takes its traits from the end of each step. An affect changes
the run at the event time; the rows before it are those of the step as
computed, and a crossing it leaves in place is reported once, also by a
callback that only records it and after a second change at the same time.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from Solverz import made_numerical
from Solverz.integrator import (ContinuousCallback, DiscreteCallback, ImplicitEuler, Rodas3, Rodas4, Rodas5P,
                                Rosenbrock, Trapezoid, init, solve)
from Solverz.integrator.callbacks import _brackets, _ContinuousState, find_root, locate
from Solverz.solvers.daesolver.rodas.param import Rodas_param
from Solverz.solvers.option import Opt

from tests.integrator import models
from tests.integrator.test_legacy_transcription import SCHEMES

G = 9.8
FIRST_IMPACT = 40 / G
CONFIGS = [False, True]


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _event(value, terminal, direction):
    """A legacy ``opt.event`` from ``value(t, y)`` and fixed traits."""
    def event(t, y):
        return np.atleast_1d(value(t, y)), np.array(terminal), np.array(direction)
    return event


def _level_times(level):
    """The two times at which the ball, from height 0 with velocity 20, passes ``level``."""
    root = np.sqrt(20.0 ** 2 - 2 * G * level)
    return 2 * level / (20.0 + root), (20.0 + root) / G


# -- the bouncing ball and the orbit of test_rodas_event.py --------------------

LEGACY_BOUNCES = np.array([4.081633047365558, 7.755103268847604, 11.061226369910107, 14.03673684129459,
                           16.714695668619328, 19.124858513456523, 21.294005283692695, 23.246237159791765,
                           25.003245923211466, 26.584554164220997])


@pytest.mark.i6a
def test_ten_bounces_as_the_legacy_test():
    """``test_rodas_event.py:20-46`` with ``Rodas4()``: each call stops at the
    impact, and the next starts there on the root with the velocity reversed."""
    sdae, y0 = models.ball_model()
    dae = made_numerical(sdae, y0, sparse=True)
    opt = Opt(event=_event(lambda t, y: y[0], [1], [-1]))
    te, tstart = [], 0
    for _ in range(10):
        sol = Rodas4()(dae, np.linspace(tstart, 30, 100), y0, opt)
        assert sol.stats.ret == 'terminated' and sol.stats.succeed is True
        assert sol.ie.tolist() == [0] and sol.te[0] > tstart
        assert sol.T[-1] == sol.te[0] and np.all(np.diff(sol.T) > 0)
        te.append(sol.te[0])
        y0['x'][0] = 0
        y0['x'][1] = -0.9 * sol.Y[-1]['x'][1]
        tstart = sol.T[-1]
    np.testing.assert_allclose(te, LEGACY_BOUNCES, rtol=1e-5, atol=0)
    exact = np.cumsum(2 * 20 * 0.9 ** np.arange(10) / G)
    print(f"max relative deviation of the bounce times: from legacy "
          f"{np.max(np.abs(te / LEGACY_BOUNCES - 1)):.2e}, from the exact times {np.max(np.abs(te / exact - 1)):.2e}")


def _orbit_event(t, y):
    dDSQdt = 2 * (y[0:2] - np.array([1.2, 0])).dot(y[2:4])
    return np.array([dDSQdt, dDSQdt]), np.array([1, 0]), np.array([1, -1])


@pytest.fixture(scope='module')
def orbit_reference(model):
    dae, y0 = model('orbit')
    return Rodas5P()(dae, [0, 7], y0, Opt(event=_orbit_event, rtol=1e-11, atol=1e-13))


@pytest.mark.i6a
@pytest.mark.parametrize('scheme', SCHEMES)
def test_orbit_against_a_tight_reference(model, orbit_reference, scheme):
    """``test_rodas_event.py:49-97`` at ``rtol=1e-8``: the non-terminal
    component 1 is recorded at the far point of the orbit, and the terminal
    component 0 stops the run at the return.

    The bounds are those of the integration, not of the event location: at
    this tolerance the runs without events already deviate from the
    reference by up to ``9.3e-7`` of the state at the event times (Rodas3),
    and the largest deviations with events are ``1.1e-7`` for ``te`` and
    ``1.8e-6`` for ``ye``. ``ye`` is measured relative to the largest entry
    of its row, since ``y[1]`` at the return is about ``1e-9``.
    """
    ref = orbit_reference
    assert ref.stats.ret == 'terminated' and ref.ie.tolist() == [1, 0]
    dae, y0 = model('orbit')
    sol = Rosenbrock.from_scheme(scheme)(dae, [0, 7], y0, Opt(event=_orbit_event, rtol=1e-8, atol=1e-10))
    assert sol.stats.ret == 'terminated' and sol.ie.tolist() == [1, 0]
    assert sol.T[-1] == sol.te[-1] and _byte_equal(sol.Y[-1], sol.ye[-1])
    dte = np.max(np.abs(sol.te / ref.te - 1))
    dye = np.max(np.abs(sol.ye - ref.ye) / np.max(np.abs(ref.ye), axis=1, keepdims=True))
    print(f"{scheme}: te {sol.te}, relative deviation of te {dte:.2e}, of ye {dye:.2e}")
    assert dte <= 1e-6
    assert dye <= 1e-5


@pytest.mark.i6a
@pytest.mark.parametrize('scheme', SCHEMES)
def test_the_event_time_is_the_crossing_on_the_interpolant(model, scheme):
    """With ``'right'`` the event time is the first float at which the
    component has crossed or is zero on the interpolant of its step."""
    dae, y0 = model('orbit')
    integ = init(dae, [0, 7], y0, alg=Rosenbrock.from_scheme(scheme),
                 opt=Opt(event=_orbit_event, rtol=1e-6, atol=1e-8))
    while integ.step():
        pass
    assert integ.terminated
    te = integ.t
    assert integ.tprev < te < integ.t_step or te == integ.t_step

    def g(tq):
        return _orbit_event(tq, integ.interp(tq))[0][0]

    # the terminal component crosses from negative to non-negative
    assert g(te) >= 0
    assert g(te) == 0 or g(np.nextafter(te, -np.inf)) < 0


# -- terminal events and the last row -----------------------------------------

def _twin_event():
    return _event(lambda t, y: np.array([y[0], y[0]]), [1, 1], [-1, -1])


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
@pytest.mark.parametrize('grid', [False, True])
def test_identical_components_are_reported_together(model, legacy_compat, grid):
    dae, y0 = model('ball')
    tspan = np.linspace(0, 10, 21) if grid else [0, 10]
    sol = Rodas4(legacy_compat=legacy_compat)(dae, tspan, y0, Opt(event=_twin_event()))
    assert sol.stats.ret == 'terminated'
    assert sol.te.dtype == np.float64 and sol.ye.dtype == np.float64 and sol.ie.dtype == np.int64
    assert sol.te.shape == (2,) and sol.te[0] == sol.te[1]
    assert sol.ie.tolist() == [0, 1]
    assert _byte_equal(sol.ye[0], sol.ye[1])
    assert sol.T[-1] == sol.te[0] and _byte_equal(sol.Y[-1], sol.ye[-1])
    assert np.all(np.diff(sol.T) > 0)
    # 'right': the returned state lies on the ground or below it
    assert sol.ye[0][0] <= 0
    assert abs(sol.te[0] - FIRST_IMPACT) <= 1e-9 * FIRST_IMPACT
    if grid:
        nodes = tspan[tspan < sol.te[0]]
        assert _byte_equal(sol.T[:-1], nodes)


@pytest.mark.i6a
@pytest.mark.parametrize('rootfind', ['left', 'right'])
def test_components_at_te_and_before_it(model, rootfind):
    """On the descent, component 2 passes 15 first and is logged at its own
    time; the terminal component 0 then stops the run at 10, and its twin,
    component 1, which only records, is reported at the same float. With
    ``'left'`` the twin's root equals ``te`` when the crossing lies between
    ``te`` and the next float, so the test for a crossing by ``te`` is made
    at that next float."""
    dae, y0 = model('ball')
    cb = ContinuousCallback(lambda t, y, integ: np.array([y[0] - 10, y[0] - 10, y[0] - 15]),
                            direction=-1, terminal=[True, False, False], record=True, rootfind=rootfind)
    sol = solve(dae, [0, 5], y0, callbacks=[cb])
    assert sol.stats.ret == 'terminated'
    assert sol.ie.tolist() == [2, 0, 1]
    np.testing.assert_allclose(sol.te, [_level_times(15.0)[1]] + 2 * [_level_times(10.0)[1]], rtol=1e-9)
    assert sol.te[1] == sol.te[2] == sol.T[-1] and sol.te[0] < sol.te[1]
    assert _byte_equal(sol.ye[1], sol.ye[2]) and _byte_equal(sol.Y[-1], sol.ye[-1])
    if rootfind == 'left':
        assert sol.ye[1][0] >= 10
    else:
        assert sol.ye[1][0] <= 10


def _height_signs(integ, te, n=6):
    """The signs of the height on the interpolant of the last step at the
    ``2n + 1`` floats around ``te``."""
    taus = [te]
    for _ in range(n):
        taus.insert(0, math.nextafter(taus[0], -math.inf))
        taus.append(math.nextafter(taus[-1], math.inf))
    return [float(np.sign(integ._interpolate((tau - integ.tprev) / integ.dt_step, np.empty(2))[0]))
            for tau in taus]


@pytest.mark.i7b
@pytest.mark.parametrize('rootfind', ['left', 'right'])
@pytest.mark.parametrize('terminal', [[True, True], [True, False]], ids=['both_terminal', 'recorded_twin'])
def test_identical_components_on_a_rounded_interpolant(model, rootfind, terminal):
    """On ``[0, 30]`` Rodas4 crosses the ground in one step of 8.4, and the
    rounded interpolant of the height changes sign several times within a
    few floats of the impact. Two brackets of that one function can end on
    different sign changes, yet the twin components are reported together.

    The sign changes come from the rounding of ``K`` and of the solves, and
    so of the platform's BLAS and linear solver; they were measured on the
    server with both backends. On a platform that rounds so that the
    interpolant changes sign only once, the guard fails, which reports that
    the test no longer exercises what it names instead of passing without
    it."""
    dae, y0 = model('ball')
    cb = ContinuousCallback(lambda t, y, integ: np.array([y[0], y[0]]), direction=-1, terminal=terminal,
                            record=True, rootfind=rootfind)
    integ = init(dae, [0, 30], y0, opt=Opt(rtol=1e-6, atol=1e-8), callbacks=[cb])
    sol = integ.solve()
    signs = _height_signs(integ, sol.te[0])
    assert sum(a != b for a, b in zip(signs, signs[1:])) >= 2, signs
    assert sol.stats.ret == 'terminated'
    assert sol.ie.tolist() == [0, 1] and sol.te[0] == sol.te[1] and _byte_equal(sol.ye[0], sol.ye[1])
    assert abs(sol.te[0] - FIRST_IMPACT) <= 1e-14 * FIRST_IMPACT


def _at_half():
    return _event(lambda t, y: t - 0.5, [1], [0])


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_a_terminal_stop_on_a_grid(model, legacy_compat):
    """``[t0, nodes <= te, te]``, with no duplicate when ``te`` is a node."""
    dae, y0 = model('ball')
    alg = Rodas4(legacy_compat=legacy_compat)
    grid = np.linspace(0, 1, 11)
    sol = alg(dae, grid, y0, Opt(event=_at_half()))
    assert sol.stats.ret == 'terminated' and sol.te.tolist() == [0.5]
    assert _byte_equal(sol.T, grid[:6])
    assert _byte_equal(sol.Y[-1], sol.ye[0])
    grid = np.linspace(0, 1, 8)
    sol = alg(dae, grid, y0, Opt(event=_at_half()))
    assert sol.te.tolist() == [0.5]
    assert _byte_equal(sol.T, np.append(grid[:4], 0.5))
    assert _byte_equal(sol.Y[-1], sol.ye[0])


@pytest.mark.i6a
@pytest.mark.parametrize('method', [Rodas3, Rodas4])
def test_a_terminal_crossing_at_a_step_end_that_is_a_node(model, method):
    """The legacy-compatible configuration saves every node from the
    interpolant, but a node at ``te`` is the state there, so the last row is
    the recorded state. With the fixed step 0.1 the nodes are the step ends,
    the sums of 0.1. The event lies at a step end where the interpolant of
    Rodas4 differs from the step end in the last bits, which depends on the
    rounding of the platform, so the node is chosen where this run shows it;
    the interpolant of Rodas3 returns the step end exactly. In the default
    configuration the fixed step, or a stop time, puts the step end on the
    node."""
    dae, y0 = model('ball')
    fixed = dict(fix_h=True, hinit=0.1)
    ends = method(legacy_compat=True)(dae, [0.0, 1.0], y0, Opt(**fixed))
    grid = ends.T
    assert grid.size == 11
    k = 5
    if method is Rodas4:
        nodes = method(legacy_compat=True)(dae, grid, y0, Opt(**fixed))
        differ = [j for j in range(2, grid.size - 1) if not _byte_equal(nodes.Y[j], ends.Y[j])]
        if not differ:
            pytest.skip('the interpolant of Rodas4 gives every step end exactly on this platform')
        k = differ[0]
    event = _event(lambda t, y: t - grid[k], [1], [0])
    runs = [method(legacy_compat=True)(dae, grid, y0, Opt(event=event, **fixed)),
            solve(dae, grid, y0, alg=method(), opt=Opt(event=event, **fixed)),
            solve(dae, grid, y0, alg=method(), opt=Opt(event=event), tstops=[grid[k]])]
    for sol in runs:
        assert sol.stats.ret == 'terminated' and sol.te.tolist() == [grid[k]]
        assert _byte_equal(sol.T, grid[:k + 1])
        assert _byte_equal(sol.Y[-1], sol.ye[0])


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_a_terminal_event_stops_the_run(model, legacy_compat):
    dae, y0 = model('ball')
    t1, _ = _level_times(10.0)
    integ = init(dae, [0, 4], y0, alg=Rodas4(legacy_compat=legacy_compat),
                 opt=Opt(event=_event(lambda t, y: y[0] - 10, [1], [0])))
    while integ.step():
        pass
    assert integ.terminated and integ.t < 4 and abs(integ.t - t1) <= 1e-9
    assert integ.step() is False
    sol = integ.postamble()
    assert sol.stats.ret == 'terminated' and sol.T[-1] == integ.t == sol.te[0]
    assert _byte_equal(sol.Y[-1], sol.ye[0])


# -- where crossings are and are not reported ---------------------------------

@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_no_event_at_a_start_on_a_root(model, legacy_compat):
    """The ball starts on the ground: a component that is zero at ``t0`` is
    reported only at its crossing, here the impact."""
    dae, y0 = model('ball')
    assert y0[0] == 0
    alg = Rodas4(legacy_compat=legacy_compat)
    sol = alg(dae, [0, 5], y0, Opt(event=_event(lambda t, y: y[0], [0], [0])))
    assert sol.stats.ret == 'success' and sol.ie.tolist() == [0]
    assert abs(sol.te[0] - FIRST_IMPACT) <= 1e-9 * FIRST_IMPACT
    # zero at t0, then negative, then back through zero at the impact
    for direction, expected in ((0, 1), (+1, 1), (-1, 0)):
        sol = alg(dae, [0, 5], y0, Opt(event=_event(lambda t, y: -y[0], [0], [direction])))
        assert (0 if sol.te is None else sol.te.size) == expected
        if expected:
            assert abs(sol.te[0] - FIRST_IMPACT) <= 1e-9 * FIRST_IMPACT


@pytest.mark.i6a
@pytest.mark.parametrize('hmax', [None, 0.01])
def test_a_zero_at_a_step_end_is_reported_once(model, hmax):
    """``g = (0.5 - t)(0.8 - t)`` reaches zero at the stop time 0.5, which is
    reported once, turns negative from there, which is not a crossing, and
    crosses at 0.8. Without ``hmax`` the step from 0.5 reaches past 0.8, so
    the crossing lies in the step that leaves zero."""
    dae, y0 = model('ball')
    event = _event(lambda t, y: (0.5 - t) * (0.8 - t), [0], [0])
    sol = solve(dae, [0, 1], y0, opt=Opt(event=event, hmax=hmax), tstops=[0.5])
    assert sol.stats.ret == 'success' and sol.T[-1] == 1.0
    T = sol.T.tolist()
    k = T.index(0.5)
    assert (T[k + 1] > 0.8) == (hmax is None)
    assert sol.te.tolist() == [0.5, 0.8] and sol.ie.tolist() == [0, 0]
    assert _byte_equal(sol.ye[0], sol.Y[k])


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_a_crossing_just_after_the_start_of_a_long_step(model, legacy_compat):
    """The ball passes ``2e-9`` about ``1e-10`` after the start of a first
    step of length 1."""
    dae, y0 = model('ball')
    # the root of 20 t - 4.9 t**2 = 2e-9 in the form without cancellation
    root = 4e-9 / (20.0 + np.sqrt(400.0 - 4 * 4.9 * 2e-9))
    sol = Rodas4(legacy_compat=legacy_compat)(dae, [0, 10], y0,
                                              Opt(hinit=1.0, event=_event(lambda t, y: y[0] - 2e-9, [1], [0])))
    assert sol.stats.nstep == 1
    assert sol.stats.ret == 'terminated' and abs(sol.te[0] - root) <= 1e-6 * root


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_the_first_of_three_crossings_in_one_step(model, legacy_compat):
    dae, y0 = model('ball')
    cb = ContinuousCallback(lambda t, y, integ: (t - 0.2) * (t - 0.5) * (t - 0.8), terminal=True, record=True)
    sol = solve(dae, [0, 2], y0, alg=Rodas4(legacy_compat=legacy_compat), opt=Opt(hinit=1.0, hmax=1.0),
                callbacks=[cb])
    assert sol.stats.nstep == 1 and sol.stats.ret == 'terminated'
    assert abs(sol.te[0] - 0.2) <= 1e-12 and sol.ie.tolist() == [0]
    assert sol.T[-1] == sol.te[0] and _byte_equal(sol.Y[-1], sol.ye[0])


# steps of 4 on [0, 8]: the ball passes 10 upwards and downwards inside the first step
LONG_STEPS = dict(hinit=4.0, hmax=4.0)


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
@pytest.mark.parametrize('direction', [0, -1, +1])
def test_every_crossing_of_a_recorded_component_in_one_step(model, legacy_compat, direction):
    """A component that only records never shortens the step, so each of its
    crossings in one step is logged that its direction allows: with -1 the
    descent, which follows an ascent against the direction, and with 0
    both. The adapter of ``opt.event`` records the same."""
    dae, y0 = model('ball')
    up, down = _level_times(10.0)
    expected = {0: [up, down], -1: [down], +1: [up]}[direction]
    alg = Rodas4(legacy_compat=legacy_compat)
    cb = ContinuousCallback(lambda t, y, integ: y[0] - 10, direction=direction, record=True)
    runs = [solve(dae, [0, 8], y0, alg=alg, opt=Opt(**LONG_STEPS), callbacks=[cb]),
            alg(dae, [0, 8], y0, Opt(event=_event(lambda t, y: y[0] - 10, [0], [direction]), **LONG_STEPS))]
    for sol in runs:
        assert sol.stats.ret == 'success' and sol.T.tolist() == [0.0, 4.0, 8.0]
        np.testing.assert_allclose(sol.te, expected, rtol=1e-9)
        assert sol.ie.tolist() == [0] * len(expected)
        np.testing.assert_allclose(sol.ye[:, 0], 10.0, rtol=1e-9)


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_every_recorded_crossing_before_te_in_one_step(model, legacy_compat):
    """Both crossings of 10 by a recorded component lie in the first step,
    before the terminal component of another callback stops the run at 3.9,
    and both are logged."""
    dae, y0 = model('ball')
    up, down = _level_times(10.0)
    record = ContinuousCallback(lambda t, y, integ: y[0] - 10, record=True)
    stop = ContinuousCallback(lambda t, y, integ: t - 3.9, terminal=True)
    sol = solve(dae, [0, 8], y0, alg=Rodas4(legacy_compat=legacy_compat), opt=Opt(**LONG_STEPS),
                callbacks=[record, stop])
    assert sol.stats.ret == 'terminated' and sol.stats.nstep == 1
    assert abs(sol.T[-1] - 3.9) <= 1e-15
    np.testing.assert_allclose(sol.te, [up, down], rtol=1e-9)
    assert sol.ie.tolist() == [0, 0]


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_an_acting_crossing_after_one_against_its_direction(model, legacy_compat):
    """Inside the first step the ball passes 10 upwards, against the direction
    -1 of the terminal component, and then downwards, which stops the run."""
    dae, y0 = model('ball')
    _, down = _level_times(10.0)
    cb = ContinuousCallback(lambda t, y, integ: y[0] - 10, direction=-1, terminal=True, record=True)
    sol = solve(dae, [0, 8], y0, alg=Rodas4(legacy_compat=legacy_compat), opt=Opt(**LONG_STEPS),
                callbacks=[cb])
    assert sol.stats.ret == 'terminated' and sol.stats.nstep == 1
    np.testing.assert_allclose(sol.te, [down], rtol=1e-9)
    assert sol.T[-1] == sol.te[0] and _byte_equal(sol.Y[-1], sol.ye[0])


@pytest.mark.i6a
def test_a_crossing_that_enters_and_leaves_within_one_step(model):
    """``g = (t - 0.5)**2 - 1e-4`` is negative on ``(0.49, 0.51)`` only. The
    first step is ``[0, 0.9]``, so that one of its ten sample points is 0.5;
    with no interior sample the crossing is not seen."""
    dae, y0 = model('ball')

    def run(interp_points):
        cb = ContinuousCallback(lambda t, y, integ: (t - 0.5) ** 2 - 1e-4, terminal=True, record=True,
                                interp_points=interp_points)
        return solve(dae, [0, 2], y0, opt=Opt(hinit=0.9, hmax=1.0), callbacks=[cb])

    sol = run(10)
    assert sol.stats.nstep == 1 and sol.stats.ret == 'terminated'
    assert abs(sol.te[0] - 0.49) <= 1e-12
    sol = run(2)
    assert sol.stats.ret == 'success' and sol.te is None and sol.T[-1] == 2.0


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_the_direction_filter(model, legacy_compat):
    dae, y0 = model('ball')
    up, down = _level_times(10.0)
    alg = Rodas4(legacy_compat=legacy_compat)
    for direction, expected in ((0, [up, down]), (+1, [up]), (-1, [down])):
        cb = ContinuousCallback(lambda t, y, integ: y[0] - 10, direction=direction, record=True)
        sol = solve(dae, [0, 4], y0, alg=alg, callbacks=[cb])
        assert sol.stats.ret == 'success'
        np.testing.assert_allclose(sol.te, expected, rtol=1e-9)
        assert sol.ie.tolist() == [0] * len(expected)
        sol = alg(dae, [0, 4], y0, Opt(event=_event(lambda t, y: y[0] - 10, [0], [direction])))
        np.testing.assert_allclose(sol.te, expected, rtol=1e-9)


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
@pytest.mark.parametrize('trait', ['terminal', 'direction'])
def test_the_adapter_reads_its_traits_at_the_end_of_each_step(model, legacy_compat, trait):
    """A legacy ``opt.event`` may change ``isterminal`` and ``direction`` with
    the state. Here they change once the ball falls: the descent through 10
    becomes terminal, or is no longer reported, since only ascents are
    allowed from then on. The steps are at most 0.5, so the ball still rises
    at the end of the step of the ascent and falls at the end of the step of
    the descent."""
    dae, y0 = model('ball')
    up, down = _level_times(10.0)

    def event(t, y):
        falling = int(y[1] < 0)
        if trait == 'terminal':
            return np.array([y[0] - 10]), np.array([falling]), np.array([0])
        return np.array([y[0] - 10]), np.array([0]), np.array([falling])

    sol = Rodas4(legacy_compat=legacy_compat)(dae, [0, 4], y0, Opt(event=event, hmax=0.5))
    if trait == 'terminal':
        assert sol.stats.ret == 'terminated' and sol.T[-1] == sol.te[-1]
        np.testing.assert_allclose(sol.te, [up, down], rtol=1e-9)
    else:
        assert sol.stats.ret == 'success' and sol.T[-1] == 4
        np.testing.assert_allclose(sol.te, [up], rtol=1e-9)


@pytest.mark.i6a
@pytest.mark.parametrize('direction, reported', [(0, True), (+1, True), (-1, False)])
def test_a_component_that_leaves_zero_against_its_direction(model, direction, reported):
    """``g = t (t - 0.5)`` is zero at the start of the first step ``[0, 1]``,
    leaves zero to the negative side and crosses upwards at 0.5, inside the
    same step. With ``direction = -1`` a component on the negative side can
    only cross upwards, which is not allowed, so nothing is reported."""
    dae, y0 = model('ball')
    cb = ContinuousCallback(lambda t, y, integ: t * (t - 0.5), direction=direction, record=True)
    sol = solve(dae, [0, 2], y0, opt=Opt(hinit=1.0, hmax=1.0), callbacks=[cb])
    assert sol.stats.ret == 'success' and sol.T[1] == 1.0
    if reported:
        assert sol.ie.tolist() == [0] and abs(sol.te[0] - 0.5) <= 1e-12
    else:
        assert sol.te is None


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('save_positions, rows', [((True, True), 2), ((True, False), 1), ((False, False), 1)])
def test_the_rows_at_a_terminal_crossing(model, legacy_compat, grid, save_positions, rows):
    """The state at ``te`` is saved once before the event, whether the saving
    of the step has pushed it already, as it does when every step is saved,
    or not, as on a grid whose nodes miss ``te``; once more after the event
    when ``save_positions`` asks for it; and by the end of the run when no
    row at ``te`` was saved."""
    dae, y0 = model('ball')
    cb = ContinuousCallback(lambda t, y, integ: y[0] - 10, direction=-1, terminal=True,
                            save_positions=save_positions)
    tspan = np.linspace(0, 5, 11) if grid else [0, 5]
    sol = solve(dae, tspan, y0, alg=Rodas4(legacy_compat=legacy_compat), callbacks=[cb])
    assert sol.stats.ret == 'terminated'
    te = sol.T[-1]
    assert abs(te - _level_times(10.0)[1]) <= 1e-9
    assert np.count_nonzero(sol.T == te) == rows
    if grid:
        assert _byte_equal(sol.T[:-rows], tspan[tspan < te])


@pytest.mark.i6a
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_the_result_without_and_with_events(model, legacy_compat):
    dae, y0 = model('ball')
    alg = Rodas4(legacy_compat=legacy_compat)
    sol = alg(dae, [0, 1], y0, Opt(event=_event(lambda t, y: y[0] + 100, [1], [0])))
    assert sol.stats.ret == 'success' and sol.te is None and sol.ye is None and sol.ie is None
    sol = alg(dae, [0, 1], y0, Opt(event=_event(lambda t, y: y[0] - 1, [0], [0])))
    assert sol.ie.dtype == np.int64 and sol.te.dtype == np.float64 and sol.ye.shape == (1, 2)


@pytest.mark.i6a
def test_a_left_event_restarted_from_its_state(model):
    """The condition jumps from -1 to +1 between the float ``c`` and the
    next, so ``'left'`` returns ``c``. A new call from ``(c, ye)`` meets the
    same jump between the start of its first step and the next float, and
    reports it at that next float, never at its initial point."""
    dae, y0 = model('ball')
    c = 0.3
    cb = ContinuousCallback(lambda t, y, integ: -1.0 if t <= c else 1.0, terminal=True, record=True,
                            rootfind='left')
    first = solve(dae, [0, 1], y0, callbacks=[cb])
    assert first.stats.ret == 'terminated' and first.te.tolist() == [c] and first.T[-1] == c
    again = solve(dae, [c, 1], first.Y[-1], callbacks=[cb])
    assert again.stats.ret == 'terminated' and again.stats.nstep == 1
    assert again.te.tolist() == [math.nextafter(c, math.inf)]
    assert again.T[0] == c and again.T[-1] == again.te[0]


def _tent(t, y, integ):
    """``t - 0.5`` up to 1.5 and ``2.5 - t`` after it: on the first step
    ``[0, 1]`` the first regula falsi point is 0.5, where the condition is
    exactly zero, whatever the rounding of the platform."""
    return t - 0.5 if t < 1.5 else 2.5 - t


@pytest.mark.i6a
def test_a_left_event_on_an_exact_zero_restarted_from_its_state(model):
    """The search meets an exact zero, which it returns whatever the side, so
    the run stops on the surface. A new call from there starts on a root,
    which is not a crossing, and reports the next crossing."""
    dae, y0 = model('ball')
    cb = ContinuousCallback(_tent, terminal=True, record=True, rootfind='left', interp_points=2)
    opt = Opt(hinit=1.0, hmax=1.0)
    first = solve(dae, [0, 4], y0, opt=opt, callbacks=[cb])
    assert first.stats.nstep == 1 and first.te.tolist() == [0.5] and first.T[-1] == 0.5
    assert _tent(first.te[0], first.ye[0], None) == 0
    again = solve(dae, [0.5, 4], first.Y[-1], opt=opt, callbacks=[cb])
    assert again.stats.ret == 'terminated' and again.T[0] == 0.5
    assert abs(again.te[0] - 2.5) <= 1e-12


@pytest.mark.i6a
def test_find_root_returns_adjacent_floats():
    """The bracket shrinks to two adjacent floats, from which ``'left'``
    takes the lower and ``'right'`` the upper, except that ``'left'`` never
    returns the start of the step; an exact zero met on the way is returned
    whatever the side."""
    def jump(c, strict):
        return lambda tau: -1.0 if (tau < c if strict else tau <= c) else 1.0

    below = math.nextafter(0.3, -math.inf)
    for tstart in (0.0, 0.25):
        assert find_root(jump(0.3, True), 0.25, 1.0, -1.0, 1.0, 'left', tstart) == below
        assert find_root(jump(0.3, True), 0.25, 1.0, -1.0, 1.0, 'right', tstart) == 0.3
    # the crossing between the bottom of the bracket and the next float
    after = math.nextafter(0.25, math.inf)
    g = jump(0.25, False)
    assert find_root(g, 0.25, 1.0, -1.0, 1.0, 'left', 0.0) == 0.25
    assert find_root(g, 0.25, 1.0, -1.0, 1.0, 'left', 0.25) == after
    assert find_root(g, 0.25, 1.0, -1.0, 1.0, 'right', 0.25) == after
    for side in ('left', 'right'):
        # the first regula falsi point of a linear function on [0, 1]
        assert find_root(lambda tau: tau - 0.5, 0.0, 1.0, -0.5, 0.5, side, 0.0) == 0.5
        assert find_root(jump(0.3, True), 0.0, 1.0, -1.0, 0.0, side, 0.0) == 1.0


@pytest.mark.i6a
@pytest.mark.parametrize('rootfind', ['left', 'right'])
def test_a_component_crossing_after_te_costs_no_search(model, rootfind):
    """In the one bracket ``[0, 1]`` of each component, the terminal
    component 0 crosses at 0.5, where the first regula falsi point finds an
    exact zero, and the recorded component 1 jumps at 0.7, which a search
    would locate in about fifty calls. Component 1 costs the probe after
    ``te`` only: for ``'right'`` the probe is ``te``, whose value is known,
    and for ``'left'`` the float after it."""
    dae, y0 = model('ball')
    calls = []

    def condition(t, y, integ):
        calls.append(t)
        return np.array([t - 0.5, 1.0 if t < 0.7 else -1.0])

    cb = ContinuousCallback(condition, terminal=[True, False], record=True, rootfind=rootfind, interp_points=2)
    sol = solve(dae, [0, 2], y0, opt=Opt(hinit=1.0, hmax=1.0), callbacks=[cb])
    assert sol.stats.nstep == 1 and sol.stats.ret == 'terminated'
    assert sol.te.tolist() == [0.5] and sol.ie.tolist() == [0]
    probe = [math.nextafter(0.5, math.inf)] if rootfind == 'left' else []
    assert calls == [0.0, 1.0, 0.5] + probe
    assert sol.stats.ncondition == len(calls)


class _Step:
    """The accepted step ``[tprev, t]`` of an integrator, as the detection
    reads it, for a condition of ``t`` alone."""

    def __init__(self, tprev):
        self.stats = SimpleNamespace(ncondition=0)
        self.tprev = self.t = tprev
        self.u = np.zeros(1)
        self._cb_y = np.zeros(1)

    def interp(self, tau, out):
        return out


def _nudged(condition, repeat_nudge, interp_points, fired):
    """The brackets of the step ``[0, 1]`` after an event at 0 of the
    components ``fired`` of a ``'left'`` callback."""
    cb = ContinuousCallback(lambda t, y, integ: condition(t), record=True, repeat_nudge=repeat_nudge,
                            interp_points=interp_points)
    integ = _Step(0.0)
    st = _ContinuousState(cb, 0, integ)
    integ.t = 1.0
    st.fired, st.fired_t = np.array(fired), 0.0
    st.fired_g = st.g0[st.fired]
    return {c.i: (c.bottom, c.top) for c in _brackets(st, integ)}


@pytest.mark.i6a
def test_the_repeat_nudge():
    """A component that fired at the start of the step and has crossed by
    ``tn`` has no event in the step; otherwise its bracket starts at ``tn``,
    and when ``tn`` lies at or after the top of its bracket, the next
    bracket after ``tn`` is searched. A component that did not fire keeps
    its bracket. The nudge takes effect once the modification protocol
    stores the fired components and their values at the event, so the
    stored state is set here, with the values left unchanged."""
    def condition(t):
        return np.array([0.05 - t, 0.3 - t, 0.3 - t])

    assert _nudged(condition, 0.1, 3, [0, 1]) == {1: (0.1, 0.5), 2: (0.0, 0.5)}

    def condition(t):
        return np.array([0.3 - t, (t - 0.2) * (t - 0.55) * (t - 0.8), 0.3 - t, (t - 0.2) * (t - 0.55)])

    # tn = 0.6 lies after the first sample 0.5: component 0 has crossed by tn,
    # 1 and 3 are back on their side, and only 1 crosses again, at 0.8
    assert _nudged(condition, 0.6, 3, [0, 1, 3]) == {1: (0.6, 1.0), 2: (0.0, 0.5)}

    # a component that only records and has crossed by tn = 0.1, the event
    # already reported, is followed from tn and crosses back after 0.5
    def condition(t):
        return np.array([(0.05 - t) * (0.6 - t)])

    assert _nudged(condition, 0.1, 3, [0]) == {0: (0.5, 1.0)}


def _all_brackets(condition, **kwargs):
    """``(i, bottom, top)`` of every bracket in the step ``[0, 1]``, sampled at
    multiples of 0.2."""
    cb = ContinuousCallback(lambda t, y, integ: condition(t), interp_points=6, **kwargs)
    integ = _Step(0.0)
    st = _ContinuousState(cb, 0, integ)
    integ.t = 1.0
    return sorted((c.i, c.bottom, c.top) for c in _brackets(st, integ))


@pytest.mark.i6a
def test_the_brackets_of_every_crossing_of_a_recorded_component():
    """``g = (t - 0.1)(t - 0.5)(t - 0.7)`` changes sign between the samples 0
    and 0.2, 0.4 and 0.6, and 0.6 and 0.8. A component that only records has
    a bracket for each change its direction allows, and a change against its
    direction moves its reference only; a terminal twin has the bracket of
    its first crossing only."""
    def g(t):
        return (t - 0.1) * (t - 0.5) * (t - 0.7)

    def twins(t):
        return np.array([g(t), g(t)])

    assert _all_brackets(twins, record=True, terminal=[False, True]) == [
        (0, 0.0, 0.2), (0, 0.4, 0.6), (0, 0.6, 0.8), (1, 0.0, 0.2)]
    assert _all_brackets(twins, record=True, terminal=[False, True], direction=-1) == [
        (0, 0.4, 0.6), (1, 0.4, 0.6)]
    assert _all_brackets(twins, record=True, terminal=[False, True], direction=+1) == [
        (0, 0.0, 0.2), (0, 0.6, 0.8), (1, 0.0, 0.2)]


@pytest.mark.i7b
def test_a_root_found_later_lowers_te_below_an_earlier_one():
    """Two terminal components share the bracket ``[0, 1]``. Component 0
    changes sign at 0.1, 0.2 and 0.7, and its search, from the regula falsi
    point 0.5, ends on 0.7; component 1 crosses at 0.15, which lowers ``te``
    to 0.15. Component 0 has crossed by then, so it is located again against
    0.15, which finds 0.1, the earliest crossing of the step."""
    def g0(t):
        return 1.0 if 0.1 <= t < 0.2 or t >= 0.7 else -1.0

    def g1(t):
        return 1.0 if t >= 0.15 else -1.0

    cb = ContinuousCallback(lambda t, y, integ: np.array([g0(t), g1(t)]), terminal=True, rootfind='right',
                            interp_points=2)
    integ = _Step(0.0)
    st = _ContinuousState(cb, 0, integ)
    integ.t = 1.0
    te, found = locate(integ, [st])
    assert te == 0.1
    assert [(c.i, c.root) for c in found] == [(0, 0.1)]


# -- the step as computed, the counts and the arguments -----------------------

@pytest.mark.i6a
@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('legacy_compat', CONFIGS)
def test_rows_before_a_terminal_event_are_those_of_the_run_without_it(model, scheme, legacy_compat):
    """The state moves to ``te``, but the step's interpolant reads the end
    state as computed, so every node before ``te`` is saved as without the
    event."""
    dae, y0 = model('ball')
    grid = np.linspace(0, 30, 61)
    alg = Rosenbrock.from_scheme(scheme, legacy_compat=legacy_compat)
    opt = dict(rtol=1e-6, atol=1e-8)
    with_event = alg(dae, grid, y0, Opt(event=_event(lambda t, y: y[0] - 10, [1], [-1]), **opt))
    without = alg(dae, grid, y0, Opt(**opt))
    k = int(np.searchsorted(grid, with_event.te[0]))
    assert k >= 2 and grid[k - 1] < with_event.te[0]
    assert _byte_equal(with_event.T[:k], without.T[:k])
    assert _byte_equal(with_event.Y[:k], without.Y[:k])


@pytest.mark.i6a
def test_the_condition_calls_are_counted(model):
    """The adapter is called at the end of every accepted step and at eight
    interior samples of every step with a component that may cross."""
    dae, y0 = model('ball')
    calls = []

    def event(t, y):
        calls.append(t)
        return np.array([y[0] + 100]), np.array([0]), np.array([0])

    sol = Rodas4()(dae, [0, 1], y0, Opt(event=event))
    assert sol.stats.ncondition == len(calls) == 1 + 9 * sol.stats.nstep
    calls.clear()
    sol = Rodas4()(dae, [0, 5], y0, Opt(event=_counted(calls, lambda t, y: y[0] - 10)))
    assert sol.stats.ncondition == len(calls) and sol.te.size == 2


def _counted(calls, value):
    def event(t, y):
        calls.append(t)
        return np.array([value(t, y)]), np.array([0]), np.array([0])
    return event


@pytest.mark.i6a
def test_callback_arguments(model):
    cond = lambda t, y, integ: y[0]
    for kwargs in (dict(direction=2), dict(direction=0.5), dict(direction=[[1]]), dict(rootfind='middle'),
                   dict(interp_points=-1), dict(interp_points=2.5), dict(repeat_nudge=1.0),
                   dict(save_positions=(True,)), dict(terminal=[[True]])):
        with pytest.raises(ValueError):
            ContinuousCallback(cond, **kwargs)
    with pytest.raises(TypeError):
        ContinuousCallback(None)
    dae, y0 = model('ball')
    recording = ContinuousCallback(cond, record=True)
    with pytest.raises(ValueError, match='at most one'):
        solve(dae, [0, 1], y0, callbacks=[recording, ContinuousCallback(cond, record=True)])
    with pytest.raises(ValueError, match='at most one'):
        solve(dae, [0, 1], y0, opt=Opt(event=_at_half()), callbacks=[recording])
    with pytest.raises(TypeError, match='ContinuousCallback'):
        solve(dae, [0, 1], y0, callbacks=[lambda t, y, integ: y[0]])
    # traits of the wrong length, and a condition whose length changes
    with pytest.raises(ValueError, match='direction'):
        solve(dae, [0, 1], y0, callbacks=[ContinuousCallback(cond, direction=[1, -1])])
    with pytest.raises(ValueError, match='fixed length'):
        solve(dae, [0, 1], y0, callbacks=[ContinuousCallback(lambda t, y, integ: y[:1] if t == 0 else y)])


# -- affects ------------------------------------------------------------------

@pytest.mark.i6b
def test_ten_bounces_in_one_call(model):
    """The bounces of ``test_ten_bounces_as_the_legacy_test`` in one call: a
    ``'left'`` callback reverses 0.9 of the velocity at each impact. The
    state at ``te`` lies just above the ground, so the next step starts on
    the side the impact was reported from, and each bounce is reported
    once, with the rows before and after the affect."""
    dae, y0 = model('ball')

    def bounce(integ, idx):
        assert idx.dtype == np.int64 and idx.tolist() == [0]
        integ.u[1] = -0.9 * integ.u[1]

    cb = ContinuousCallback(lambda t, y, integ: y[0], bounce, direction=-1, record=True, rootfind='left')
    sol = solve(dae, [0, 27], y0, callbacks=[cb])
    assert sol.stats.ret == 'success' and sol.T[-1] == 27.0
    assert sol.ie.tolist() == [0] * 10 and np.all(np.diff(sol.te) > 0)
    np.testing.assert_allclose(sol.te, LEGACY_BOUNCES, rtol=1e-5, atol=0)
    for te, ye in zip(sol.te, sol.ye):
        k = np.flatnonzero(sol.T == te)
        assert k.size == 2 and k[1] == k[0] + 1
        before, after = sol.Y[k[0]], sol.Y[k[1]]
        assert _byte_equal(before, ye) and ye[0] >= 0 and ye[1] < 0
        assert after[0] == before[0] and after[1] == -0.9 * before[1]
    exact = np.cumsum(2 * 20 * 0.9 ** np.arange(10) / G)
    print(f"max relative deviation of the bounce times: from legacy "
          f"{np.max(np.abs(sol.te / LEGACY_BOUNCES - 1)):.2e}, from the exact times "
          f"{np.max(np.abs(sol.te / exact - 1)):.2e}")


def _bounded(integ, nsteps=1000):
    """``integ`` run to the end in at most ``nsteps`` steps, so that a
    crossing met again at every step fails the test instead of advancing
    the run by one float per step."""
    for _ in range(nsteps):
        if not integ.step():
            return integ.postamble()
    pytest.fail(f"the run is at t = {integ.t!r} after {nsteps} steps")


@pytest.mark.i6b
@pytest.mark.parametrize('direction', [0, -1])
def test_an_affect_that_leaves_the_state_unchanged_reports_each_crossing_once(model, direction):
    """With ``'left'`` the state at ``te`` has not crossed yet, so the next
    step starts before the crossing it was reported at, and meets it again
    within its first float. The repeat nudge, which reads the components the
    modification protocol stored, tells it apart from a new crossing."""
    dae, y0 = model('ball')
    at = []
    cb = ContinuousCallback(lambda t, y, integ: y[0] - 10, lambda integ, idx: at.append(integ.t),
                            direction=direction, record=True, rootfind='left')
    sol = _bounded(init(dae, [0, 4], y0, callbacks=[cb]))
    up, down = _level_times(10.0)
    expected = [up, down] if direction == 0 else [down]
    assert sol.stats.ret == 'success'
    np.testing.assert_allclose(sol.te, expected, rtol=1e-9)
    assert sol.te.tolist() == at and sol.ie.tolist() == [0] * len(expected)


@pytest.mark.i6b
def test_a_recorded_left_crossing_at_the_time_of_an_affect_is_logged_once(model):
    """A callback that only records crosses at the ``te`` of another
    callback's affect, both with ``'left'``. The modification protocol stores
    the recorded component as well, so the next step, which meets the same
    crossing within its first float, does not log it again."""
    dae, y0 = model('ball')
    at = []
    acting = ContinuousCallback(lambda t, y, integ: y[0] - 10, lambda integ, idx: at.append(integ.t),
                                direction=-1, rootfind='left')
    recording = ContinuousCallback(lambda t, y, integ: y[0] - 10, direction=-1, record=True, rootfind='left')
    sol = _bounded(init(dae, [0, 4], y0, callbacks=[acting, recording]))
    assert sol.stats.ret == 'success' and len(at) == 1
    assert sol.te.tolist() == at and sol.ie.tolist() == [0]


@pytest.mark.i6b
@pytest.mark.parametrize('later', ['discrete', 'model_modified'])
def test_a_later_change_at_the_same_time_keeps_the_stored_components(model, later):
    """After the affect at the descent through 10, a discrete callback, or
    ``model_modified()`` between two steps, runs the modification protocol
    once more at the same time. That protocol stores no crossing of its own
    and keeps the components the first one stored, so the next step still
    tells the crossing it meets within its first float from a new one."""
    dae, y0 = model('ball')
    at = []
    callbacks = [ContinuousCallback(lambda t, y, integ: y[0] - 10, lambda integ, idx: at.append(integ.t),
                                    direction=-1, record=True, rootfind='left')]
    if later == 'discrete':
        callbacks.append(DiscreteCallback(lambda t, y, integ: at[-1:] == [t], lambda integ: None))
    integ = init(dae, [0, 4], y0, callbacks=callbacks)
    while not at:
        assert integ.step()
    if later == 'model_modified':
        integ.model_modified()
    assert integ.model_epoch == 2 and integ.t == at[0]
    sol = _bounded(integ)
    assert sol.stats.ret == 'success' and len(at) == 1
    assert sol.te.tolist() == at and sol.ie.tolist() == [0]


@pytest.mark.i6b
@pytest.mark.parametrize('rootfind', ['left', 'right'])
def test_an_affect_that_moves_the_component_off_the_surface_ends_the_nudge(model, rootfind):
    """The affect at the descent through 15 puts the ball back at 16 with the
    velocity -2000, so it crosses 15 again about ``1/2000`` later, inside
    the first percent of the next step, which has the length of the
    crossing step. The component is no longer at its value at the event, so
    with ``'left'`` the repeat nudge does not take that crossing for the
    one already reported, and its affect ends the run."""
    dae, y0 = model('ball')
    at = []

    def affect(integ, idx):
        at.append(integ.t)
        if len(at) == 1:
            integ.u[0], integ.u[1] = 16.0, -2000.0
        else:
            integ.terminate()

    cb = ContinuousCallback(lambda t, y, integ: y[0] - 15, affect, direction=-1, record=True,
                            rootfind=rootfind)
    integ = init(dae, [0, 5], y0, callbacks=[cb])
    while not at:
        assert integ.step()
    integ.step()
    sol = integ.postamble()
    assert sol.stats.ret == 'terminated' and len(at) == 2
    assert sol.te.tolist() == at and sol.ie.tolist() == [0, 0]
    np.testing.assert_allclose(at[1] - at[0], 1 / 2000, rtol=1e-3)
    # the second crossing lies where the nudge reads a crossing as the event already reported
    assert at[1] - at[0] < cb.repeat_nudge * integ.dt_step


@pytest.mark.i6b
def test_callbacks_crossing_at_the_same_te(model):
    """Two acting callbacks and a recording one cross at the same float on
    the descent through 10. Both affects run in list order at ``te`` with
    the state before any change, the recording callback logs its component
    there as well, and the modification protocol runs once. The component
    of the recording callback that passes 15 earlier is logged at its own
    time."""
    dae, y0 = model('ball')
    calls = []

    def note(integ, idx):
        calls.append(('a', integ.t, idx.tolist(), integ.model_epoch, integ.u[1]))

    def reverse(integ, idx):
        calls.append(('c', integ.t, idx.tolist(), integ.model_epoch, integ.u[1]))
        integ.u[1] = -integ.u[1]

    a = ContinuousCallback(lambda t, y, integ: y[0] - 10, note, direction=-1)
    b = ContinuousCallback(lambda t, y, integ: np.array([y[0] - 15, y[0] - 10]), direction=-1, record=True)
    c = ContinuousCallback(lambda t, y, integ: y[0] - 10, reverse, direction=-1)
    integ = init(dae, [0, 5], y0, callbacks=[a, b, c])
    while not calls:
        assert integ.step()
    te = integ.t
    assert integ.model_epoch == 1 and integ.u[1] > 0
    assert [x[:4] for x in calls] == [('a', te, [0], 0), ('c', te, [0], 0)]
    assert calls[0][4] == calls[1][4] < 0
    sol = integ.solve()
    assert sol.stats.ret == 'success' and integ.model_epoch == 1 and len(calls) == 2
    assert sol.ie.tolist() == [0, 1]
    np.testing.assert_allclose(sol.te, [_level_times(15.0)[1], _level_times(10.0)[1]], rtol=1e-9)
    assert sol.te[1] == te and sol.ye[1][1] < 0
    k = np.flatnonzero(sol.T == te)
    assert k.size == 2 and _byte_equal(sol.Y[k[0]], sol.ye[1]) and sol.Y[k[1]][1] == -sol.ye[1][1]


class _LinearRodas4(Rosenbrock):
    """Rodas4 with the linear interpolant of ``Algorithm``, which reads the
    end state of the step as the interpolants of ``ImplicitEuler`` and
    ``Trapezoid`` do."""

    scheme = 'rodas4_linear'
    tableau = Rodas_param('rodas4')
    interpolation = 'linear'


@pytest.mark.i6b
@pytest.mark.parametrize('method', [Rodas3, _LinearRodas4, ImplicitEuler, Trapezoid])
def test_rows_before_an_acting_event_are_those_of_the_run_without_it(model, method):
    """The affect reverses the velocity at the descent through 10, and the
    state moves to ``te`` before it; the interpolants read the end state of
    the step as computed, so every node before ``te`` is saved as without
    the event. The grid puts nodes inside the crossing step before ``te``
    for every method, which only the interpolant of that step gives.

    The steps of ``ImplicitEuler`` and ``Trapezoid`` are shorter than the
    spacing of the grid, so the grid also gets a node halfway between the
    start of the crossing step and ``te``, both taken from runs without a
    grid; nodes change neither the steps nor ``te``."""
    dae, y0 = model('ball')
    opt = Opt(rtol=1e-6, atol=1e-8)

    def reverse(integ, idx):
        integ.u[1] = -integ.u[1]

    cb = ContinuousCallback(lambda t, y, integ: y[0] - 10, reverse, direction=-1, record=True)
    steps = solve(dae, [0, 30], y0, alg=method(), opt=opt).T
    te = solve(dae, [0, 30], y0, alg=method(), opt=opt, callbacks=[cb]).te[0]
    start = steps[np.searchsorted(steps, te) - 1]
    grid = np.union1d(np.linspace(0, 30, 601), [0.5 * (start + te)])
    integ = init(dae, grid, y0, alg=method(), opt=opt, callbacks=[cb])
    while integ.model_epoch == 0:
        assert integ.step()
    tprev = integ.tprev
    with_event = integ.solve()
    without = solve(dae, grid, y0, alg=method(), opt=opt)
    te = with_event.te[0]
    assert np.any((grid > tprev) & (grid < te))
    k = int(np.searchsorted(grid, te))
    assert k >= 2 and grid[k - 1] < te
    assert _byte_equal(with_event.T[:k], without.T[:k])
    assert _byte_equal(with_event.Y[:k], without.Y[:k])
    # the rows at te, before and after the affect, follow
    assert with_event.T[k] == with_event.T[k + 1] == te
