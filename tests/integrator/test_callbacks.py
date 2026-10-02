"""Discrete callbacks, affects between steps and the modification protocol.

The state or the model changes inside a call in a callback's affect, or
between two ``step()`` calls followed by ``model_modified()``. Either way
the core makes the state consistent with ``DaeIc``, derives nothing more
from the old model, returns the linear-solver cache to the state of a new
call, and saves the rows before and after the change that
``save_positions`` asks for. A run changed at ``t1`` is then byte-equal
after ``t1`` to two calls split there, on either backend. A ``DaeIc``
failure ends the run with the rows saved so far and never raises, and
``model_modified()`` after a failure does nothing.
"""
import math

import numpy as np
import pytest

from Solverz import Eqn, Model, Ode, Param, Var, made_numerical
from Solverz.integrator import (ContinuousCallback, DiscreteCallback, Rodas3, Rodas4, init,
                                preset_time_callback, solve)
from Solverz.solvers import klu_backend
from Solverz.solvers.klu_backend import KLU_AVAILABLE, set_klu_matching
from Solverz.solvers.laesolver import linsolver
from Solverz.solvers.option import Opt

pytestmark = pytest.mark.i6b

TIGHT = dict(rtol=1e-8, atol=1e-10)


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _numerical(m):
    sdae, y0 = m.create_instance()
    return made_numerical(sdae, y0, sparse=True), np.array(y0.array, dtype=np.float64)


def _relay(k=1.0):
    """``x' = -x + z``, ``0 = z - k`` from ``x = 0``, ``z = k``: with ``k``
    fixed, ``x = k (1 - exp(-t))``. Each test builds its own model, since
    the affects change its parameters and its mass matrix."""
    m = Model()
    m.x = Var('x', [0.0])
    m.z = Var('z', [k])
    m.k = Param('k', [k])
    m.fx = Ode('fx', f=-m.x + m.z, diff_var=m.x)
    m.gz = Eqn('gz', m.z - m.k)
    return _numerical(m)


def _oscillator():
    """``x' = -x + z1``, ``0 = z1 - k z2 - s``, ``0 = z1 + z2 - c``, ``s' =
    c``, ``c' = -s`` from ``x = s = 0``, ``c = 1`` and the consistent ``z1 =
    0.375``, ``z2 = 0.625`` of ``k = 0.6``: the forcing is carried by an
    oscillator, so the model is autonomous. In every iteration matrix the
    maximum-product row matching pairs the two algebraic rows with ``z1``
    and ``z2`` in that order for ``k < 1`` and in the reverse order for ``k
    > 1``."""
    m = Model()
    m.x = Var('x', [0.0])
    m.z1 = Var('z1', [0.375])
    m.z2 = Var('z2', [0.625])
    m.s = Var('s', [0.0])
    m.c = Var('c', [1.0])
    m.k = Param('k', [0.6])
    m.fx = Ode('fx', f=-m.x + m.z1, diff_var=m.x)
    m.g1 = Eqn('g1', m.z1 - m.k * m.z2 - m.s)
    m.g2 = Eqn('g2', m.z1 + m.z2 - m.c)
    m.fs = Ode('fs', f=m.c, diff_var=m.s)
    m.fc = Ode('fc', f=-m.s, diff_var=m.c)
    return _numerical(m)


def _set_k(value):
    def affect(integ):
        integ.dae.p['k'][0] = value
    return affect


def _rows_at(sol, t):
    return np.flatnonzero(sol.T == t)


def _to_half(integ):
    while integ.t < 0.5:
        assert integ.step()
    assert integ.t == 0.5


# -- rows before and after a change -------------------------------------------

@pytest.mark.parametrize('grid', [False, True])
def test_a_preset_time_changes_a_parameter_of_an_algebraic_equation(grid):
    """At 0.5 ``k`` goes from 1 to 2. The rows before and after the affect are
    both saved at 0.5, on a grid that does not contain 0.5 as well, and
    ``DaeIc`` moves ``z`` to the new solution of its equation."""
    dae, y0 = _relay()
    tspan = np.linspace(0, 1, 4) if grid else [0, 1]
    sol = solve(dae, tspan, y0, opt=Opt(**TIGHT), callbacks=[preset_time_callback([0.5], _set_k(2.0))])
    assert sol.stats.ret == 'success' and sol.T[-1] == 1.0
    k = _rows_at(sol, 0.5)
    assert k.size == 2 and k[1] == k[0] + 1
    left, right = sol.Y[k[0]], sol.Y[k[1]]
    assert abs(left[1] - 1.0) <= 1e-6 and abs(right[1] - 2.0) <= 1e-6
    assert right[0] == left[0]
    if grid:
        assert _byte_equal(np.delete(sol.T, k), tspan)
    x05 = 1 - math.exp(-0.5)
    assert abs(left[0] - x05) <= 1e-7
    assert abs(sol.Y[-1][0] - (2 + (x05 - 2) * math.exp(-0.5))) <= 1e-7


@pytest.mark.parametrize('grid, rows', [(False, 2), (True, 1)])
def test_save_positions_false_true(grid, rows):
    """With ``save_positions=(False, True)`` a grid that does not contain 0.5
    gets the row after the affect only. When every step is saved, the
    step's own row at 0.5 is saved before the affect as well, as in SciML."""
    dae, y0 = _relay()
    tspan = np.linspace(0, 1, 4) if grid else [0, 1]
    cb = preset_time_callback([0.5], _set_k(2.0), save_positions=(False, True))
    sol = solve(dae, tspan, y0, callbacks=[cb])
    k = _rows_at(sol, 0.5)
    assert k.size == rows
    assert abs(sol.Y[k[-1]][1] - 2.0) <= 1e-6
    if rows == 2:
        assert abs(sol.Y[k[0]][1] - 1.0) <= 1e-6


def test_save_positions_false_false_on_a_grid_that_contains_the_time():
    dae, y0 = _relay()
    grid = np.linspace(0, 1, 5)
    cb = preset_time_callback([0.5], _set_k(2.0), save_positions=(False, False))
    sol = solve(dae, grid, y0, callbacks=[cb])
    assert _byte_equal(sol.T, grid)
    assert abs(sol.Y[2][1] - 1.0) <= 1e-6 and abs(sol.Y[3][1] - 2.0) <= 1e-6


# -- the mass matrix, the state, and model_modified ---------------------------

def _slow_down(integ):
    """``2 x' = -x + z``, a change of ``M.data`` in place with the pattern
    kept, as ``ModeSwitch`` makes it."""
    integ.dae.M.data[0] = 2.0


def _slowed_exact(t):
    x05 = 1 - math.exp(-0.5)
    return 1 + (x05 - 1) * math.exp(-(t - 0.5) / 2)


def test_a_change_of_the_mass_matrix_at_a_stop_time_is_used_by_the_next_step():
    dae, y0 = _relay()
    M = dae.M
    sol = solve(dae, [0, 2], y0, opt=Opt(**TIGHT), callbacks=[preset_time_callback([0.5], _slow_down)])
    assert dae.M is M and sol.stats.ret == 'success'
    after = sol.T > 0.5
    err = np.max(np.abs(sol.Y[after, 0] - [_slowed_exact(t) for t in sol.T[after]]))
    assert err <= 1e-7


def test_a_row_made_algebraic_at_a_stop_time():
    """``M.data[0] = 0`` keeps the entry as an explicit zero and turns the
    equation of ``x`` into ``0 = -x + z``; ``DaeIc`` then moves ``x`` onto
    it, and it stays there."""
    dae, y0 = _relay()

    def pin(integ):
        integ.dae.M.data[0] = 0.0

    sol = solve(dae, [0, 1], y0, opt=Opt(**TIGHT), callbacks=[preset_time_callback([0.5], pin)])
    assert sol.stats.ret == 'success' and dae.M.nnz == 1
    k = _rows_at(sol, 0.5)
    assert abs(sol.Y[k[0]][0] - (1 - math.exp(-0.5))) <= 1e-7
    assert np.all(np.abs(sol.Y[k[1]:, 0] - 1.0) <= 1e-6)


def test_model_modified_between_steps():
    """The change of ``M.data`` between two ``step()`` calls, followed by
    ``model_modified()``, is used by the next step, gives the run of the
    same change made by a discrete callback that saves no row of its own,
    and leaves ``interp`` only ``tq == t`` until the next step."""
    dae, y0 = _relay()
    integ = init(dae, [0, 2], y0, opt=Opt(**TIGHT), tstops=[0.5])
    while integ.t < 0.5:
        assert integ.step()
    assert integ.t == 0.5
    tprev, u = integ.tprev, integ.u.copy()
    dae.M.data[0] = 2.0
    integ.model_modified()
    assert integ.model_epoch == 1 and _byte_equal(integ.interp(0.5), u)
    with pytest.raises(ValueError, match='only tq == t'):
        integ.interp(tprev)
    with pytest.raises(ValueError, match='only tq == t'):
        integ.interp(0.5 * (tprev + 0.5))
    assert integ.step()
    assert integ.tprev == 0.5
    integ.interp(0.5 * (integ.tprev + integ.t))
    sol = integ.solve()

    dae2, y02 = _relay()
    cb = preset_time_callback([0.5], _slow_down, save_positions=(False, False))
    ref = solve(dae2, [0, 2], y02, opt=Opt(**TIGHT), callbacks=[cb])
    assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y)
    after = sol.T > 0.5
    assert np.max(np.abs(sol.Y[after, 0] - [_slowed_exact(t) for t in sol.T[after]])) <= 1e-7


def test_an_interpolation_after_an_affect_describes_the_step_as_computed():
    """The affect at 0.5 sets ``x`` to 0 and doubles ``M.data[0]``. Between
    the two ``step()`` calls, the interpolant of Rodas3 inside the step that
    ended at 0.5 still reads the end state and the slopes of that step, as
    in the run without the affect, and ``interp(0.5)`` is the changed
    state."""
    def change(integ):
        integ.u[0] = 0.0
        integ.dae.M.data[0] = 2.0

    dae, y0 = _relay()
    ref = init(dae, [0, 1], y0, alg=Rodas3(), tstops=[0.5])
    _to_half(ref)
    dae, y0 = _relay()
    integ = init(dae, [0, 1], y0, alg=Rodas3(), callbacks=[preset_time_callback([0.5], change)])
    _to_half(integ)
    assert integ.model_epoch == 1 and integ.u[0] == 0.0
    tq = 0.5 * (integ.tprev + 0.5)
    assert integ.tprev == ref.tprev and _byte_equal(integ.interp(tq), ref.interp(tq))
    assert _byte_equal(integ.interp(0.5), integ.u)


def test_an_interpolation_after_a_continuous_affect_at_a_step_end_describes_the_step_as_computed():
    """``t - 0.5`` crosses at the stop time 0.5 with ``'right'``, so ``te`` is
    the end of the step and ``interp(te)`` needs no interpolant, and without
    interior samples the location interpolates nowhere either. The data of
    the Rodas3 interpolant are still computed before the affect, which sets
    ``x`` to 0 and doubles ``M.data[0]``, so an interpolation inside the step
    is that of the run without the affect."""
    def change(integ, idx):
        integ.u[0] = 0.0
        integ.dae.M.data[0] = 2.0

    dae, y0 = _relay()
    ref = init(dae, [0, 1], y0, alg=Rodas3(), tstops=[0.5])
    _to_half(ref)
    dae, y0 = _relay()
    cb = ContinuousCallback(lambda t, y, integ: t - 0.5, change, rootfind='right', interp_points=2)
    integ = init(dae, [0, 1], y0, alg=Rodas3(), callbacks=[cb], tstops=[0.5])
    _to_half(integ)
    assert integ.t_step == 0.5 and integ.model_epoch == 1 and integ.u[0] == 0.0
    tq = 0.5 * (integ.tprev + 0.5)
    assert integ.tprev == ref.tprev and _byte_equal(integ.interp(tq), ref.interp(tq))
    assert _byte_equal(integ.interp(0.5), integ.u)


def test_a_change_of_the_state_between_steps():
    """``x`` is set to 0 at 0.5: the next step starts from it, as when a
    discrete callback makes the same change."""
    dae, y0 = _relay()
    integ = init(dae, [0, 1], y0, tstops=[0.5])
    while integ.t < 0.5:
        assert integ.step()
    integ.u[0] = 0.0
    integ.model_modified()
    integ.step()
    assert integ.tprev == 0.5 and integ.uprev[0] == 0.0
    sol = integ.solve()

    def reset(integ):
        integ.u[0] = 0.0

    dae2, y02 = _relay()
    ref = solve(dae2, [0, 1], y02, callbacks=[preset_time_callback([0.5], reset, save_positions=(False, False))])
    assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y)


def test_an_affect_that_rebinds_the_parameters():
    """The affect at 0.5 binds ``dae.p`` to a new dict in which ``k`` is 2.
    The modification protocol reads it, so ``DaeIc`` and every later step
    use it, and the run is that of the same change made in place."""
    dae, y0 = _relay()
    old = dae.p

    def rebind(integ):
        integ.dae.p = {**integ.dae.p, 'k': np.array([2.0])}

    sol = solve(dae, [0, 1], y0, opt=Opt(**TIGHT), callbacks=[preset_time_callback([0.5], rebind)])
    assert dae.p is not old and old['k'][0] == 1.0
    dae2, y02 = _relay()
    ref = solve(dae2, [0, 1], y02, opt=Opt(**TIGHT), callbacks=[preset_time_callback([0.5], _set_k(2.0))])
    assert ref.stats.ret == 'success' and abs(ref.Y[-1][1] - 2.0) <= 1e-6
    assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y)


def test_model_modified_after_a_rebinding_of_the_mass_matrix():
    """``dae.M`` is bound to a copy with ``M.data[0] = 2`` between two
    ``step()`` calls; after ``model_modified()`` the next steps use it, and
    the run is that of the same change made in place."""
    dae, y0 = _relay()
    integ = init(dae, [0, 2], y0, opt=Opt(**TIGHT), tstops=[0.5])
    _to_half(integ)
    old = dae.M
    dae.M = old.copy()
    dae.M.data[0] = 2.0
    integ.model_modified()
    sol = integ.solve()
    assert old.data[0] == 1.0 and sol.stats.ret == 'success'
    dae2, y02 = _relay()
    cb = preset_time_callback([0.5], _slow_down, save_positions=(False, False))
    ref = solve(dae2, [0, 2], y02, opt=Opt(**TIGHT), callbacks=[cb])
    assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y)


def test_model_modified_before_the_first_step():
    """The changed state starts the first step; the row at ``t0`` is the
    initial state of the call."""
    dae, y0 = _relay()
    integ = init(dae, [0, 1], y0)
    integ.u[0] = 0.25
    integ.model_modified()
    assert integ.uprev[0] == 0.25 and integ.u_step is integ.u
    sol = integ.solve()
    assert sol.stats.ret == 'success' and sol.Y[0][0] == 0.0
    assert abs(sol.Y[-1][0] - (1 - 0.75 * math.exp(-1.0))) <= 1e-3


# -- termination and conditions -----------------------------------------------

@pytest.mark.parametrize('save_positions, rows', [((True, True), 2), ((False, False), 1)])
def test_an_affect_that_terminates(save_positions, rows):
    """The run ends at 0.5, where the affect terminates it; a discrete
    callback after it in the list is not applied."""
    dae, y0 = _relay()
    later = []
    stop = preset_time_callback([0.5], lambda integ: integ.terminate(), save_positions=save_positions)
    after = preset_time_callback([0.5], lambda integ: later.append(integ.t))
    sol = solve(dae, [0, 1], y0, callbacks=[stop, after])
    assert sol.stats.ret == 'terminated' and sol.stats.succeed is True
    assert sol.T[-1] == 0.5 and _rows_at(sol, 0.5).size == rows and later == []


def test_a_terminal_crossing_skips_the_discrete_callbacks():
    """``x`` reaches 0.3 at ``te``, where a terminal ``'right'`` crossing ends
    the run and a discrete condition first holds. The discrete callbacks are
    skipped at ``te``: the condition is not evaluated there, and the affect
    never runs."""
    dae, y0 = _relay()
    tested, ran = [], []

    def condition(t, y, integ):
        tested.append(t)
        return y[0] >= 0.3

    stop = ContinuousCallback(lambda t, y, integ: y[0] - 0.3, direction=1, terminal=True, record=True,
                              rootfind='right')
    sol = solve(dae, [0, 1], y0, callbacks=[stop, DiscreteCallback(condition, lambda integ: ran.append(integ.t))])
    assert sol.stats.ret == 'terminated' and ran == []
    te = sol.te[0]
    assert sol.T[-1] == te and sol.Y[-1][0] >= 0.3
    assert len(tested) == sol.stats.nstep - 1 and max(tested) < te


def _set_x(value, fired):
    def affect(integ, *idx):
        fired.append(integ.t)
        integ.u[0] = value
    return affect


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('first', ['discrete', 'continuous'])
def test_two_callbacks_at_one_time_repeat_no_row(legacy, first):
    """A first callback sets ``x`` to 0.5 where it reaches 0.3, and a discrete
    callback sets it to 0.6 at the same time. The rows there are the state
    before the first affect and the state after each, with no row repeated,
    in both configurations: the rows up to that time are saved once."""
    dae, y0 = _relay()
    fired = []
    if first == 'discrete':
        a = DiscreteCallback(lambda t, y, integ: y[0] >= 0.3 and not fired, _set_x(0.5, fired))
    else:
        a = ContinuousCallback(lambda t, y, integ: y[0] - 0.3, _set_x(0.5, fired), direction=1)
    b = DiscreteCallback(lambda t, y, integ: fired == [t], _set_x(0.6, fired))
    sol = solve(dae, [0, 1], y0, alg=Rodas4(legacy_compat=legacy), callbacks=[a, b])
    assert sol.stats.ret == 'success' and len(fired) == 2 and fired[0] == fired[1]
    k = _rows_at(sol, fired[0])
    assert k.size == 3 and np.all(np.diff(k) == 1)
    before = sol.Y[k[0]][0]
    assert abs(before - 0.3) <= 1e-12 if first == 'continuous' else before >= 0.3
    assert sol.Y[k[1]][0] == 0.5 and sol.Y[k[2]][0] == 0.6


def test_terminate_between_steps():
    dae, y0 = _relay()
    integ = init(dae, [0, 1], y0)
    assert integ.step() and integ.step()
    t = integ.t
    integ.terminate()
    assert integ.step() is False
    sol = integ.solve()
    assert sol.stats.ret == 'terminated' and sol.T[-1] == t and _rows_at(sol, t).size == 1


def test_a_discrete_callback_without_tstops():
    """The condition is evaluated at the end of every accepted step, and
    counted; its affect runs at the first step end where it holds."""
    dae, y0 = _relay()
    fired = []

    def condition(t, y, integ):
        return y[0] >= 0.3 and not fired

    def restart(integ):
        fired.append(integ.t)
        integ.u[0] = 0.0

    sol = solve(dae, [0, 1], y0, callbacks=[DiscreteCallback(condition, restart)])
    assert sol.stats.ret == 'success' and len(fired) == 1
    k = _rows_at(sol, fired[0])
    assert k.size == 2 and sol.Y[k[0]][0] >= 0.3 and sol.Y[k[1]][0] == 0.0
    assert sol.stats.ncondition == sol.stats.nstep


def test_callback_arguments():
    for bad in (dict(condition=None, affect=lambda integ: None), dict(condition=lambda t, y, integ: True,
                                                                       affect=None)):
        with pytest.raises(TypeError):
            DiscreteCallback(**bad)
    with pytest.raises(ValueError, match='save_positions'):
        DiscreteCallback(lambda t, y, integ: True, lambda integ: None, save_positions=(True,))
    cb = preset_time_callback(iter([0.25, 0.5]), lambda integ: None)
    assert cb.tstops == (0.25, 0.5)
    dae, y0 = _relay()
    with pytest.raises(ValueError, match='tstops'):
        solve(dae, [0, 1], y0, alg=Rodas4(legacy_compat=True), callbacks=[cb])


# -- the linear-solver cache and a split run ----------------------------------

def test_the_protocol_empties_the_superlu_ordering():
    dae, y0 = _relay()
    with linsolver('superlu'):
        integ = init(dae, [0, 1], y0, tstops=[0.5])
        _to_half(integ)
        assert integ.linalg.cache.superlu is not None
        integ.model_modified()
        assert integ.linalg.cache.superlu is None
        integ.step()
        assert integ.tprev == 0.5 and integ.linalg.cache.superlu is not None


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('with_matching', [False, True])
def test_the_protocol_drops_a_klu_analysis_with_a_matching(klu_matching_low, with_matching):
    """An analysis with a row matching depends on the values that triggered
    it and is dropped; one without depends only on the pattern and is kept."""
    if not with_matching:
        set_klu_matching(False)
    dae, y0 = _relay()
    with linsolver('klu'):
        integ = init(dae, [0, 1], y0, tstops=[0.5])
        _to_half(integ)
        sym = integ.linalg.cache.symbolic
        assert sym is not None and (sym.perm is not None) == with_matching
        integ.model_modified()
        assert integ.linalg.cache.symbolic is (None if with_matching else sym)


@pytest.fixture(params=['plain', 'matching'])
def matching_setting(request):
    """The KLU setting of the test: the default, or the row matching from
    two unknowns up, restored afterwards."""
    if request.param == 'matching':
        saved = (klu_backend._MATCHING, klu_backend.MATCHING_MIN_N)
        set_klu_matching(True, min_n=2)
        try:
            yield request.param
        finally:
            set_klu_matching(*saved)
    else:
        yield request.param


@pytest.mark.parametrize('method', [Rodas3, Rodas4])
def test_a_run_changed_at_half_equals_two_calls_split_there(backend, matching_setting, method):
    """``k`` goes from 0.6 to 1.7 at 0.5 in one run, and in the second of two
    calls split at 0.5, from the last row of the first. The modification
    protocol returns the run to the state of a new call, so every row after
    0.5 is byte-equal between the two. With KLU and the row matching, the
    matching found at ``t0`` differs from the one of a new call at 0.5, so
    the case fails if the protocol keeps that analysis. The SuperLU ordering
    depends on the pattern only and matters only at an exact pivot tie, so
    ``test_the_protocol_empties_the_superlu_ordering`` covers its reset."""
    opt = Opt(fix_h=True, hinit=2 ** -6)
    dae, y0 = _oscillator()
    run = solve(dae, [0, 1], y0, alg=method(), opt=opt,
                callbacks=[preset_time_callback([0.5], _set_k(1.7), save_positions=(False, True))])
    assert run.stats.ret == 'success'
    dae, y0 = _oscillator()
    first = solve(dae, [0, 0.5], y0, alg=method(), opt=opt)
    dae.p['k'][0] = 1.7
    second = solve(dae, [0.5, 1], first.Y[-1], alg=method(), opt=opt)
    k = _rows_at(run, 0.5)[-1]
    assert _byte_equal(run.T[:k], first.T) and _byte_equal(run.Y[:k], first.Y)
    assert _byte_equal(run.T[k:], second.T) and _byte_equal(run.Y[k:], second.Y)


# -- failure -------------------------------------------------------------------

def _unsolvable():
    """``x' = -x + z``, ``0 = z**2 + k`` from ``x = 0``, ``z = 2`` with ``k =
    -4``; the algebraic equation has no real solution once ``k`` is 1."""
    m = Model()
    m.x = Var('x', [0.0])
    m.z = Var('z', [2.0])
    m.k = Param('k', [-4.0])
    m.fx = Ode('fx', f=-m.x + m.z, diff_var=m.x)
    m.gz = Eqn('gz', m.z ** 2 + m.k)
    return _numerical(m)


@pytest.mark.parametrize('kind', ['discrete', 'continuous'])
@pytest.mark.parametrize('grid', [False, True])
def test_a_daeic_failure_after_a_callback(capsys, grid, kind):
    """``k`` becomes 1 at 0.5. The run fails there with the rows saved so far,
    the row before the affect included and no row after it, and never
    raises. With ``'right'`` the crossing of ``t - 0.5`` is 0.5 itself."""
    dae, y0 = _unsolvable()
    capsys.readouterr()
    tspan = np.linspace(0, 1, 4) if grid else [0, 1]
    if kind == 'discrete':
        cb = preset_time_callback([0.5], _set_k(1.0))
    else:
        cb = ContinuousCallback(lambda t, y, integ: t - 0.5, lambda integ, idx: _set_k(1.0)(integ),
                                rootfind='right')
    sol = solve(dae, tspan, y0, callbacks=[cb])
    assert sol.stats.ret == 'failed' and sol.stats.succeed is False and sol.stats.t_fail == 0.5
    assert sol.T[-1] == 0.5 and _rows_at(sol, 0.5).size == 1 and abs(sol.Y[-1][1] - 2.0) <= 1e-9
    if grid:
        assert _byte_equal(sol.T, np.append(tspan[:2], 0.5))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith('rodas4: DaeIc found no consistent initial values')
    assert lines[0].endswith('at t = 0.5; the solution is returned up to t = 0.5.')


@pytest.mark.parametrize('kind', ['discrete', 'continuous'])
def test_a_daeic_failure_after_an_affect_that_terminates(capsys, kind):
    """The affect sets ``k = 1`` and ends the run, and no row is saved at 0.5.
    ``DaeIc`` then fails, and the failure decides the result: no row of the
    state it rejected follows the rows saved so far, which end where the
    printed line says."""
    dae, y0 = _unsolvable()
    capsys.readouterr()
    tspan = np.linspace(0, 1, 4)

    def kill(integ, idx=None):
        _set_k(1.0)(integ)
        integ.terminate()

    if kind == 'discrete':
        cb = preset_time_callback([0.5], kill, save_positions=(False, False))
    else:
        cb = ContinuousCallback(lambda t, y, integ: t - 0.5, kill, terminal=True, rootfind='right',
                                save_positions=(False, False))
    sol = solve(dae, tspan, y0, callbacks=[cb])
    assert sol.stats.ret == 'failed' and sol.stats.succeed is False and sol.stats.t_fail == 0.5
    assert _byte_equal(sol.T, tspan[:2])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith('rodas4: DaeIc found no consistent initial values')
    assert lines[0].endswith(f"at t = 0.5; the solution is returned up to t = {float(sol.T[-1])!r}.")


def test_model_modified_after_a_failure_does_nothing(capsys):
    """After the run failed there is nothing to modify: the call neither runs
    ``DaeIc`` again nor prints a second line."""
    dae, y0 = _unsolvable()
    integ = init(dae, [0, 1], y0, callbacks=[preset_time_callback([0.5], _set_k(1.0))])
    while integ.step():
        pass
    assert integ.failed and integ.stats.t_fail == 0.5
    capsys.readouterr()
    epoch, nfeval, u = integ.model_epoch, integ.stats.nfeval, integ.u.copy()
    integ.model_modified()
    assert (integ.model_epoch, integ.stats.nfeval) == (epoch, nfeval) and _byte_equal(integ.u, u)
    assert capsys.readouterr().out == ''
    assert integ.postamble().stats.ret == 'failed'
