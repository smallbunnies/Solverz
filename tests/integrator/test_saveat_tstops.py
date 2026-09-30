"""Saving and stop times in the default configuration.

The nodes of an output grid are saved byte-exactly from the interpolant and
never change the steps, so a run on a grid accepts the same times as a run
on ``[t0, tend]`` and saves at each node the value that ``interp`` gives
right after the step that holds it; a node at the end of a step is the
state itself, not the interpolant there. A step ends exactly on every stop time
in ``(t0, tend)`` and on ``tend``, so ``T[-1] == tend`` also for spans whose
sum of steps misses ``tend``; stop times outside that interval are ignored.
A step shortened only to meet a stop time does not shrink the next one, a
step that would end within 100 units in the last place before a stop time is
stretched to it, and ``interp`` accepts every time up to the stop time after
such a step, even where ``tprev + dt_step`` falls short of it. The Hermite
interpolant of Rodas3 pairs the rows of ``M`` with the variables, so on
``permuted`` its nodes are accurate and its algebraic variable is linear
within a step.
"""
import math

import numpy as np
import pytest

from Solverz.integrator import Rodas3, Rodas4, Rodas5P, Rosenbrock, init
from Solverz.solvers.option import Opt

from tests.integrator.test_legacy_transcription import SCHEMES

pytestmark = pytest.mark.i5


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _run(integ, nodes=()):
    """Advance ``integ`` step by step to the end.

    Returns the ``daesol``, the time of every accepted step, and ``interp``
    at each node right after the step whose interval holds it.
    """
    times, at_nodes = [], []
    nodes = list(nodes)
    while True:
        nstep = integ.stats.nstep
        more = integ.step()
        if integ.stats.nstep > nstep:
            times.append(integ.t)
            while nodes and nodes[0] <= integ.t:
                at_nodes.append(integ.interp(nodes.pop(0)))
        if not more:
            return integ.postamble(), times, at_nodes


# name, variant, tspan, Opt arguments
GRID_CASES = [
    ('dae_test', 'inline_sparse', (0.0, 20.0), dict(hinit=0.1)),
    ('dae_test', 'rendered', (0.0, 20.0), dict(rtol=1e-6, atol=1e-8)),
    ('forced', 'inline_dense', (0.0, 1.0), dict(rtol=1e-6, atol=1e-8)),
    ('trace', 'inline_sparse', (0.0, 1.0), dict(rtol=1e-6, atol=1e-8)),
    ('vdp', 'inline_sparse', (0.0, 20.0), dict(rtol=1e-6, atol=1e-9)),
]


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('name, variant, span, kwargs', GRID_CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in GRID_CASES])
def test_a_grid_saves_the_nodes_and_keeps_the_steps(model, name, variant, span, kwargs, scheme):
    dae, y0 = model(name, variant)
    grid = np.linspace(span[0], span[1], 201)
    alg = Rosenbrock.from_scheme(scheme)
    plain, times, at_nodes = _run(init(dae, list(span), y0, alg=alg, opt=Opt(**kwargs)), grid[1:])
    dense, dense_times, _ = _run(init(dae, grid, y0, alg=alg, opt=Opt(**kwargs)))
    assert plain.stats.ret == dense.stats.ret == 'success'
    assert _byte_equal(np.array(dense_times), np.array(times))
    assert _byte_equal(plain.T, np.array([span[0]] + times))
    assert plain.T[-1] == span[1]
    assert (dense.stats.nstep, dense.stats.nreject) == (plain.stats.nstep, plain.stats.nreject)
    assert _byte_equal(dense.T, grid)
    assert _byte_equal(dense.Y[0], plain.Y[0])
    assert len(at_nodes) == grid.size - 1
    for k, y in enumerate(at_nodes):
        assert _byte_equal(dense.Y[k + 1], y), k
    # tend is the last stop time, so its row is the state itself
    assert _byte_equal(dense.Y[-1], plain.Y[-1])


def test_a_node_at_a_step_end_is_the_state_itself(model):
    """A node that is the end of a step, a stop time or ``tend``, is saved as
    the state and not from the interpolant. The dense output at ``theta = 1``
    is associated differently from the step itself and differs from the end
    state in the last bits in some of these runs; the count of such nodes
    shows that the check is not vacuous."""
    differs = 0
    for name, tend, kwargs in (('ball', 2.0, {}), ('orbit', 5.0, dict(rtol=1e-6, atol=1e-8))):
        dae, y0 = model(name)
        tstops = [tend / 3, tend / 2]
        grid = np.unique(np.concatenate([np.linspace(0, tend, 7), tstops]))
        for stops in ([], tstops):
            for scheme in SCHEMES:
                alg = Rosenbrock.from_scheme(scheme)
                integ = init(dae, grid, y0, alg=alg, opt=Opt(**kwargs), tstops=stops)
                ends = {}
                while True:
                    more = integ.step()
                    if integ.t in grid:
                        interpolated = np.empty(integ.n)
                        alg.addsteps(integ, integ.cache)
                        alg.interpolant(integ, integ.cache, (integ.t - integ.tprev) / integ.dt_step, interpolated)
                        ends[integ.t] = (integ.u.copy(), interpolated)
                    if not more:
                        break
                sol = integ.postamble()
                assert sorted(ends) == stops + [tend]
                for k, tq in enumerate(grid):
                    if tq in ends:
                        u, interpolated = ends[tq]
                        assert _byte_equal(sol.Y[k], u), (name, scheme, tq)
                        differs += not _byte_equal(interpolated, u)
    assert differs > 0


@pytest.mark.parametrize('span', [(0.0, 0.3), (0.1, 0.7), (0.3, 1.9)], ids=['0-0.3', '0.1-0.7', '0.3-1.9'])
@pytest.mark.parametrize('fixed', [False, True], ids=['adaptive', 'fixed'])
def test_the_last_step_ends_on_tend(model, span, fixed):
    """``0.1 + 0.2`` exceeds ``0.3``, and a fixed step of ``0.1`` does not sum
    to ``0.3``; both runs end on ``tend`` exactly."""
    dae, y0 = model('dae_test')
    opt = Opt(fix_h=True, hinit=0.1) if fixed else Opt(rtol=1e-6, atol=1e-8)
    for grid in (False, True):
        tspan = np.linspace(span[0], span[1], 7) if grid else list(span)
        sol = Rodas4()(dae, tspan, y0, opt)
        assert sol.stats.ret == 'success'
        assert sol.T[-1] == span[1] and np.all(np.diff(sol.T) > 0)
        if fixed and not grid:
            # the sum of the steps misses tend by a few roundings, and the last
            # step is stretched to it
            assert sol.T.size == round((span[1] - span[0]) / 0.1) + 1
            assert np.all(np.abs(np.diff(sol.T) - 0.1) <= 1e-12)


@pytest.mark.parametrize('scheme', SCHEMES)
def test_every_stop_time_is_a_step_end(model, scheme):
    dae, y0 = model('dae_test')
    tstops = [19.9, 1 / 3, 0.1, 7.77, 2.5, 1 / 3]
    alg = Rosenbrock.from_scheme(scheme)
    opt = Opt(rtol=1e-6, atol=1e-8)
    sol = alg(dae, [0, 20], y0, opt)
    stopped, times, _ = _run(init(dae, [0, 20], y0, alg=alg, opt=opt, tstops=tstops))
    assert stopped.stats.ret == 'success'
    for ts in tstops:
        assert ts in times and ts in stopped.T.tolist()
    assert np.all(np.diff(stopped.T) > 0) and stopped.T[-1] == 20.0
    assert not any(ts in sol.T.tolist() for ts in tstops)
    # a numpy array and a grid give the same steps, and the grid only the nodes
    grid = np.linspace(0, 20, 201)
    on_grid, grid_times, _ = _run(init(dae, grid, y0, alg=alg, opt=opt, tstops=np.array(tstops)))
    assert _byte_equal(np.array(grid_times), np.array(times))
    assert _byte_equal(on_grid.T, grid)
    assert np.max(np.abs(on_grid.Y[-1] - sol.Y[-1])) < 1e-5


def test_stop_times_outside_the_span_are_ignored(model):
    dae, y0 = model('dae_test')
    alg, opt = Rodas4(), Opt(rtol=1e-6, atol=1e-8)
    ref = alg(dae, [0.5, 20], y0, opt)
    for tstops in ([-1.0, 0.0, 0.5, 20.0, 25.0], np.array([20, 21, 0.5]), [0.5, 0.5]):
        integ = init(dae, [0.5, 20], y0, alg=alg, opt=opt, tstops=tstops)
        assert integ.tstops == [20.0]
        sol = integ.solve()
        assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y)


def test_a_truncated_step_does_not_shrink_the_next(model):
    """A stop time one thousandth into a step truncates it far below the
    length the controller can grow a step to in one acceptance, and the step
    after it is proposed at the length the truncated one replaced."""
    dae, y0 = model('dae_test')
    opt = Opt(rtol=1e-6, atol=1e-8)
    _, times, _ = _run(init(dae, [0, 20], y0, alg=Rodas4(), opt=opt))
    tstop = times[4] + 1e-3 * (times[5] - times[4])
    integ = init(dae, [0, 20], y0, alg=Rodas4(), opt=opt, tstops=[tstop])
    while integ.step() and integ.t < tstop:
        pass
    assert integ.t == tstop and integ.tprev == times[4]
    assert integ.dt_untruncated > 100 * integ.dt_step
    assert integ.dtpropose == integ.dt_untruncated
    sol = integ.solve()
    assert sol.stats.ret == 'success' and sol.T[-1] == 20.0 and tstop in sol.T.tolist()


def _stretch(tstop, t0, k):
    """The step from ``t0`` that ends ``k`` units in the last place of the
    distance before ``tstop``."""
    d = tstop - t0
    return d - k * math.ulp(d)


@pytest.mark.parametrize('t0, tstop', [(0.1, 0.7), (0.0, 0.3), (0.3, 1.9), (2.0 ** -53, 1.0 + 2.0 ** -52)],
                         ids=['0.1-0.7', '0-0.3', '0.3-1.9', 'tie'])
def test_a_step_just_before_a_stop_time_is_stretched_to_it(model, t0, tstop):
    """Within ``100 ulp(max(|t|, |tstop|))`` of the stop time the step ends on
    it; beyond that the next step does. The ball's solution is a quadratic,
    which every scheme integrates exactly, so every step is accepted."""
    dae, y0 = model('ball')
    tol = 100 * math.ulp(max(abs(t0), abs(tstop)))
    d = tstop - t0
    near = max(k for k in range(1, 400) if (d - k * math.ulp(d)) + tol >= d)
    for fixed in (True, False):
        for k, stretched in ((1, True), (near, True), (near + 1, False)):
            dt = _stretch(tstop, t0, k)
            if fixed:
                integ = init(dae, [t0, 2.0], y0, alg=Rodas4(), opt=Opt(fix_h=True, hinit=dt), tstops=[tstop])
            else:
                integ = init(dae, [t0, 2.0], y0, alg=Rodas4(), tstops=[tstop])
                integ.dt = dt
            assert integ.step()
            if stretched:
                assert integ.t == tstop and integ.dt_step == d
            else:
                assert integ.t == t0 + dt < tstop
                assert integ.step() and integ.t == tstop


def test_interp_reaches_the_stop_time_after_a_stretched_step(model):
    """From ``t0 = 2**-53`` to the stop time ``1 + 2**-52`` both roundings are
    ties to even: the distance rounds to ``1`` and ``t0 + 1`` rounds to ``1``,
    so the computed step ends one unit in the last place before the stop time
    it is assigned."""
    dae, y0 = model('ball')
    t0, tstop = 2.0 ** -53, 1.0 + 2.0 ** -52
    for alg in (Rodas4(), Rodas3()):
        integ = init(dae, [t0, 2.0], y0, alg=alg, tstops=[tstop])
        integ.dt = _stretch(tstop, t0, 50)
        assert integ.step() and integ.t == tstop
        end = integ.tprev + integ.dt_step
        assert end < integ.t
        tq = np.nextafter(end, np.inf)
        assert tq == integ.t
        assert _byte_equal(integ.interp(tq), integ.u)
        # the computed end itself lies inside the step; the ball is interpolated exactly
        y = integ.interp(end)
        assert np.all(np.abs(y - integ.u) <= 1e-12 * (1 + np.abs(integ.u)))
        with pytest.raises(ValueError, match='outside the last step'):
            integ.interp(np.nextafter(integ.t, np.inf))


def test_rodas3_pairs_rows_and_variables_on_permuted(model):
    """``permuted`` declares its algebraic equation first, so row 0 of ``F``
    is not the derivative of variable 0; the default Hermite interpolant
    pairs them through ``M`` and interpolates the algebraic variable
    linearly.

    The linear interpolant's error is of order ``h**2``, and the algebraic
    variable moves fastest in the first steps, so the tolerance is chosen at
    which the interpolated nodes, not only the step ends, meet the bound. The
    legacy-compatible interpolant, which takes the residual rows as the
    slopes of the variables, misses it at the same tolerance.
    """
    dae, y0 = model('permuted')
    grid = np.linspace(0, 20, 201)
    ref = Rodas5P()(dae, grid, y0, Opt(rtol=1e-11, atol=1e-13))
    opt = Opt(rtol=1e-8, atol=1e-10)
    integ = init(dae, grid, y0, alg=Rodas3(), opt=opt)
    rows, cols = dae.M.nonzero()
    algebraic = sorted(set(range(y0.size)) - set(cols.tolist()))
    assert algebraic == [1] and rows.tolist() == [1] and cols.tolist() == [0]
    checked = 0
    while True:
        nstep = integ.stats.nstep
        idx = integ.saveat_idx
        more = integ.step()
        if integ.stats.nstep > nstep:
            for tq in grid[1 + idx:1 + integ.saveat_idx]:
                if tq == integ.t:
                    continue
                y = integ.interp(tq)
                theta = (tq - integ.tprev) / integ.dt_step
                lin = integ.uprev + theta * (integ.u_step - integ.uprev)
                assert abs(y[1] - lin[1]) <= 1e-12 * (1 + abs(lin[1])), tq
                checked += 1
        if not more:
            break
    sol = integ.postamble()
    assert sol.stats.ret == ref.stats.ret == 'success' and checked > 0
    dev = np.max(np.abs(sol.Y - ref.Y))
    unpaired = Rodas3(legacy_compat=True)(dae, grid, y0, opt)
    dev_unpaired = np.max(np.abs(unpaired.Y - ref.Y))
    print(f"Rodas3 on permuted against Rodas5P at rtol=1e-11: max|dY| = {dev:.3e} over {checked} "
          f"interpolated nodes, {dev_unpaired:.3e} without the pairing")
    assert dev <= 1e-5 < dev_unpaired
