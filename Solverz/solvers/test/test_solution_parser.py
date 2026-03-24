import numpy as np

from Solverz.num_api.num_eqn import nDAE
from Solverz.solvers.option import Opt
from Solverz.solvers.parser import dae_io_parser
from Solverz.solvers.solution import daesol
from Solverz.solvers.stats import Stats
from Solverz.utilities.address import Address
from Solverz.variable.variables import TimeVars, Vars


def make_timevars(values):
    addr = Address()
    addr.add('x', 1)
    y0 = Vars(addr, np.array([values[0]]))
    out = TimeVars(y0, len(values))
    out.array[:, 0] = np.array(values, dtype=float)
    return y0, out


def test_dae_io_parser_profiles_and_wraps_time_series(capsys):
    @dae_io_parser
    def fake_solver(eqn, tspan, y0, opt):
        stats = Stats('fake')
        return daesol(T=np.array(tspan, dtype=float),
                      Y=np.array([[y0[0]], [y0[0] + 1.0]]),
                      te=np.array([0.5]),
                      ye=np.array([[y0[0] + 0.5]]),
                      ie=np.array([0]),
                      stats=stats)

    y0, _ = make_timevars([1.0])
    eqn = nDAE(np.eye(1),
               lambda t, y, p: y,
               lambda t, y, p: np.eye(1),
               {})

    sol = fake_solver(eqn, [0.0, 1.0], y0, Opt(profile=True))

    assert "Time elapsed:" in capsys.readouterr().out
    assert isinstance(sol.Y, TimeVars)
    assert isinstance(sol.ye, TimeVars)
    np.testing.assert_allclose(sol.Y.array[:, 0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(sol.ye.array[:, 0], np.array([1.5]))


def test_daesol_slice_keeps_events_and_order_variation():
    _, Y = make_timevars([0.0, 1.0, 2.0, 3.0])
    _, YE = make_timevars([0.5, 2.5])

    stats = Stats('demo')
    stats.order_variation = np.array([1, 2, 3, 4])
    sol = daesol(np.array([0.0, 1.0, 2.0, 3.0]),
                 Y,
                 np.array([0.5, 2.5]),
                 YE,
                 np.array([0, 1]),
                 stats)

    sub = sol[0:2]

    np.testing.assert_allclose(sub.T, np.array([0.0, 1.0]))
    np.testing.assert_allclose(sub.te, np.array([0.5]))
    np.testing.assert_allclose(sub.ye.array[:, 0], np.array([0.5]))
    np.testing.assert_array_equal(sub.ie, np.array([0]))
    np.testing.assert_array_equal(sub.stats.order_variation, np.array([1, 2]))
    np.testing.assert_array_equal(sol.stats.order_variation, np.array([1, 2, 3, 4]))


def test_daesol_slice_without_events_returns_empty_event_series():
    _, Y = make_timevars([0.0, 1.0, 2.0, 3.0])
    _, YE = make_timevars([0.5, 2.5])

    sol = daesol(np.array([0.0, 1.0, 2.0, 3.0]),
                 Y,
                 np.array([0.5, 2.5]),
                 YE,
                 np.array([0, 1]),
                 Stats('demo'))

    sub = sol[2:3]

    assert sub.te.size == 0
    assert sub.ie.size == 0
    assert sub.ye.len == 0

    chained = sub[0:1]

    np.testing.assert_allclose(chained.T, np.array([2.0]))
    np.testing.assert_allclose(chained.Y.array[:, 0], np.array([2.0]))
    assert chained.te.size == 0
    assert chained.ie.size == 0
    assert chained.ye.len == 0


def test_daesol_append_accumulates_stats_and_events():
    _, Y1 = make_timevars([0.0, 1.0])
    _, Y2 = make_timevars([1.0, 2.0])
    _, YE1 = make_timevars([0.5])
    _, YE2 = make_timevars([1.5])

    stats1 = Stats('demo')
    stats1.nstep = 1
    stats1.nfeval = 2
    stats1.ndecomp = 3
    stats1.nJeval = 4
    stats1.nsolve = 5
    stats1.nreject = 6
    stats1.order_variation = np.array([1, 2])

    stats2 = Stats('demo')
    stats2.nstep = 10
    stats2.nfeval = 20
    stats2.ndecomp = 30
    stats2.nJeval = 40
    stats2.nsolve = 50
    stats2.nreject = 60
    stats2.order_variation = np.array([3, 4])

    sol1 = daesol(np.array([0.0, 1.0]), Y1, np.array([0.5]), YE1, np.array([0]), stats1)
    sol2 = daesol(np.array([1.0, 2.0]), Y2, np.array([1.5]), YE2, np.array([1]), stats2)

    sol1.append(sol2)

    np.testing.assert_allclose(sol1.T, np.array([0.0, 1.0, 1.0, 2.0]))
    np.testing.assert_allclose(sol1.Y.array[:, 0], np.array([0.0, 1.0, 1.0, 2.0]))
    np.testing.assert_allclose(sol1.te, np.array([0.5, 1.5]))
    np.testing.assert_allclose(sol1.ye.array[:, 0], np.array([0.5, 1.5]))
    np.testing.assert_array_equal(sol1.ie, np.array([0, 1]))

    assert sol1.stats.nstep == 11
    assert sol1.stats.nfeval == 22
    assert sol1.stats.ndecomp == 33
    assert sol1.stats.nJeval == 44
    assert sol1.stats.nsolve == 55
    assert sol1.stats.nreject == 66
    np.testing.assert_array_equal(sol1.stats.order_variation, np.array([1, 2, 3, 4]))
