"""The saved rows, the recorded events, and the ``daesol`` built from them."""
import numpy as np

from Solverz.integrator.saving import EventLog, SolutionBuffer, to_daesol
from Solverz.solvers.stats import Stats
from Solverz.utilities.address import Address
from Solverz.variable.variables import TimeVars, Vars


def _buffer(rows):
    sol = SolutionBuffer()
    for t, y in rows:
        sol.push(t, np.array(y, dtype=np.float64))
    return sol


def test_buffer():
    sol = SolutionBuffer()
    assert sol.last_t is None and len(sol) == 0
    sol.push(np.int64(0), np.zeros(2))
    sol.push(0.5, np.ones(2))
    assert sol.last_t == 0.5 and len(sol) == 2


def test_result_without_events():
    stats = Stats('toy')
    sol = _buffer([(np.int64(0), [1.0, -0.0]), (0.25, [2.0, 3.0]), (0.5, [4.0, 5.0])])
    res = to_daesol(sol, EventLog(), stats)
    assert res.T.dtype == np.float64 and res.T.shape == (3,)
    assert res.T.tobytes() == np.array([0.0, 0.25, 0.5]).tobytes()
    assert res.Y.dtype == np.float64 and res.Y.shape == (3, 2)
    assert res.Y.flags.c_contiguous and res.Y.flags.owndata and res.T.flags.owndata
    assert res.Y.tobytes() == np.array([[1.0, -0.0], [2.0, 3.0], [4.0, 5.0]]).tobytes()
    assert res.te is None and res.ye is None and res.ie is None
    assert res.stats is stats
    sol.rows[1][0] = 9.0
    assert res.Y[1, 0] == 2.0


def test_result_with_events():
    sol = _buffer([(0.0, [1.0, 2.0]), (0.5, [3.0, 4.0])])
    events = EventLog()
    ye = np.array([3.0, 4.0])
    events.record(0.5, ye, 0)
    events.record(0.5, ye, 1)
    assert len(events) == 2
    res = to_daesol(sol, events, Stats())
    assert res.te.dtype == np.float64 and res.te.tolist() == [0.5, 0.5]
    assert res.ie.dtype == np.int64 and res.ie.tolist() == [0, 1]
    assert res.ye.dtype == np.float64 and res.ye.shape == (2, 2)
    assert res.ye.tobytes() == np.array([ye, ye]).tobytes()
    assert res.Y[-1].tobytes() == res.ye[0].tobytes()


def _address():
    a = Address()
    a.add('x', 2)
    a.add('z', 1)
    return a


def test_vars_conversion():
    a = _address()
    y0 = Vars(a, np.array([1.0, 2.0, 3.0]))
    sol = _buffer([(0.0, [1.0, 2.0, 3.0]), (1.0, [4.0, 5.0, 6.0])])
    res = to_daesol(sol, EventLog(), Stats(), y0.a)
    assert isinstance(res.Y, TimeVars)
    assert res.Y.array.tobytes() == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).tobytes()
    assert res.Y['z'].tolist() == [[3.0], [6.0]]
    assert res.ye is None

    events = EventLog()
    events.record(1.0, np.array([4.0, 5.0, 6.0]), 2)
    res = to_daesol(sol, events, Stats(), y0.a)
    assert isinstance(res.ye, TimeVars)
    assert res.ye['x'].tolist() == [[4.0, 5.0]]
    assert res.ie.tolist() == [2]
