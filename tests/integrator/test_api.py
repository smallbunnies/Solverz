"""The public API of the integrator core.

``solve(...)``, ``init(...).solve()``, a loop of ``step()`` and the legacy-shaped
``alg(dae, tspan, y0, opt)`` run the same code and give byte-equal results. A
``Vars`` initial state gives ``TimeVars`` rows and event states. The package
exports exactly the names of its ``__all__``, and the top level of Solverz
re-exports the four Rosenbrock methods and the two callback classes, but not
``solve``, ``init`` or ``Integrator``.
"""
import numpy as np
import pytest

import Solverz
import Solverz.integrator
from Solverz.integrator import (ContinuousCallback, ImplicitEuler, Rodas3, Rodas4, Trapezoid, init,
                                solve)
from Solverz.solvers.option import Opt
from Solverz.variable.variables import TimeVars

from tests.integrator import models

pytestmark = pytest.mark.i8

ALL = ['solve', 'init', 'Integrator', 'IntegratorOptions',
       'Algorithm', 'StepFailure',
       'Rosenbrock', 'RosenbrockTableau', 'Rodas3', 'Rodas4', 'Rodasp', 'Rodas5P',
       'ImplicitEuler', 'Trapezoid',
       'ContinuousCallback', 'DiscreteCallback', 'preset_time_callback',
       'Controller', 'IController', 'PIController', 'LegacyRodasController']

TOP_LEVEL = ['Rodas3', 'Rodas4', 'Rodasp', 'Rodas5P', 'ContinuousCallback', 'DiscreteCallback']


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _counters(stats):
    return (stats.scheme, stats.ret, stats.succeed, stats.nstep, stats.nreject, stats.nfeval,
            stats.nJeval, stats.ndecomp, stats.nsolve, stats.ncondition)


def _stepped(dae, tspan, y0, alg, opt, callbacks=()):
    """The result of ``init`` advanced by a loop of ``step()``; ``solve()`` after
    the loop only returns the result, since the run is over."""
    integ = init(dae, tspan, y0, alg=alg, opt=opt, callbacks=callbacks)
    nsteps = 0
    while integ.step():
        nsteps += 1
    assert integ.finished and integ.step() is False
    sol = integ.solve()
    assert nsteps == sol.stats.nstep - 1
    return sol


def _four(dae, tspan, y0, make_alg, make_opt, callbacks=()):
    """The four entries, each with its own algorithm and ``Opt``."""
    return [
        solve(dae, tspan, y0, alg=make_alg(), opt=make_opt(), callbacks=callbacks),
        init(dae, tspan, y0, alg=make_alg(), opt=make_opt(), callbacks=callbacks).solve(),
        _stepped(dae, tspan, y0, make_alg(), make_opt(), callbacks),
        make_alg()(dae, tspan, y0, make_opt()) if not callbacks else None,
    ]


def _assert_same(results):
    ref = results[0]
    for sol in results[1:]:
        if sol is None:
            continue
        assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y)
        for a, b in ((sol.te, ref.te), (sol.ye, ref.ye), (sol.ie, ref.ie)):
            assert (a is None) == (b is None)
            if a is not None:
                assert _byte_equal(a, b)
        assert _counters(sol.stats) == _counters(ref.stats)


ALGS = {
    'default': lambda: None,
    'rodas4': Rodas4,
    'rodas3': Rodas3,
    'rodas4_compat': lambda: Rodas4(legacy_compat=True),
    'implicit_euler': ImplicitEuler,
    'trapezoid': Trapezoid,
}


@pytest.mark.parametrize('tspan', [[0, 20], np.linspace(0, 20, 201)], ids=['span', 'grid'])
@pytest.mark.parametrize('alg', list(ALGS))
def test_the_four_entries_are_byte_equal(model, alg, tspan):
    dae, y0 = model('dae_test')
    make_alg = ALGS[alg]
    if alg == 'default':
        results = [solve(dae, tspan, y0, opt=Opt(hinit=0.1)),
                   init(dae, tspan, y0, opt=Opt(hinit=0.1)).solve(),
                   _stepped(dae, tspan, y0, None, Opt(hinit=0.1)),
                   Rodas4()(dae, tspan, y0, Opt(hinit=0.1))]
    else:
        results = _four(dae, tspan, y0, make_alg, lambda: Opt(hinit=0.1))
    assert results[0].stats.ret == 'success'
    _assert_same(results)


def _ball_event(t, y):
    return np.array([y[0]]), np.array([1]), np.array([-1])


@pytest.mark.parametrize('compat', [False, True], ids=['default', 'compat'])
def test_the_four_entries_agree_with_a_terminal_event(model, compat):
    dae, y0 = model('ball')
    results = _four(dae, np.linspace(0, 30, 61), y0, lambda: Rodas4(legacy_compat=compat),
                    lambda: Opt(event=_ball_event, rtol=1e-6, atol=1e-8))
    assert results[0].stats.ret == 'terminated' and results[0].te is not None
    _assert_same(results)


def test_the_entries_agree_with_a_callback(model):
    dae, y0 = model('ball')

    def bounce(integ, idx):
        integ.u[1] = -0.9 * integ.u[1]

    cb = ContinuousCallback(lambda t, y, integ: y[0], bounce, direction=-1)
    results = _four(dae, [0, 10], y0, Rodas4, lambda: Opt(rtol=1e-6, atol=1e-8), callbacks=[cb])
    assert results[0].stats.ret == 'success'
    # the ball stays above the ground only if the affect ran
    assert results[0].Y[:, 0].min() >= 0
    _assert_same(results)


def test_vars_in_gives_timevars_rows_and_event_states():
    sdae, y0 = models.ball_model()
    dae = Solverz.made_numerical(sdae, y0, sparse=True)
    y = np.array(y0.array, dtype=np.float64)
    tspan = np.linspace(0, 30, 61)
    make_opt = lambda: Opt(event=_ball_event, rtol=1e-6, atol=1e-8)
    arrays = _four(dae, tspan, y, Rodas4, make_opt)
    with_vars = _four(dae, tspan, y0, Rodas4, make_opt)
    for sol, ref in zip(with_vars, arrays):
        assert isinstance(sol.Y, TimeVars) and isinstance(sol.ye, TimeVars)
        assert isinstance(ref.Y, np.ndarray) and isinstance(ref.ye, np.ndarray)
        assert _byte_equal(sol.Y.array, ref.Y) and _byte_equal(sol.ye.array, ref.ye)
        assert _byte_equal(sol.Y['x'], ref.Y)
        assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.te, ref.te) and _byte_equal(sol.ie, ref.ie)
    # the caller's Vars is not written
    assert _byte_equal(y0.array, y)


def test_the_package_exports_exactly_its_all():
    assert Solverz.integrator.__all__ == ALL
    for name in ALL:
        assert getattr(Solverz.integrator, name) is not None


def test_the_top_level_reexports_the_methods_and_the_callbacks():
    from Solverz import Rodas4 as R4, ContinuousCallback as CC
    assert R4 is Rodas4 and CC is ContinuousCallback
    for name in TOP_LEVEL:
        assert getattr(Solverz, name) is getattr(Solverz.integrator, name)
    ns = {}
    exec('from Solverz import *', ns)
    for name in TOP_LEVEL:
        assert ns[name] is getattr(Solverz.integrator, name)
    for name in ('solve', 'init', 'Integrator'):
        assert name not in ns
        assert not hasattr(Solverz, name)
    # the legacy solver keeps its top-level name
    from Solverz.solvers.daesolver.rodas.rodas import Rodas
    assert ns['Rodas'] is Rodas
