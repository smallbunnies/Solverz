"""The conformance kit: ``check_algorithm`` runs the checks every algorithm of
the core must pass.

Import it as ``from Solverz.integrator.testing import check_algorithm``. The
package does not import this module, since the checks build their models
with the symbolic layer.
"""
import contextlib
import importlib
import io
import math
import numbers
import sys
import tempfile
import uuid
from types import SimpleNamespace

import numpy as np

from Solverz.num_api.num_eqn import nDAE
from Solverz.integrator.algorithm import check_style
from Solverz.integrator.callbacks import preset_time_callback
from Solverz.integrator.integrator import init, solve

__all__ = ['check_algorithm']

# the step of every run of an algorithm without an error estimate, and of the history check
H_FIX = 2.0 ** -6
# the fixed steps 2**-k of the order runs, and the error below which a slope is rounding
ORDER_STEPS = range(1, 7)
ERROR_FLOOR = 1e-12

# rendered against inline: the one criterion wherever the two forms of a model are compared
MAX_STEP_DIFFERENCE = 2
MAX_DY_IN_RTOL = 10

CHECKS = ('contract', 'order', 'interpolant', 'saveat', 'tstops', 'events', 'events on a grid',
          'inconsistent start', 'history', 'failure', 'out', 'Opt', 'y0', 'counts', 'rendered')


def rendered_matches_inline(rendered, inline, rtol):
    """``(dsteps, dY, ok)`` of a rendered and an inline run of one model on one grid.

    Byte equality is not required, since Numba and NumPy evaluate ``**`` and
    transcendental functions with different routines: the numbers of
    accepted steps may differ by 2, and the saved rows by ``10 * rtol``.
    """
    if not np.array_equal(rendered.T, inline.T):
        raise ValueError('the two runs are not on one grid')
    dsteps = abs(rendered.stats.nstep - inline.stats.nstep)
    dY = float(np.max(np.abs(np.asarray(rendered.Y) - np.asarray(inline.Y))))
    return dsteps, dY, dsteps <= MAX_STEP_DIFFERENCE and dY <= MAX_DY_IN_RTOL * rtol


def check_algorithm(alg, *, order_tol=0.3, rendered=False):
    """Run the conformance checks on the algorithm instance ``alg``.

    The checks run in order and the first that fails raises
    ``AssertionError('check_algorithm[<check>]: ...')``; an exception raised
    inside a check fails that check. The result maps each check to the
    values it measured. Which models and step sizes a check uses depends only
    on the declared traits: an explicit algorithm runs on an ODE, and an
    algorithm without an error estimate with the fixed step ``2**-6``. The
    measured orders must reach the declared ``order`` and ``interp_order``
    within ``order_tol``. ``rendered=True`` adds the comparison with a model
    rendered by ``module_printer``, which compiles it with Numba.
    """
    names = CHECKS if rendered else CHECKS[:-1]
    return _check(alg, names, order_tol)


def _check(alg, names, order_tol=0.3):
    """The checks ``names``, in the order of ``CHECKS``, on ``alg``."""
    # the symbolic layer is imported here, so that importing the package does not import it
    from Solverz import Eqn, Model, Ode, Param, Var, made_numerical, module_printer
    from Solverz.solvers.option import Opt

    if isinstance(alg, type):
        raise TypeError(f"check_algorithm takes an instance of the algorithm, such as {alg.__name__}()")
    if alg.legacy_compat:
        raise ValueError("check_algorithm checks the default configuration; the legacy-compatible "
                         "configuration refuses tstops and reproduces legacy Rodas instead")
    sz = SimpleNamespace(Eqn=Eqn, Model=Model, Ode=Ode, Param=Param, Var=Var,
                         made_numerical=made_numerical, module_printer=module_printer, Opt=Opt)
    kit = _Kit(alg, sz, order_tol)
    results = {}
    for name in CHECKS:
        if name not in names:
            continue
        if name == 'inconsistent start' and kit.explicit:
            continue
        try:
            results[name] = _CHECK[name](kit)
        except _Failed as e:
            raise AssertionError(f"check_algorithm[{name}]: {kit.name}: {e}") from None
        except Exception as e:
            raise AssertionError(f"check_algorithm[{name}]: {kit.name}: {type(e).__name__}: {e}") from e
    return results


class _Failed(Exception):
    """A criterion of a check does not hold."""


def _require(condition, message):
    if not condition:
        raise _Failed(message)


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _x_exact(t):
    return 0.5 * (np.exp(-t) + np.sin(t) - np.cos(t))


def _steps(integ):
    """Advance ``integ`` by one accepted step per iteration until the run is over."""
    while not integ.finished:
        integ.step()
        if integ.failed:
            return
        yield integ


def _slopes(errors):
    """``log2(e_k / e_{k+1})`` of errors at halved steps."""
    e = np.asarray(errors, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log2(e[:-1] / e[1:])


def _succeeded(sol, what):
    _require(sol.stats.ret == 'success', f"{what} ended with ret = {sol.stats.ret!r}")


# -- the models ---------------------------------------------------------------------


def _symbolic(sz, name):
    """The symbolic model ``name`` and its ``Vars``.

    A: ``x' = -x + z``, ``0 = z - k s``, ``s' = c``, ``c' = -s`` from ``x = z =
    s = 0``, ``c = 1``, with ``k = 1``; the forcing is carried by an
    oscillator, so that ``dF/dt`` is zero and cannot limit a measured order,
    and ``x(t) = (exp(-t) + sin t - cos t)/2``. P: A with its equations
    declared as ``c'``, ``z``, ``x'``, ``s'``, so that the rows of ``M`` are not
    aligned with the variables. E: A without ``z``, declared as ``c'``, ``x'``,
    ``s'``, so that ``M`` is a permutation. A_delta: A from ``z = 1e-7``,
    whose algebraic residual lies below the threshold of ``DaeIc`` and stays.
    B: the bouncing ball from ``[0, 20]``. C: ``x' = x**3`` from 1, which
    blows up at ``t = 0.5`` with no real continuation.
    """
    Var, Param, Ode, Eqn = sz.Var, sz.Param, sz.Ode, sz.Eqn
    m = sz.Model()
    if name in ('A', 'P', 'A_delta', 'E'):
        m.x = Var('x', 0.0)
        if name != 'E':
            m.z = Var('z', 1e-7 if name == 'A_delta' else 0.0)
        m.s = Var('s', 0.0)
        m.c = Var('c', 1.0)
        m.k = Param('k', 1.0)
        if name == 'E':
            m.fc = Ode('fc', -m.s, m.c)
            m.fx = Ode('fx', -m.x + m.k * m.s, m.x)
            m.fs = Ode('fs', m.c, m.s)
        elif name == 'P':
            m.fc = Ode('fc', -m.s, m.c)
            m.gz = Eqn('gz', m.z - m.k * m.s)
            m.fx = Ode('fx', -m.x + m.z, m.x)
            m.fs = Ode('fs', m.c, m.s)
        else:
            m.fx = Ode('fx', -m.x + m.z, m.x)
            m.gz = Eqn('gz', m.z - m.k * m.s)
            m.fs = Ode('fs', m.c, m.s)
            m.fc = Ode('fc', -m.s, m.c)
    elif name == 'B':
        m.x = Var('x', [0.0, 20.0])
        m.f1 = Ode('f1', m.x[1], m.x[0])
        m.f2 = Ode('f2', -9.8, m.x[1])
    elif name == 'C':
        m.x = Var('x', 1.0)
        m.f = Ode('f', m.x ** 3, m.x)
    else:
        raise ValueError(f"unknown model {name!r}")
    return m.create_instance()


class _Kit:
    """The algorithm under test, its traits, and the models of one ``check_algorithm`` call."""

    def __init__(self, alg, sz, order_tol):
        self.alg = alg
        self.sz = sz
        self.order_tol = order_tol
        self.name = alg.scheme if isinstance(alg.scheme, str) else type(alg).__name__
        self.explicit = bool(alg.explicit)
        self.adaptive = bool(alg.adaptive)
        # the models the checks that name A or P use
        self.main = ('E',) if self.explicit else ('A', 'P')
        self.base = self.main[0]
        self._models = {}
        self._memo = {}

    def build(self, name):
        """``(dae, y0, Vars)`` of a new instance of the model ``name``."""
        with contextlib.redirect_stdout(io.StringIO()):
            sdae, y0 = _symbolic(self.sz, name)
            dae = self.sz.made_numerical(sdae, y0, sparse=True)
        return dae, np.array(y0.array, dtype=np.float64), y0

    def model(self, name):
        """``(dae, y0)`` of the model ``name``, built once per call of the kit;
        a check that changes the model builds its own."""
        dae, y0, _ = self._built(name)
        return dae, y0.copy()

    def vars(self, name):
        """The ``Vars`` of the model ``name``."""
        return self._built(name)[2]

    def _built(self, name):
        if name not in self._models:
            self._models[name] = self.build(name)
        return self._models[name]

    def opt(self, **kwargs):
        """An ``Opt``; without an error estimate the run takes the fixed step ``H_FIX``."""
        if not self.adaptive:
            kwargs.setdefault('hinit', H_FIX)
        return self.sz.Opt(**kwargs)

    def memo(self, key, compute):
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]


# -- the checks ---------------------------------------------------------------------


def _contract(kit):
    alg = kit.alg
    try:
        check_style(alg)
    except TypeError as e:
        raise _Failed(str(e)) from None
    _require(isinstance(alg.scheme, str), f"scheme is {alg.scheme!r}; it must be a str")
    order = alg.order
    _require(isinstance(order, numbers.Integral) and not isinstance(order, bool) and order > 0,
             f"order is {order!r}; it must be a positive int")
    error_order = order if alg.error_order is None else alg.error_order
    for trait, value in (('error_order', error_order), ('interp_order', alg.interp_order)):
        _require(isinstance(value, numbers.Real) and not isinstance(value, bool) and value > 0
                 and math.isfinite(value), f"{trait} is {value!r}; it must be a positive number")
    if kit.explicit:
        dae, y0 = kit.model('A')
        try:
            init(dae, [0, 1], y0, alg, kit.opt())
        except TypeError as e:
            # the core's refusal, and not a TypeError of the algorithm's own
            _require('is explicit and cannot integrate a model with algebraic equations' in str(e),
                     f"the Integrator raised on a model with an algebraic equation, but not for the "
                     f"explicit trait: TypeError: {e}")
        else:
            raise _Failed("the Integrator accepted an explicit algorithm on a model with an algebraic "
                          "equation")
    return {'order': order, 'error_order': error_order, 'interp_order': alg.interp_order}


def _fixed_step_runs(kit):
    """Per main model, the runs with ``h = 2**-k``, ``k = 1..6``, through ``step()``.

    Each run gives the largest error of ``x`` over its saved rows and the
    largest error of ``x`` interpolated at ``theta`` 0.25, 0.5 and 0.75 of
    every step. The first fault of the interpolant is kept for the
    interpolant check, so that the order check reads only the steps.
    """
    runs = {}
    for name in kit.main:
        dae, y0 = kit.model(name)
        run = SimpleNamespace(errors=[], interior=[], end_deviation=0.0, fault=None)
        for k in ORDER_STEPS:
            opt = kit.opt(fix_h=True, hinit=2.0 ** -k, rtol=1e-12, atol=1e-14)
            integ = init(dae, [0, 1], y0, kit.alg, opt)
            worst = 0.0
            for _ in _steps(integ):
                if run.fault is not None:
                    continue
                try:
                    worst = max(worst, _interpolate_step(integ, run))
                except _Failed as e:
                    run.fault = f"on {name} with h = 2**-{k} {e}"
                except Exception as e:
                    run.fault = f"on {name} with h = 2**-{k} the interpolant raised {type(e).__name__}: {e}"
            sol = integ.solve()
            _succeeded(sol, f"the run on {name} with h = 2**-{k}")
            run.errors.append(float(np.max(np.abs(sol.Y[:, 0] - _x_exact(sol.T)))))
            run.interior.append(worst)
        runs[name] = run
    return runs


def _interpolate_step(integ, run):
    """The largest interior error of ``x`` in the step just accepted.

    The two ends are evaluated through the algorithm's interpolant, since
    ``interp`` returns ``uprev`` and ``u`` themselves there: ``theta = 0`` must
    give ``uprev`` exactly, and ``theta = 1`` the end of the step within
    ``1e-10`` relative.
    """
    buf = np.empty(integ.n)
    start = integ._interpolate(0.0, buf)
    _require(np.array_equal(start, integ.uprev),
             f"the interpolant at theta = 0 is not uprev at t = {integ.tprev!r}")
    end = integ._interpolate(1.0, buf)
    deviation = float(np.max(np.abs(end - integ.u_step)) / (1.0 + np.max(np.abs(integ.u_step))))
    run.end_deviation = max(run.end_deviation, deviation)
    _require(deviation <= 1e-10, f"the interpolant at theta = 1 deviates from the end of the step at "
                                 f"t = {integ.t!r} by {deviation:.3e} relative")
    worst = 0.0
    for theta in (0.25, 0.5, 0.75):
        tq = integ.tprev + theta * integ.dt_step
        worst = max(worst, abs(integ.interp(tq)[0] - _x_exact(tq)))
    return worst


def _order_criterion(kit, errors, declared, what, model):
    """The mean of the two finest slopes whose errors lie above ``ERROR_FLOOR``
    must reach ``declared - order_tol``."""
    slopes = _slopes(errors)
    usable = [x for x, e in zip(slopes, errors[1:]) if e >= ERROR_FLOOR]
    steps = f"h = 2**-{ORDER_STEPS[0]} .. 2**-{ORDER_STEPS[-1]}"
    _require(len(usable) >= 2,
             f"on {model} the {what} of {steps} are {errors}; fewer than two slopes lie above the "
             f"rounding level {ERROR_FLOOR}")
    measured = float(np.mean(usable[-2:]))
    _require(measured >= declared - kit.order_tol,
             f"on {model} the {what} of {steps} are {errors}, with the slopes {slopes.tolist()}; the mean "
             f"of the two finest above {ERROR_FLOOR}, {measured:.3f}, is below {declared} - {kit.order_tol}")
    return {'errors': errors, 'slopes': slopes.tolist(), 'measured': measured}


def _order(kit):
    """The largest error of ``x`` over the saved rows, since the error at ``t = 1``
    alone can be ruled by a higher-order term where the first-order error
    coefficient passes near zero, as for ``ImplicitEuler`` on A."""
    runs = kit.memo('fixed', lambda: _fixed_step_runs(kit))
    return {name: _order_criterion(kit, r.errors, kit.alg.order, 'largest errors of x', name)
            for name, r in runs.items()}


def _interpolant(kit):
    runs = kit.memo('fixed', lambda: _fixed_step_runs(kit))
    out = {}
    for name, r in runs.items():
        _require(r.fault is None, r.fault)
        out[name] = _order_criterion(kit, r.interior, kit.alg.interp_order,
                                     'largest interpolation errors of x', name)
        out[name]['end_deviation'] = r.end_deviation
    return out


def _saveat(kit):
    """Saving on a grid does not change the steps, and a saved node is the interpolant there.

    The run on ``[0, 1]`` records ``interp`` at the nodes right after each
    step, so that both runs interpolate at the same times; a third run
    interpolates nowhere, which shows an ``addsteps`` or an interpolant that
    changes the steps.
    """
    dae, y0 = kit.model(kit.base)
    opt = kit.opt(rtol=1e-6, atol=1e-8)
    nodes = np.linspace(0, 1, 11)
    integ = init(dae, [0, 1], y0, kit.alg, opt)
    times, recorded, idx = [], [], 1
    for _ in _steps(integ):
        times.append(integ.t)
        while idx < nodes.size and nodes[idx] <= integ.t:
            recorded.append(integ.interp(nodes[idx]))
            idx += 1
    first = integ.solve()
    integ = init(dae, nodes, y0, kit.alg, opt)
    grid_times = [integ.t for _ in _steps(integ)]
    grid = integ.solve()
    plain = solve(dae, [0, 1], y0, kit.alg, opt)
    for sol, what in ((first, 'the run on [0, 1]'), (grid, 'the run on the grid'),
                      (plain, 'the run without interpolation')):
        _succeeded(sol, what)
    times = np.array(times, dtype=np.float64)
    _require(_byte_equal(times, np.array(grid_times, dtype=np.float64)),
             f"the run on the grid takes other steps than the run on [0, 1]: {len(grid_times)} "
             f"against {times.size} accepted steps")
    _require(_byte_equal(plain.T[1:], times),
             f"interpolating changes the steps: {times.size} accepted steps with interpolation "
             f"against {plain.T.size - 1} without")
    _require(_byte_equal(grid.T, nodes), "the saved times are not the nodes")
    _require(_byte_equal(grid.Y[0], first.Y[0]) and _byte_equal(grid.Y[1:], np.array(recorded)),
             "a saved node differs from the interpolant there")
    return {'nstep': int(times.size)}


def _tstops(kit):
    dae, y0 = kit.model(kit.base)
    stops = (0.25, 0.5, 0.75)
    sol = solve(dae, [0, 1], y0, kit.alg, kit.opt(), tstops=stops)
    _succeeded(sol, 'the run with tstops')
    for ts in stops:
        _require(np.any(sol.T == ts), f"no step ends on the stop time {ts}")
    _require(np.all(np.diff(sol.T) > 0), "the saved times do not increase strictly")
    _require(sol.T[-1] == 1.0, f"the run ends at {sol.T[-1]!r}, not at 1.0")
    return {'nstep': sol.stats.nstep}


def _ball_event(t, y):
    return np.array([y[0], y[0]]), np.array([1, 1]), np.array([-1, -1])


def _ball_run(kit):
    dae, y0 = kit.model('B')
    return solve(dae, [0, 30], y0, kit.alg, kit.opt(event=_ball_event, rtol=1e-6, atol=1e-8))


def _events(kit):
    """Two identical terminal components of the ball, which starts on their root."""
    sol = kit.memo('ball', lambda: _ball_run(kit))
    te_exact = 40 / 9.8
    tol = 1e-3 * te_exact if kit.adaptive else 10 * H_FIX
    _require(sol.stats.ret == 'terminated', f"the run ended with ret = {sol.stats.ret!r}, not at an event")
    _require(sol.ie is not None and np.array_equal(sol.ie, [0, 1]),
             f"the components recorded are {sol.ie!r}, not [0, 1]")
    te = sol.te
    _require(np.all(te > sol.T[0]), "an event is reported at the initial time")
    _require(te[0] == te[1], f"the two identical components cross at {te[0]!r} and {te[1]!r}")
    _require(abs(te[0] - te_exact) <= tol,
             f"the impact is found at {te[0]!r}, {abs(te[0] - te_exact):.3e} from {te_exact!r}")
    _require(sol.T[-1] == te[0], f"the run ends at {sol.T[-1]!r}, not at the event time {te[0]!r}")
    _require(_byte_equal(sol.Y[-1], sol.ye[0]), "the last row is not the recorded state of the event")
    return {'te': float(te[0]), 'deviation': float(abs(te[0] - te_exact))}


def _events_on_grid(kit):
    """The rows before a terminal event equal those of the run without it.

    The grid also holds a node halfway between the start of the crossing
    step and the event, both taken from the run without a grid, so that the
    crossing step is interpolated after the event moved the state to ``te``.
    """
    ref = kit.memo('ball', lambda: _ball_run(kit))
    _require(ref.stats.ret == 'terminated', "the run without a grid did not stop at the event")
    te = ref.te[0]
    grid = np.union1d(np.linspace(0, 30, 61), [0.5 * (ref.T[-2] + te)])
    dae, y0 = kit.model('B')
    with_event = solve(dae, grid, y0, kit.alg, kit.opt(event=_ball_event, rtol=1e-6, atol=1e-8))
    without = solve(dae, grid, y0, kit.alg, kit.opt(rtol=1e-6, atol=1e-8))
    _require(with_event.stats.ret == 'terminated', "the run on the grid did not stop at the event")
    _succeeded(without, 'the run without the event')
    before = with_event.T < with_event.te[0]
    n = int(np.count_nonzero(before))
    _require(_byte_equal(with_event.T[:n], without.T[:n]) and _byte_equal(with_event.Y[:n], without.Y[:n]),
             "a row saved before the event differs from the run without the event")
    return {'rows': n}


def _inconsistent_start(kit):
    dae, y0 = kit.model('A_delta')
    sol = solve(dae, [0, 1], y0, kit.alg, kit.opt(rtol=1e-6, atol=1e-10))
    # the check is vacuous unless DaeIc leaves the residual in place
    if sol.Y[0, 1] != 1e-7:
        raise RuntimeError(f"DaeIc changed z from 1e-7 to {sol.Y[0, 1]!r}; the check tests nothing")
    _succeeded(sol, 'the run from an algebraic residual of 1e-7 at atol = 1e-10')
    return {'nstep': sol.stats.nstep, 'nreject': sol.stats.nreject}


def _history(kit):
    """``k = 2`` from 0.5 in one run, and in the second of two calls split at 0.5.

    The modification protocol returns the run to the state of a new call, so
    every row from the change on is byte-equal between the two unless the
    algorithm keeps data across it.
    """
    dae, y0, _ = kit.build(kit.base)
    opt = kit.opt(fix_h=True, hinit=H_FIX)

    def set_k(integ):
        integ.dae.p['k'][0] = 2.0

    run = solve(dae, [0, 1], y0, kit.alg, opt, callbacks=[preset_time_callback([0.5], set_k)])
    _succeeded(run, 'the run changed at 0.5')
    dae.p['k'][0] = 1.0
    first = solve(dae, [0, 0.5], y0, kit.alg, opt)
    _succeeded(first, 'the call on [0, 0.5]')
    dae.p['k'][0] = 2.0
    second = solve(dae, [0.5, 1], first.Y[-1], kit.alg, opt)
    _succeeded(second, 'the call on [0.5, 1]')
    k = int(np.flatnonzero(run.T == 0.5)[-1])
    _require(_byte_equal(run.T[k:], second.T) and _byte_equal(run.Y[k:], second.Y),
             "the run after the change at 0.5 differs from a new call from there; the algorithm keeps "
             "data across the modification, which reset_history must clear")
    return {'rows': int(second.T.size)}


def _failure(kit):
    """``x' = x**3`` blows up at ``t = 0.5``; the run fails, returns its rows and
    prints one line."""
    dae, y0 = kit.model('C')
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        sol = solve(dae, [0, 1], y0, kit.alg, kit.opt())
    lines = printed.getvalue().splitlines()
    _require(sol.stats.ret == 'failed' and sol.stats.succeed is False,
             f"the run ended with ret = {sol.stats.ret!r} and succeed = {sol.stats.succeed!r}")
    _require(sol.T[-1] < 1, f"the run failed at {sol.T[-1]!r}, not before the end 1")
    _require(sol.T.size >= 2, "the failed run returns no accepted step")
    _require(len(lines) == 1, f"the failed run printed {len(lines)} lines, not one: {lines!r}")
    return {'t_end': float(sol.T[-1]), 'message': lines[0]}


def _counters(stats):
    return (stats.nstep, stats.nreject, stats.nfeval, stats.nJeval, stats.ndecomp, stats.nsolve,
            stats.ncondition)


def _out(kit):
    """A residual without ``out``, which the core adapts, gives the same run."""
    dae, y0 = kit.model(kit.base)
    F = dae.F
    plain = nDAE(dae.M, lambda t, y, p: F(t, y, p), dae.J, dae.p)
    opt = kit.opt()
    a = solve(dae, [0, 1], y0, kit.alg, opt)
    b = solve(plain, [0, 1], y0, kit.alg, opt)
    _succeeded(a, 'the run')
    _require(_byte_equal(a.T, b.T) and _byte_equal(a.Y, b.Y),
             "a residual without out gives another trajectory")
    _require(_counters(a.stats) == _counters(b.stats),
             f"a residual without out gives other counters: {_counters(b.stats)} against "
             f"{_counters(a.stats)}")
    return {'nstep': a.stats.nstep}


def _opt(kit):
    """Ten calls with one ``Opt``, which no call changes, give one trajectory."""
    dae, y0 = kit.model(kit.base)
    opt = kit.opt()
    fields = dict(vars(opt))
    ref = None
    for i in range(10):
        sol = solve(dae, [0, 1], y0, kit.alg, opt)
        _require(vars(opt) == fields, f"call {i + 1} changed the Opt")
        _succeeded(sol, f"call {i + 1}")
        if ref is None:
            ref = sol
        else:
            _require(_byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y),
                     f"call {i + 1} gives another trajectory than the first")
    return {'calls': 10}


def _y0(kit):
    """Neither form of the initial state is written, and both give one trajectory."""
    dae, y0 = kit.model(kit.base)
    y0v = kit.vars(kit.base)
    keep, keep_v = y0.copy(), y0v.array.copy()
    opt = kit.opt()
    a = solve(dae, [0, 1], y0, kit.alg, opt)
    b = solve(dae, [0, 1], y0v, kit.alg, opt)
    _require(_byte_equal(y0, keep), "the run wrote into the ndarray y0")
    _require(_byte_equal(y0v.array, keep_v), "the run wrote into the Vars y0")
    _succeeded(a, 'the run')
    _require(_byte_equal(a.T, b.T) and _byte_equal(a.Y, b.Y.array),
             "a Vars y0 gives another trajectory than its array")
    return {'nstep': a.stats.nstep}


def _counts(kit):
    """The counters of ``Stats`` are the calls the model received."""
    dae, y0 = kit.model(kit.base)
    F, J = dae.F, dae.J
    calls = {'F': 0, 'J': 0}

    def counted_F(t, y, p, out=None):
        calls['F'] += 1
        return F(t, y, p, out=out)

    def counted_J(t, y, p):
        calls['J'] += 1
        return J(t, y, p)

    sol = solve(nDAE(dae.M, counted_F, counted_J, dae.p), [0, 1], y0, kit.alg, kit.opt())
    s = sol.stats
    _succeeded(sol, 'the run')
    _require(s.nfeval == calls['F'], f"nfeval is {s.nfeval}, but F was called {calls['F']} times")
    _require(s.nJeval == calls['J'], f"nJeval is {s.nJeval}, but J was called {calls['J']} times")
    _require(s.nstep == sol.T.size - 1, f"nstep is {s.nstep}, but {sol.T.size - 1} steps were saved")
    return {'nfeval': s.nfeval, 'nJeval': s.nJeval, 'nstep': s.nstep}


def _rendered(kit):
    """The model rendered with ``jit=True`` against the same model inline."""
    sdae, y0v = _symbolic(kit.sz, kit.base)
    dae, y0 = kit.model(kit.base)
    name = f"_sz_check_algorithm_{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory() as directory:
        # the renderer and the import of the module print their progress
        with contextlib.redirect_stdout(io.StringIO()):
            kit.sz.module_printer(sdae, y0v, name, directory=directory, jit=True).render()
            sys.path.insert(0, directory)
            try:
                module = importlib.import_module(name)
            finally:
                sys.path.remove(directory)
        try:
            nodes = np.linspace(0, 1, 11)
            rtol = 1e-6
            opt = kit.opt(rtol=rtol, atol=1e-8)
            inline = solve(dae, nodes, y0, kit.alg, opt)
            rendered = solve(module.mdl, nodes, y0, kit.alg, opt)
        finally:
            for key in [k for k in sys.modules if k == name or k.startswith(name + '.')]:
                del sys.modules[key]
    _succeeded(inline, 'the inline run')
    _succeeded(rendered, 'the rendered run')
    dsteps, dY, ok = rendered_matches_inline(rendered, inline, rtol)
    _require(ok, f"the rendered run takes {rendered.stats.nstep} steps against {inline.stats.nstep} and "
                 f"deviates by {dY:.3e}; at most {MAX_STEP_DIFFERENCE} steps and {MAX_DY_IN_RTOL} * rtol")
    return {'dsteps': dsteps, 'dY': dY}


_CHECK = {
    'contract': _contract,
    'order': _order,
    'interpolant': _interpolant,
    'saveat': _saveat,
    'tstops': _tstops,
    'events': _events,
    'events on a grid': _events_on_grid,
    'inconsistent start': _inconsistent_start,
    'history': _history,
    'failure': _failure,
    'out': _out,
    'Opt': _opt,
    'y0': _y0,
    'counts': _counts,
    'rendered': _rendered,
}
