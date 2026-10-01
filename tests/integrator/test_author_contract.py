"""The author contract and the conformance kit.

An algorithm writes its step as ``perform_step`` in the formula or the
in-place style; the core raises ``TypeError`` when the step or its results
break the contract. A subclass may change the style and keeps every hook of
its parent, and a Rosenbrock method is its table.

``check_algorithm`` passes for every built-in algorithm and for toy
algorithms of the other traits: explicit, and without an error estimate.
Each check rejects an algorithm broken in the way the check guards against,
and each broken algorithm below passes every earlier check; an algebraic
variable left wrong by the step, or interpolated wrongly, is rejected
although ``x`` stays accurate. The checks ``Opt`` and ``y0`` are shown
alone. The contract gives no algorithm access to the caller's ``Opt`` or
``y0``, so ``y0`` fails only with a core that writes the initial state. An
algorithm with a state across calls breaks the repeatability that ``Opt``
checks, but the two calls of ``saveat`` meet it first.
"""
import re
from types import SimpleNamespace

import numpy as np
import pytest

from Solverz.integrator import (Algorithm, ImplicitEuler, Integrator, Rodas3, Rodas4, Rodas5P, Rodasp,
                                Rosenbrock, RosenbrockTableau, Trapezoid, init, solve)
from Solverz.integrator.algorithm import StepContext
from Solverz.integrator.testing import CHECKS, _check, check_algorithm
from Solverz.solvers.daesolver.rodas.param import Rodas_param
from Solverz.solvers.daesolver.rodas.rodas import ntrp2
from Solverz.solvers.option import Opt
from Solverz.variable.variables import Vars

pytestmark = pytest.mark.i7b


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _fails(alg, check, run=check_algorithm, **kwargs):
    """The message of the ``AssertionError`` by which the kit rejects ``alg`` at ``check``."""
    with pytest.raises(AssertionError) as e:
        run(alg, **kwargs)
    message = str(e.value)
    print(message)
    assert message.startswith(f"check_algorithm[{check}]: "), message
    return message


# -- the contract, enforced by the core -------------------------------------------


class _Step(Algorithm):
    """Backward Euler as a formula, with its traits and result set per test."""

    scheme = 'step'
    order = 1
    error_order = 2
    adaptive = True

    def __init__(self, result=lambda y, err, n: (y, err)):
        self.result = result

    def perform_step(self, s):
        y = s.implicit(s.t + s.h, 1.0, s.M @ s.y0)
        err = s.W(1.0).solve(0.5 * (s.M @ (y - s.y0) - s.h * (s.D * s.F0)))
        return self.result(y, err, s.n)


class _NoPerformStep(Algorithm):
    scheme = 'no_perform_step'
    order = 1


class _WrongStyle(Algorithm):
    scheme = 'wrong_style'
    order = 1
    inplace = True

    def perform_step(self, s):
        return s.y0


class _Unadaptive(_Step):
    adaptive = False


class _NoEEst(Algorithm):
    scheme = 'no_eest'
    order = 1
    adaptive = True
    inplace = True

    def perform_step(self, integ, cache):
        integ.implicit(integ.t + integ.dt, 1.0, integ.M @ integ.uprev, out=integ.u)


class _ExplicitEuler(Algorithm):
    """Explicit Euler, without an error estimate."""

    scheme = 'explicit_euler'
    order = 1
    explicit = True

    def perform_step(self, s):
        return s.y0 + s.h * s.f(s.t, s.y0)


def test_the_core_raises_on_a_broken_contract(model):
    ode, y0 = model('vdp')
    dae, z0 = model('dae_test')
    opt = Opt(hinit=2 ** -6)
    cases = [
        (_NoPerformStep(), ode, y0, Opt(), '_NoPerformStep does not define perform_step'),
        (_WrongStyle(), ode, y0, Opt(), '_WrongStyle.perform_step takes 1 parameters after self, but '
                                        'inplace = True needs 2'),
        (_Step(lambda y, err, n: y), ode, y0, Opt(), 'step.perform_step returned no error estimate; '
                                                     'set adaptive = False or run with opt.fix_h'),
        (_Unadaptive(), ode, y0, opt, 'step.perform_step returned an error estimate but declares '
                                      'adaptive = False'),
        (_NoEEst(), ode, y0, Opt(), 'no_eest.perform_step set no error estimate'),
        (_Step(lambda y, err, n: (y[:-1], err)), ode, y0, Opt(),
         'step.perform_step returned y of shape (1,)'),
        (_Step(lambda y, err, n: (y, np.linalg.norm(err))), ode, y0, Opt(),
         'step.perform_step returned an error estimate of shape ()'),
        (_ExplicitEuler(), dae, z0, opt, 'explicit_euler is explicit and cannot integrate a model with '
                                         'algebraic equations or a singular mass matrix'),
    ]
    for alg, m, y, o, message in cases:
        with pytest.raises(TypeError, match=re.escape(message)):
            solve(m, [0, 1], y, alg=alg, opt=o)
    # with opt.fix_h the estimate of an adaptive algorithm is not read, and a
    # missing one is no error
    sol = solve(ode, [0, 1], y0, alg=_Step(lambda y, err, n: y), opt=Opt(fix_h=True, hinit=2 ** -6))
    assert sol.stats.ret == 'success' and sol.stats.nstep == 64


class _InplaceToy(Algorithm):
    """Backward Euler in place, with hooks that leave a trace in its cache."""

    scheme = 'inplace_toy'
    order = 1
    error_order = 2
    adaptive = True
    inplace = True

    def alloc(self, integ):
        return SimpleNamespace(owner='inplace_toy', addsteps=0, interpolated=0, contexts=set())

    def addsteps(self, integ, cache):
        cache.addsteps += 1

    def interpolant(self, integ, cache, theta, out):
        cache.interpolated += 1
        return integ.uprev + theta * (integ.u_step - integ.uprev)

    def perform_step(self, integ, cache):
        y0, M, dt = integ.uprev, integ.M, integ.dt
        integ.implicit(integ.t + dt, 1.0, M @ y0, out=integ.u)
        err = integ.W(1.0).solve(0.5 * (M @ (integ.u - y0) - dt * (integ.D() * integ.F0())))
        integ.EEst = integ.error_norm(err)


class _FormulaToy(_InplaceToy):
    """The same method as a formula, which keeps every hook of its parent."""

    scheme = 'formula_toy'
    inplace = False

    def perform_step(self, s):
        s.cache.contexts.add(type(s))
        return ImplicitEuler.perform_step(self, s)


def test_a_formula_subclass_of_an_in_place_class(model):
    """The subclass takes the formula dispatch and its parent's ``alloc``,
    ``addsteps`` and ``interpolant``; in place and as a formula the method
    takes the steps of ``ImplicitEuler`` bit for bit."""
    dae, y0 = model('dae_test')
    opt = Opt(rtol=1e-6, atol=1e-8)
    runs = {}
    for alg in (_InplaceToy(), _FormulaToy(), ImplicitEuler()):
        integ = init(dae, np.linspace(0, 2, 21), y0, alg=alg, opt=opt)
        sol = integ.solve()
        assert sol.stats.ret == 'success'
        runs[alg.scheme] = (sol, integ.cache)
    ref = runs['implicit_euler'][0]
    for scheme in ('inplace_toy', 'formula_toy'):
        sol, cache = runs[scheme]
        assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y) and sol.stats.nstep == ref.stats.nstep
        assert cache.owner == 'inplace_toy' and 0 < cache.addsteps <= cache.interpolated <= 20
    assert runs['formula_toy'][1].contexts == {StepContext}
    assert runs['inplace_toy'][1].contexts == set()


def _rodas4_by_its_tables():
    ref = Rodas_param('rodas4')

    class TableRodas4(Rosenbrock):
        scheme = 'table_rodas4'
        tableau = RosenbrockTableau.from_hairer(ref.gamma, ref.alpha.T, beta=ref.beta, b=ref.b, bd=ref.bd,
                                                pord=ref.pord, c=ref.c, d=ref.d, e=ref.e)

    return TableRodas4


@pytest.mark.parametrize('name', ['dae_test', 'vdp'])
def test_a_table_only_subclass_takes_the_steps_of_rodas4(model, name):
    TableRodas4 = _rodas4_by_its_tables()
    assert (TableRodas4.order, TableRodas4.interpolation) == (4, 'ntrp1')
    dae, y0 = model(name)
    for tspan in ([0, 5], np.linspace(0, 5, 51)):
        opt = Opt(rtol=1e-6, atol=1e-8)
        a = solve(dae, tspan, y0, alg=TableRodas4(), opt=opt)
        b = solve(dae, tspan, y0, alg=Rodas4(), opt=opt)
        assert a.stats.ret == b.stats.ret == 'success'
        assert _byte_equal(a.T, b.T) and _byte_equal(a.Y, b.Y) and a.stats.nstep == b.stats.nstep


# -- the kit on the built-in algorithms and on toy algorithms ------------------------


@pytest.mark.parametrize('method', [Rodas3, Rodas4, Rodasp, Rodas5P, ImplicitEuler, Trapezoid])
def test_the_built_in_algorithms_pass_the_kit(method):
    results = check_algorithm(method(), rendered=True)
    assert tuple(results) == CHECKS
    for check in ('order', 'interpolant'):
        for name, measured in results[check].items():
            print(f"{method.scheme} {check} on {name}: slopes {np.round(measured['slopes'], 3).tolist()}, "
                  f"measured {measured['measured']:.3f}")
    print(f"{method.scheme}: te deviation {results['events']['deviation']:.3e}, "
          f"rendered {results['rendered']}")


class _Heun(Algorithm):
    """Heun's method with the explicit Euler step as its error estimate."""

    scheme = 'heun'
    order = 2
    error_order = 2
    adaptive = True
    explicit = True

    def perform_step(self, s):
        k1 = s.f(s.t, s.y0)
        k2 = s.f(s.t + s.h, s.y0 + s.h * k1)
        return s.y0 + 0.5 * s.h * (k1 + k2), 0.5 * s.h * (k2 - k1)


class _FsalTrapezoid(Trapezoid):
    """The trapezoidal rule whose ``f0`` is the slope that ``implicit`` returned
    at the end of the last accepted step, first same as last; it does not
    clear the slope when the model changes."""

    scheme = 'fsal_trapezoid'

    def alloc(self, integ):
        return SimpleNamespace(slope=None, candidate=None)

    def perform_step(self, s):
        c = s.cache
        if s.new_step:
            # the last attempt was accepted: its end slope starts this step
            c.slope = c.candidate
        f0 = s.D * s.F0 if c.slope is None else c.slope
        y, k = s.implicit(s.t + s.h, 0.5, s.M @ s.y0 + 0.5 * s.h * f0, slope=True)
        c.candidate = s.D * k
        return y, s.W(0.5).solve(s.M @ (y - s.y0) - s.h * f0)


class _ClearedFsalTrapezoid(_FsalTrapezoid):
    scheme = 'cleared_fsal_trapezoid'

    def reset_history(self, integ, cache):
        cache.slope = cache.candidate = None


# explicit Euler overflows on C, as it should
@pytest.mark.filterwarnings('ignore:overflow encountered:RuntimeWarning')
@pytest.mark.parametrize('alg, skipped', [(_Heun(), ('algebraic event', 'inconsistent start')),
                                          (_ExplicitEuler(), ('algebraic event', 'inconsistent start')),
                                          (_ClearedFsalTrapezoid(), ())],
                         ids=['explicit', 'fixed_step', 'history_cleared'])
def test_toy_algorithms_pass_the_kit(alg, skipped):
    """An explicit algorithm runs on E and skips the algebraic event and the
    inconsistent start; one without an error estimate runs with the fixed
    step ``2**-6``, and explicit Euler, whose numerical solution of ``x' =
    x**3`` overflows only after the blow-up at 0.5, still fails before the
    end."""
    results = check_algorithm(alg)
    assert tuple(results) == tuple(c for c in CHECKS[:-1] if c not in skipped)
    main = ('E',) if alg.explicit else ('A', 'P')
    assert tuple(results['order']) == main
    print(f"{alg.scheme}: {results['order']}, failure {results['failure']}")


def test_the_kit_takes_an_instance_of_the_default_configuration():
    with pytest.raises(TypeError, match=re.escape('such as Rodas4()')):
        check_algorithm(Rodas4)
    with pytest.raises(ValueError, match='the default configuration'):
        check_algorithm(Rodas4(legacy_compat=True))


# -- each check rejects the algorithm it guards against -----------------------------


class _Traits(Rodas4):
    scheme = 'traits'


@pytest.mark.parametrize('traits, message', [
    (dict(order=0), 'order is 0; it must be a positive int'),
    (dict(order=4.0), 'order is 4.0; it must be a positive int'),
    (dict(scheme=None), 'scheme is None; it must be a str'),
    (dict(error_order=-1), 'error_order is -1; it must be a positive number'),
    (dict(interp_order=None), 'interp_order is None; it must be a positive number')])
def test_contract_rejects_bad_traits(traits, message):
    alg = type('Bad', (_Traits,), traits)()
    assert message in _fails(alg, 'contract')


def test_contract_rejects_a_refusal_of_an_explicit_algorithm_for_another_reason(monkeypatch):
    """The core refuses an explicit algorithm on A with its own message; any
    other ``TypeError`` from ``init`` is no such refusal."""
    def refuses(self, *args, **kwargs):
        raise TypeError('an unrelated error')

    monkeypatch.setattr(Integrator, '__init__', refuses)
    message = _fails(_ExplicitEuler(), 'contract', run=_check, names=('contract',))
    assert 'not for the explicit trait: TypeError: an unrelated error' in message


def test_contract_rejects_a_step_of_the_wrong_style():
    assert 'does not define perform_step' in _fails(_NoPerformStep(), 'contract')
    assert 'inplace = True needs 2' in _fails(_WrongStyle(), 'contract')


class _Overclaimed(Rodas4):
    scheme = 'overclaimed'
    order = 5


def test_order_rejects_an_order_the_method_does_not_reach():
    assert 'is below 5 - 0.3' in _fails(_Overclaimed(), 'order')


def _algebraic(integ):
    """The variables whose columns of ``M`` hold no nonzero value."""
    return np.flatnonzero(np.asarray(abs(integ.M).sum(axis=0)).ravel() == 0)


class _FrozenAlgebraic(Rodas4):
    """Keeps every algebraic variable at its value at the start of the step,
    as a step built from the slopes of the differential rows alone does, and
    interpolates it so, consistently with the step. ``x`` stays accurate,
    since every stage solves the algebraic equation again."""

    scheme = 'frozen_algebraic'

    def perform_step(self, integ, cache):
        super().perform_step(integ, cache)
        alg = _algebraic(integ)
        integ.u[alg] = integ.uprev[alg]

    def interpolant(self, integ, cache, theta, out):
        super().interpolant(integ, cache, theta, out)
        alg = _algebraic(integ)
        out[alg] = integ.uprev[alg]


def test_order_rejects_an_algebraic_variable_that_the_step_leaves_wrong():
    assert 'is below 4 - 0.3' in _fails(_FrozenAlgebraic(), 'order')


class _OverclaimedDense(Rodas4):
    scheme = 'overclaimed_dense'
    interp_order = 5


class _HeldStart(Rodas4):
    """An interpolant that holds the start of the step."""

    scheme = 'held_start'

    def interpolant(self, integ, cache, theta, out):
        np.copyto(out, integ.uprev)


class _HeldEnd(Rodas4):
    """An interpolant that holds the end of the step."""

    scheme = 'held_end'

    def interpolant(self, integ, cache, theta, out):
        np.copyto(out, integ.u_step)


class _SecantAlgebraic(Rodas4):
    """Interpolates the algebraic variables along the secant of the step, an
    interpolant of order 2 where Rodas4 declares 3; ``x`` keeps its own."""

    scheme = 'secant_algebraic'

    def interpolant(self, integ, cache, theta, out):
        super().interpolant(integ, cache, theta, out)
        alg = _algebraic(integ)
        out[alg] = integ.uprev[alg] + theta * (integ.u_step[alg] - integ.uprev[alg])


@pytest.mark.parametrize('alg, message', [(_OverclaimedDense(), 'largest interpolation errors'),
                                          (_SecantAlgebraic(), 'largest interpolation errors'),
                                          (_HeldStart(), 'at theta = 1 deviates from the end'),
                                          (_HeldEnd(), 'at theta = 0 is not uprev')],
                         ids=['order', 'algebraic', 'end', 'start'])
def test_interpolant_rejects_a_wrong_interpolant(alg, message):
    assert message in _fails(alg, 'interpolant')


class _GuessFromInterpolant(ImplicitEuler):
    """Backward Euler whose Newton iteration starts from the last value its
    interpolant returned, so that interpolating changes the next step."""

    scheme = 'guess_from_interpolant'

    def alloc(self, integ):
        return SimpleNamespace(guess=None)

    def interpolant(self, integ, cache, theta, out):
        super().interpolant(integ, cache, theta, out)
        cache.guess = out.copy()

    def perform_step(self, s):
        guess, s.cache.guess = s.cache.guess, None
        y = s.implicit(s.t + s.h, 1.0, s.M @ s.y0, y=guess)
        return y, s.W(1.0).solve(0.5 * (s.M @ (y - s.y0) - s.h * (s.D * s.F0)))


def test_saveat_rejects_an_interpolant_that_changes_the_steps():
    assert 'interpolating changes the steps' in _fails(_GuessFromInterpolant(), 'saveat')


class _OwnStep(Rodas4):
    """An adaptive run that takes at least 0.3 whatever step the core gives,
    and reports no error."""

    scheme = 'own_step'

    def perform_step(self, integ, cache):
        if integ.opts.adaptive:
            integ.dt = max(integ.dt, 0.3)
        super().perform_step(integ, cache)
        if integ.opts.adaptive:
            integ.EEst = 0.0


def test_tstops_rejects_an_algorithm_that_changes_its_step():
    assert 'do not increase strictly' in _fails(_OwnStep(), 'tstops')


class _Careless(ImplicitEuler):
    """Backward Euler whose error estimate is a billion times too small."""

    scheme = 'careless'

    def perform_step(self, s):
        y, err = super().perform_step(s)
        return y, 1e-9 * err


def test_events_rejects_an_inaccurate_event_time():
    assert 'the impact is found at' in _fails(_Careless(), 'events')


class _ReadsU(Rodas3):
    """The Hermite interpolant of Rodas3 with the end state read from ``u``,
    which an event moves, instead of ``u_step``."""

    scheme = 'reads_u'

    def interpolant(self, integ, cache, theta, out):
        out[...] = ntrp2(integ.uprev, integ.u, cache.s0, cache.s1, theta, integ.dt_step)


def test_events_on_a_grid_rejects_an_interpolant_that_reads_u():
    assert 'a row saved before the event differs' in _fails(_ReadsU(), 'events on a grid')


class _WrongBetweenSamples(Rodas4):
    """Adds ``0.1 sin(4 pi theta)`` to the algebraic variables, which vanishes
    at the points 0, 0.25, 0.5, 0.75 and 1 that the interpolant check reads,
    and moves a crossing of ``z`` located between them."""

    scheme = 'wrong_between_samples'

    def interpolant(self, integ, cache, theta, out):
        super().interpolant(integ, cache, theta, out)
        out[_algebraic(integ)] += 0.1 * np.sin(4 * np.pi * theta)


def test_algebraic_event_rejects_a_wrong_interpolant_of_an_algebraic_variable():
    assert 'z reaches 0.5 at' in _fails(_WrongBetweenSamples(), 'algebraic event')


class _WithoutD(ImplicitEuler):
    """Backward Euler whose error estimate keeps the algebraic rows of ``F0``."""

    scheme = 'without_D'

    def perform_step(self, s):
        y = s.implicit(s.t + s.h, 1.0, s.M @ s.y0)
        return y, s.W(1.0).solve(0.5 * (s.M @ (y - s.y0) - s.h * s.F0))


def test_inconsistent_start_rejects_an_estimate_of_the_algebraic_residual():
    assert "ret = 'failed'" in _fails(_WithoutD(), 'inconsistent start')


def test_history_rejects_data_kept_across_a_modification():
    assert 'reset_history must clear' in _fails(_FsalTrapezoid(), 'history')


class _LoudRejection(Rodas4):
    """Prints a line at every rejected attempt."""

    scheme = 'loud_rejection'

    def perform_step(self, integ, cache):
        super().perform_step(integ, cache)
        if integ.opts.adaptive and integ.EEst > 1.0:
            print(f"{self.scheme}: attempt rejected at t = {integ.t}")


class _RefusesTinySteps(Rodas4):
    """Raises instead of failing the attempt when the step becomes tiny."""

    scheme = 'refuses_tiny_steps'

    def perform_step(self, integ, cache):
        if integ.dt < 1e-12:
            raise ValueError(f"the step {integ.dt} is too small")
        super().perform_step(integ, cache)


@pytest.mark.parametrize('alg, message', [(_LoudRejection(), 'lines, not one'),
                                          (_RefusesTinySteps(), 'ValueError: the step')],
                         ids=['prints', 'raises'])
def test_failure_rejects_a_failure_that_prints_or_raises(alg, message):
    assert message in _fails(alg, 'failure')


class _PositionalOut(Rodas4):
    """Evaluates the model's residual at the end of each attempt itself,
    passing ``out`` by position."""

    scheme = 'positional_out'

    def alloc(self, integ):
        cache = super().alloc(integ)
        cache.end = np.empty(integ.n)
        return cache

    def perform_step(self, integ, cache):
        super().perform_step(integ, cache)
        integ.dae.F(integ.t + integ.dt, integ.u, integ.p, cache.end)


def test_out_rejects_a_residual_called_with_a_positional_out():
    assert 'TypeError' in _fails(_PositionalOut(), 'out')


class _WarmStart(Rodas4):
    """Starts each call with the last step of the call before it."""

    scheme = 'warm_start'
    last = None

    def initial_dt(self, integ):
        return super().initial_dt(integ) if self.last is None else self.last

    def perform_step(self, integ, cache):
        super().perform_step(integ, cache)
        self.last = integ.dt


def test_opt_rejects_a_state_kept_across_calls():
    """In the kit the two calls of ``saveat`` meet it first."""
    assert 'call 2 gives another trajectory than the first' in _fails(_WarmStart(), 'Opt', run=_check,
                                                                        names=('Opt',))
    assert 'other steps' in _fails(_WarmStart(), 'saveat')


def test_y0_rejects_a_run_that_writes_the_initial_state(monkeypatch):
    """No algorithm reaches the caller's ``y0``, which the core copies; a core
    that writes it back fails the check."""
    original = Integrator.__init__

    def writes_y0(self, dae, tspan, y0, *args, **kwargs):
        original(self, dae, tspan, y0, *args, **kwargs)
        (y0.array if isinstance(y0, Vars) else y0)[0] += 1.0

    monkeypatch.setattr(Integrator, '__init__', writes_y0)
    assert 'the run wrote into the ndarray y0' in _fails(Rodas4(), 'y0', run=_check, names=('y0',))


class _Uncounted(Rodas4):
    """Evaluates the model's residual at the end of each attempt itself."""

    scheme = 'uncounted'

    def perform_step(self, integ, cache):
        super().perform_step(integ, cache)
        integ.dae.F(integ.t + integ.dt, integ.u, integ.p)


def test_counts_rejects_a_residual_the_core_does_not_count():
    assert 'nfeval is' in _fails(_Uncounted(), 'counts')


class _FormDependent(Rodas4):
    """Scales its error down when the Jacobian has 32-bit indices, as the one
    of a rendered model has, whose index arrays every call shares."""

    scheme = 'form_dependent'

    def perform_step(self, integ, cache):
        super().perform_step(integ, cache)
        if integ.opts.adaptive and integ.J0().indices.dtype == np.int32:
            integ.EEst = 1e-6 * integ.EEst


def test_rendered_rejects_a_step_that_depends_on_the_form_of_the_model():
    assert 'the rendered run takes' in _fails(_FormDependent(), 'rendered', rendered=True)
