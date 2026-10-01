"""The failure path, the argument errors, and what a call leaves unchanged.

I4, in the legacy-compatible configuration and with in-place test
algorithms: a failing run returns the rows saved so far with ``ret ==
'failed'``, ``succeed is False`` and ``stats.t_fail``, prints exactly one
line and never raises, whether the solution blows up, the error test rejects
more than 100 attempts in a row, every attempt raises ``StepFailure``, the
initial values have no consistent completion, or the residual divides by
zero; a residual that raises an exception of its own propagates. ``Opt``
and ``y0`` are never written, the argument errors are raised before any
step, an empty span gives one row, and the counters are the calls the
model sees, those of ``DaeIc`` and ``dF/dt`` included.

I5, in the default configuration: a residual that divides by zero at a stop
time fails the run without raising, whether Numba raises
``ZeroDivisionError`` there or NumPy returns ``inf``; ``opt=None`` is
``Opt()`` and ``alg=None`` is ``Rodas4()``; an ``opt.scheme`` other than the
algorithm's warns once per call of each entry, at the caller's line.
"""
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numba import njit
from scipy.sparse import csc_array

from Solverz import Eqn, Model, Ode, Var, made_numerical
from Solverz.integrator import Algorithm, Rodas3, Rodas4, Rosenbrock, StepFailure, init, solve
from Solverz.num_api.num_eqn import nDAE
from Solverz.solvers.option import Opt
from Solverz.variable.variables import TimeVars

from tests.integrator import models
from tests.integrator.test_legacy_transcription import SCHEMES


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _one_line(capsys, scheme):
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 1 and out.endswith('\n'), out
    assert lines[0].startswith(f"{scheme}: ") and '; the solution is returned up to t = ' in lines[0]
    return lines[0]


def _failed(sol):
    assert sol.stats.ret == 'failed' and sol.stats.succeed is False


def _blowup():
    m = Model()
    m.x = Var('x', [1.0])
    m.f = Ode('f', m.x ** 2, m.x)
    sdae, y0 = m.create_instance()
    return made_numerical(sdae, y0, sparse=True), np.array(y0.array, dtype=np.float64)


def _no_real_solution(sparse):
    m = Model()
    m.x = Var('x', [1.0])
    m.z = Var('z', [0.5])
    m.f = Ode('f', -m.x, m.x)
    m.g = Eqn('g', m.z ** 2 + 1)
    sdae, y0 = m.create_instance()
    return made_numerical(sdae, y0, sparse=sparse), np.array(y0.array, dtype=np.float64)


def _decay(F):
    """``x' = F`` on two variables, hand-built."""
    return nDAE(csc_array(np.eye(2)), F, lambda t, y, p: csc_array(-np.eye(2)), {})


def _singular_algebraic_block(dense, J=None):
    """``x' = -x``, ``0 = (z - 1)**2 + 1e-3`` from ``x = z = 1``, where the
    algebraic Jacobian ``2 (z - 1)`` is exactly zero, so ``DaeIc`` meets a
    singular matrix at its first solve."""
    M = csc_array(([1.0], ([0], [0])), shape=(2, 2))

    def F(t, y, p, out=None):
        out = np.empty(2) if out is None else out
        out[0] = -y[0]
        out[1] = (y[1] - 1.0) ** 2 + 1e-3
        return out

    def jacobian(t, y, p):
        A = np.array([[-1.0, 0.0], [0.0, 2.0 * (y[1] - 1.0)]])
        return A if dense else csc_array(A)

    return nDAE(M, F, jacobian if J is None else J, {}), np.ones(2)


class _Const(Algorithm):
    """Keeps the state and reports the same error at every attempt."""

    scheme = 'const'
    order = 1
    adaptive = True
    inplace = True
    legacy_compat = True

    def __init__(self, err):
        self.err = err

    def perform_step(self, integ, cache):
        np.copyto(integ.u, integ.uprev)
        integ.EEst = np.float64(self.err)


class _Failing(Algorithm):
    """Fails every attempt."""

    scheme = 'failing'
    order = 1
    adaptive = True
    inplace = True
    legacy_compat = True

    def perform_step(self, integ, cache):
        raise StepFailure('it always fails')


# -- the failure path -------------------------------------------------------


@pytest.mark.i4
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
def test_a_blow_up_fails_without_raising(capsys, grid):
    dae, y0 = _blowup()
    tspan = np.linspace(0, 2, 21) if grid else [0, 2]
    capsys.readouterr()
    sol = Rodas4(legacy_compat=True)(dae, tspan, y0, Opt())
    _failed(sol)
    line = _one_line(capsys, 'rodas4')
    assert sol.T.size >= 2 and sol.T[-1] < 1
    assert np.all(np.isfinite(sol.Y))
    t_fail = sol.stats.t_fail
    assert f"at t = {float(t_fail)!r};" in line and line.endswith(f"up to t = {float(sol.T[-1])!r}.")
    if grid:
        # the prefix holds nodes only, and the failure can lie after the last of them
        assert _byte_equal(sol.T, tspan[:sol.T.size]) and sol.T[-1] <= t_fail < 1
    else:
        assert t_fail == sol.T[-1]


@pytest.mark.i4
def test_more_than_100_rejections_fail_the_run(capsys):
    dae, y0 = models.build('dae_test')
    capsys.readouterr()
    sol = _Const(1.0001)(dae, [0, 1], y0, Opt())
    _failed(sol)
    assert _byte_equal(sol.T, np.array([0.0])) and _byte_equal(sol.Y, y0[None, :])
    assert (sol.stats.nstep, sol.stats.nreject) == (0, 101)
    assert sol.stats.t_fail == 0
    assert _one_line(capsys, 'const') == ("const: more than 100 consecutive attempts were rejected at t = 0.0; "
                                          "the solution is returned up to t = 0.0.")
    # an error of 1 is accepted
    integ = init(dae, [0, 1], y0, alg=_Const(1.0))
    assert integ.step() and (integ.stats.nstep, integ.stats.nreject) == (1, 0)


@pytest.mark.i4
def test_an_attempt_that_always_fails(capsys):
    dae, y0 = models.build('dae_test')
    capsys.readouterr()
    sol = _Failing()(dae, [0, 1], y0, Opt())
    _failed(sol)
    assert _byte_equal(sol.T, np.array([0.0]))
    # every failure halves the step, from 1e-6 until it lies below spacing(1)
    k = next(k for k in range(100) if 1e-6 / 2 ** k < np.spacing(1.0))
    assert (sol.stats.nstep, sol.stats.nreject) == (0, k)
    assert 'is too small at t = 0.0;' in _one_line(capsys, 'failing')
    # with a fixed step the first failure fails the run
    sol = _Failing()(dae, [0, 1], y0, Opt(fix_h=True, hinit=0.1))
    _failed(sol)
    assert sol.stats.nreject == 1
    assert _one_line(capsys, 'failing') == ("failing: the step failed at a fixed step size: it always fails "
                                            "at t = 0.0; the solution is returned up to t = 0.0.")


@pytest.mark.i4
@pytest.mark.parametrize('sparse', [True, False], ids=['sparse', 'dense'])
def test_initial_values_without_a_consistent_completion(capsys, sparse):
    dae, y0 = _no_real_solution(sparse)
    capsys.readouterr()
    sol = Rodas4(legacy_compat=True)(dae, [0, 1], y0, Opt())
    _failed(sol)
    assert _byte_equal(sol.T, np.array([0.0])) and _byte_equal(sol.Y, y0[None, :])
    assert sol.stats.t_fail == 0 and sol.stats.nstep == 0
    assert 'DaeIc found no consistent initial values' in _one_line(capsys, 'rodas4')


@pytest.mark.i4
@pytest.mark.filterwarnings('ignore:Matrix is exactly singular')
@pytest.mark.parametrize('legacy_compat', [True, False], ids=['compat', 'default'])
@pytest.mark.parametrize('dense', [False, True], ids=['sparse', 'dense'])
def test_a_singular_algebraic_jacobian_at_t0_fails_the_run(capsys, dense, legacy_compat):
    """The dense solve raises ``LinAlgError``. The sparse one falls back to
    ``spsolve``, which warns and returns ``NaN``, and ``DaeIc`` then returns
    a state that is not finite, which fails the run as well."""
    dae, y0 = _singular_algebraic_block(dense)
    capsys.readouterr()
    sol = Rodas4(legacy_compat=legacy_compat)(dae, [0, 1], y0, Opt())
    _failed(sol)
    assert _byte_equal(sol.T, np.array([0.0])) and _byte_equal(sol.Y, y0[None, :])
    assert sol.stats.t_fail == 0 and (sol.stats.nstep, sol.stats.nreject) == (0, 0)
    reason = 'LinAlgError: Singular matrix' if dense else 'it returned a state that is not finite'
    assert _one_line(capsys, 'rodas4') == (f"rodas4: DaeIc found no consistent initial values ({reason}) "
                                           f"at t = 0.0; the solution is returned up to t = 0.0.")


@pytest.mark.i4
def test_a_runtime_error_inside_daeic_fails_the_run(capsys):
    """A ``RuntimeError``, which a sparse factorization raises on a singular
    matrix, here raised by the Jacobian that ``DaeIc`` evaluates."""
    def J(t, y, p):
        raise RuntimeError('Factor is exactly singular')

    dae, y0 = _singular_algebraic_block(False, J)
    capsys.readouterr()
    sol = Rodas4()(dae, [0, 1], y0, Opt())
    _failed(sol)
    assert _byte_equal(sol.T, np.array([0.0])) and sol.stats.nstep == 0
    assert _one_line(capsys, 'rodas4') == ("rodas4: DaeIc found no consistent initial values (RuntimeError: "
                                           "Factor is exactly singular) at t = 0.0; the solution is returned "
                                           "up to t = 0.0.")


class _FailingAddsteps(_Const):
    """Accepts every attempt, and the data of its interpolant fail."""

    scheme = 'failing_addsteps'

    def __init__(self):
        super().__init__(0.0)

    def addsteps(self, integ, cache):
        raise StepFailure('ZeroDivisionError in F: float division by zero')


@pytest.mark.i4
def test_a_model_error_while_saving_fails_the_run(capsys):
    dae, y0 = models.build('dae_test')
    capsys.readouterr()
    sol = _FailingAddsteps()(dae, np.linspace(0, 1, 11), y0, Opt())
    _failed(sol)
    # the steps before the first node interpolate nothing, and the step that
    # reaches it fails the run at its end
    assert _byte_equal(sol.T, np.array([0.0])) and sol.stats.nstep >= 1
    assert sol.stats.t_fail >= 0.1
    line = _one_line(capsys, 'failing_addsteps')
    assert line.startswith("failing_addsteps: the model failed after the step was accepted: "
                           "ZeroDivisionError in F: float division by zero at t = ")
    # without a grid nothing is interpolated
    sol = _FailingAddsteps()(dae, [0, 1], y0, Opt())
    assert sol.stats.ret == 'success'


@pytest.mark.i4
def test_a_residual_error_of_its_own_propagates():
    def F(t, y, p, out=None):
        raise ValueError('a residual error of its own')

    with pytest.raises(ValueError, match='a residual error of its own'):
        Rodas4(legacy_compat=True)(_decay(F), [0, 1], np.ones(2), Opt())


@pytest.mark.i4
def test_a_division_by_zero_in_the_residual_fails_the_run(capsys):
    def F(t, y, p, out=None):
        if t >= 0.5:
            raise ZeroDivisionError('float division by zero')
        out = np.empty(2) if out is None else out
        np.negative(y, out=out)
        return out

    sol = Rodas4(legacy_compat=True)(_decay(F), [0, 1], np.ones(2), Opt())
    _failed(sol)
    assert sol.T.size >= 2 and sol.T[-1] < 0.5
    assert sol.stats.nreject > 0
    _one_line(capsys, 'rodas4')


# -- what a call leaves unchanged ---------------------------------------------


def _snapshot(opt):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(opt).items()}


def _unchanged(before, after):
    assert after.keys() == before.keys()
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert _byte_equal(after[k], v), k
        else:
            assert after[k] is v or after[k] == v, k


@pytest.mark.i4
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
def test_ten_calls_with_one_opt(grid):
    dae, y0 = models.build('dae_test')
    tspan = np.linspace(0, 20, 201) if grid else [0, 20]
    opt = Opt(rtol=1e-6, atol=np.array([1e-8, 1e-9]))
    before = _snapshot(opt)
    first = None
    for _ in range(10):
        sol = Rodas4(legacy_compat=True)(dae, tspan, y0, opt)
        _unchanged(before, _snapshot(opt))
        assert opt.hmax is None and opt.facmax == 6
        if first is None:
            first = sol
        assert _byte_equal(sol.T, first.T) and _byte_equal(sol.Y, first.Y)


@pytest.mark.i4
@pytest.mark.parametrize('consistent', [True, False], ids=['consistent', 'inconsistent'])
def test_y0_is_never_written(consistent):
    sdae, y0v = models.dae_test_model()
    dae = made_numerical(sdae, y0v, sparse=True)
    if not consistent:
        y0v.array[:] = [1.0, 1.1]
    ref = y0v.array.copy()
    alg = Rodas4(legacy_compat=True)

    y = ref.copy()
    sol = alg(dae, [0, 20], y, Opt(hinit=0.1))
    assert _byte_equal(y, ref)
    assert type(sol.Y) is np.ndarray and sol.Y.shape == (sol.T.size, 2)
    if not consistent:
        # DaeIc moved the algebraic variable onto the circle
        assert abs(sol.Y[0, 0] ** 2 + sol.Y[0, 1] ** 2 - 2) < 1e-8 and sol.Y[0, 1] != 1.1

    solv = alg(dae, [0, 20], y0v, Opt(hinit=0.1))
    assert _byte_equal(y0v.array, ref)
    assert isinstance(solv.Y, TimeVars) and solv.Y.a is y0v.a
    assert _byte_equal(solv.T, sol.T) and _byte_equal(solv.Y.array, sol.Y)


# -- argument errors ----------------------------------------------------------


@pytest.mark.i4
def test_argument_errors():
    dae, y0 = models.build('dae_test')
    alg = Rodas4(legacy_compat=True)
    with pytest.raises(ValueError, match='^opt.fix_h needs opt.hinit$'):
        alg(dae, [0, 1], y0, Opt(fix_h=True))
    with pytest.raises(ValueError, match='^t0: 1 > tend: 0$'):
        alg(dae, [1, 0], y0, Opt())
    for hinit in (0, 0.0, -0.1):
        with pytest.raises(ValueError, match='is not positive'):
            alg(dae, [0, 1], y0, Opt(hinit=hinit))
    for tstops in ([0.5], np.array([0.25, 0.5])):
        for entry in (solve, init):
            with pytest.raises(ValueError, match='has no tstops'):
                entry(dae, [0, 1], y0, alg=alg, tstops=tstops)
            # the stop times of a discrete callback count as well
            with pytest.raises(ValueError, match='has no tstops'):
                entry(dae, [0, 1], y0, alg=alg, callbacks=[SimpleNamespace(tstops=tstops)])
    # an empty sequence is no tstops
    assert init(dae, [0, 1], y0, alg=alg, tstops=[]).tstops == []


@pytest.mark.i4
def test_an_empty_span_gives_one_row(capsys):
    dae, y0 = models.build('dae_test')
    capsys.readouterr()
    for alg in (Rodas4(legacy_compat=True), Rodas3(legacy_compat=True)):
        sol = alg(dae, [0.5, 0.5], y0, Opt())
        assert _byte_equal(sol.T, np.array([0.5])) and _byte_equal(sol.Y, y0[None, :])
        assert sol.stats.ret == 'success' and sol.stats.succeed is True and sol.stats.nstep == 0
    assert capsys.readouterr().out == ''


# -- the counters -------------------------------------------------------------


class _Counted:
    """The model with its residual and Jacobian counted."""

    def __init__(self, dae):
        self.M, self.p = dae.M, dae.p
        self._dae = dae
        self.nF = self.nJ = 0

    def F(self, t, y, p, out=None):
        self.nF += 1
        return self._dae.F(t, y, p, out=out)

    def J(self, t, y, p):
        self.nJ += 1
        return self._dae.J(t, y, p)


@pytest.mark.i4
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('scheme', SCHEMES)
def test_the_counters_are_the_calls(scheme, grid):
    dae, _ = models.build('dae_test')
    model = _Counted(dae)
    # an inconsistent start, so that DaeIc evaluates the Jacobian as well
    y0 = np.array([1.0, 1.1])
    tspan = np.linspace(0, 20, 201) if grid else [0, 20]
    alg = Rosenbrock.from_scheme(scheme, legacy_compat=True)
    integ = init(model, tspan, y0, alg=alg, opt=Opt(scheme=scheme, rtol=1e-6, atol=1e-8))
    st = integ.stats
    assert (st.nfeval, st.nJeval) == (model.nF, model.nJ)
    assert model.nF > 1 and model.nJ > 0
    n_daeic, nJ_daeic = model.nF, model.nJ
    sol = integ.solve()
    assert sol.stats is st and st.ret == 'success'
    assert (st.nfeval, st.nJeval) == (model.nF, model.nJ)
    s = alg.tableau.s
    attempts = st.nstep + st.nreject
    # F0, dF/dt and s - 1 stages on the first attempt of a step, the stages on
    # a retry, and for Rodas3 on a grid the end residual of every step that
    # holds a node
    extra = model.nF - n_daeic - (st.nstep * (s + 1) + st.nreject * (s - 1))
    if scheme == 'rodas3' and grid:
        assert 0 < extra <= st.nstep
    else:
        assert extra == 0
    # one Jacobian per step, kept on its retries
    assert model.nJ - nJ_daeic == st.nstep
    assert (st.ndecomp, st.nsolve) == (attempts, s * attempts)
    if not grid:
        assert st.nstep == sol.T.size - 1


@pytest.mark.i4
def test_progress_bar_and_profile_change_nothing(capsys):
    dae, y0 = models.build('dae_test')
    alg = Rodas4(legacy_compat=True)
    ref = alg(dae, [0, 20], y0, Opt(hinit=0.1))
    capsys.readouterr()
    for kwargs in (dict(pbar=True), dict(profile=True)):
        sol = alg(dae, [0, 20], y0, Opt(hinit=0.1, **kwargs))
        assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y)
    out = capsys.readouterr().out
    assert out.startswith('Time elapsed: ') and out.count('\n') == 1
    sol = solve(dae, [0, 20], y0, alg=alg, opt=Opt(hinit=0.1, profile=True))
    assert _byte_equal(sol.Y, ref.Y)
    assert capsys.readouterr().out.startswith('Time elapsed: ')


# -- I5: the default configuration --------------------------------------------


def _pole(t):
    return 1.0 / (t - 0.5)


_pole_jit = njit(_pole)


def _pole_model(pole):
    """``x' = -x + 1e-30 / (t - 0.5)``, hand-built, with the quotient computed
    by ``pole`` at the time converted to ``np.float64``.

    The factor ``1e-30`` keeps the solution smooth up to ``t = 0.5``, so the
    run reaches the stop time there, and the residuals that fail are those
    evaluated at ``t = 0.5`` itself.
    """
    def F(t, y, p, out=None):
        out = np.empty(1) if out is None else out
        out[0] = -y[0] + 1e-30 * pole(np.float64(t))
        return out

    return nDAE(csc_array(np.eye(1)), F, lambda t, y, p: csc_array(-np.eye(1)), {})


@pytest.mark.i5
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('compiled', [True, False], ids=['njit', 'numpy'])
def test_a_division_by_zero_at_a_stop_time_fails_the_run(capsys, compiled):
    pole = _pole_jit if compiled else _pole
    # Numba's default error model raises where NumPy returns inf
    if compiled:
        with pytest.raises(ZeroDivisionError):
            pole(np.float64(0.5))
    else:
        assert pole(np.float64(0.5)) == np.inf
    capsys.readouterr()
    integ = init(_pole_model(pole), [0, 1], np.ones(1), tstops=[0.5])
    sol = integ.solve()
    _failed(sol)
    _one_line(capsys, 'rodas4')
    assert sol.T.size >= 2 and 0.49 < sol.T[-1] <= 0.5 and sol.stats.nreject > 0
    if compiled:
        # the attempts at the pole failed on the converted exception
        assert integ._stepfail_reason.startswith('ZeroDivisionError in F: ')
    else:
        # no attempt failed; the attempts at the pole were rejected on their error
        assert integ._stepfail_reason is None


@pytest.mark.i5
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
def test_opt_none_is_Opt_and_alg_none_is_Rodas4(grid):
    dae, y0 = models.build('dae_test')
    tspan = np.linspace(0, 20, 201) if grid else [0, 20]
    ref = solve(dae, tspan, y0, alg=Rodas4(), opt=Opt())
    assert ref.stats.ret == 'success'
    for sol in (solve(dae, tspan, y0), solve(dae, tspan, y0, alg=None, opt=None), init(dae, tspan, y0).solve(),
                Rodas4()(dae, tspan, y0)):
        assert sol.stats.ret == 'success' and sol.stats.scheme == 'rodas4'
        assert _byte_equal(sol.T, ref.T) and _byte_equal(sol.Y, ref.Y)


def _scheme_warnings(record):
    return [w for w in record if w.category is UserWarning and str(w.message).startswith('opt.scheme=')]


def _here():
    """The line after the caller's current one."""
    return sys._getframe(1).f_lineno + 1


@pytest.mark.i5
def test_a_scheme_other_than_the_algorithm_warns_once_at_the_callers_line():
    dae, y0 = models.build('dae_test')
    tspan = [0, 1]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        lines = [_here()]
        solve(dae, tspan, y0, alg=Rodas4(), opt=Opt(scheme='rodas3'))
        lines.append(_here())
        init(dae, tspan, y0, alg=Rodas4(), opt=Opt(scheme='rodas3')).solve()
        lines.append(_here())
        Rodas4()(dae, tspan, y0, Opt(scheme='rodas3'))
    found = _scheme_warnings(record)
    assert len(found) == 3
    for w, line in zip(found, lines):
        assert str(w.message) == ("opt.scheme='rodas3' is ignored; Rodas4() integrates with 'rodas4'. Pass "
                                  "the algorithm of the method, for example Rosenbrock.from_scheme(opt.scheme).")
        assert Path(w.filename).resolve() == Path(__file__).resolve() and w.lineno == line

    # Opt() sets scheme='rodas4' whether or not the caller chose it, and a
    # scheme that names the algorithm's own method is no disagreement
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        for alg, opt in ((Rodas4(), Opt()), (Rodas4(), None), (Rodas3(), Opt(scheme='rodas3'))):
            solve(dae, tspan, y0, alg=alg, opt=opt)
            init(dae, tspan, y0, alg=alg, opt=opt).solve()
            alg(dae, tspan, y0, opt)
    assert _scheme_warnings(record) == []
