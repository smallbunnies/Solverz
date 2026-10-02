"""The services of a step and the two formula algorithms that ship with them.

``implicit`` solves ``M y - h*gamma*F(t, y) = rhs`` by a simplified Newton
iteration whose remaining error is below ``KAPPA`` in the weighted norm of
the run, and its slope equals ``F(t, y)`` to the same tolerance; a Newton
iteration that diverges, converges too slowly or meets a non-finite value
rejects the attempt. Its weight takes the larger of the iterate and the
start of the step, and the rate estimate of a call starts the next one until
the model changes. ``W(gamma)`` is factorized once per ``gamma`` per
attempt. ``F0``, ``J0`` and ``dFdt()`` are evaluated once per step and kept
on its retries. ``D`` marks the algebraic rows, a stored zero of ``M``
included, and is rebuilt when the model changes. ``f`` is ``M^-1 F`` on a
model whose ``M`` is a scaled permutation, follows a change of ``M``, and
raises on a model with algebraic equations. ``out=`` gives the bytes of the
out-of-place call and, for ``F``, ``f`` and a KLU solve, with or without a
row matching, allocates no array. Every evaluation is counted.

``ImplicitEuler`` and ``Trapezoid`` converge with their orders, and ``D``
keeps an algebraic residual that the consistent initialization left in
place out of their error estimates; without it a tight ``atol`` fails the
run.
"""
import tracemalloc

import numpy as np
import pytest
from scipy.sparse import csc_array, diags_array, issparse

from Solverz import Eqn, Model, Ode, Param, Var, made_numerical
from Solverz.integrator import Algorithm, ImplicitEuler, Trapezoid, solve
from Solverz.integrator.integrator import Integrator
from Solverz.integrator.nlsolve import KAPPA
from Solverz.num_api.num_eqn import nDAE
from Solverz.solvers import klu_backend
from Solverz.solvers.klu_backend import KLU_AVAILABLE, klu_decomposition, set_klu_matching
from Solverz.solvers.laesolver import linsolver
from Solverz.solvers.option import Opt

pytestmark = pytest.mark.i7a


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class _Formula(Algorithm):
    """A formula algorithm whose step is ``body(s)``."""

    scheme = 'formula'
    order = 1
    error_order = 1
    adaptive = True

    def __init__(self, body):
        self.body = body

    def perform_step(self, s):
        return self.body(s)


def _zero_error(y, s):
    return y, np.zeros(s.n)


def _attempt(integ, t, dt, new_step=True):
    """One attempt of the algorithm at ``(t, dt)`` from ``integ.uprev``."""
    integ.t, integ.dt = t, dt
    integ.new_step = new_step
    integ.force_stepfail = False
    integ.perform_step()


def _weighted(integ, v, y):
    """The RMS norm of ``v`` in the weights of the Newton iteration at ``y``."""
    opts = integ.opts
    w = opts.atol + opts.rtol * np.maximum(np.abs(y), np.abs(integ.uprev))
    return float(np.sqrt(np.mean(np.square(v / w))))


def _dense(A):
    return A.toarray() if issparse(A) else np.asarray(A)


def _solved(dae, t, hgamma, rhs, y):
    """The solution of ``M y - hgamma F(t, y) = rhs`` by full Newton to rounding."""
    M = _dense(dae.M)
    y = y.copy()
    for _ in range(50):
        G = M @ y - hgamma * dae.F(t, y, dae.p) - rhs
        dy = np.linalg.solve(M - hgamma * _dense(dae.J(t, y, dae.p)), G)
        y -= dy
        if np.max(np.abs(dy)) <= 1e-15 * (1.0 + np.max(np.abs(y))):
            return y
    raise AssertionError('the reference Newton iteration did not converge')


class _Counted:
    """A residual ``F(t, y, p, out=None)`` that counts its calls."""

    def __init__(self, F):
        self.F = F
        self.calls = 0

    def __call__(self, t, y, p, out=None):
        self.calls += 1
        return self.F(t, y, p, out=out)


def _counters(integ):
    s = integ.stats
    return s.nfeval, s.nJeval, s.ndecomp, s.nsolve


# -- implicit -------------------------------------------------------------------


@pytest.mark.parametrize('name, dt', [('vdp', 0.05), ('dae_test', 0.2)])
def test_implicit_solves_its_equation_to_the_newton_tolerance(model, name, dt):
    dae, y0 = model(name)
    got = {}

    def body(s):
        rhs = s.M @ s.y0 + 0.5 * s.h * (s.D * s.F0)
        eta = integ._nl_eta
        y = s.implicit(s.t + s.h, 0.5, rhs)
        got['eta'] = integ._nl_eta
        # the same iteration, from the same rate estimate, with the slope
        integ._nl_eta = eta
        got['y'], got['k'] = s.implicit(s.t + s.h, 0.5, rhs, slope=True)
        got['rhs'], got['t'] = rhs, s.t + s.h
        return _zero_error(y, s)

    integ = Integrator(dae, [0, 1], y0, _Formula(body), Opt(rtol=1e-6, atol=1e-8))
    _attempt(integ, 0.0, dt)
    assert not integ.force_stepfail
    y, k, rhs, t = got['y'], got['k'], got['rhs'], got['t']
    assert _byte_equal(integ.u, y)
    hgamma = dt * 0.5
    exact = _solved(dae, t, hgamma, rhs, integ.uprev)
    error = _weighted(integ, y - exact, y)
    # the Newton step that would follow, which bounds the remaining error
    W = integ.W(0.5)
    residual = _dense(dae.M) @ y - hgamma * dae.F(t, y, dae.p) - rhs
    next_step = _weighted(integ, W.solve(residual), y)
    slope = _weighted(integ, W.solve(hgamma * (k - dae.F(t, y, dae.p))), y)
    print(f"{name}: error {error:.3e}, next Newton step {next_step:.3e}, slope {slope:.3e}, "
          f"eta {got['eta']:.3e}")
    assert error <= KAPPA and next_step <= KAPPA and slope <= KAPPA
    # the slope is (M y - rhs) / (h gamma), with no residual of its own
    assert _byte_equal(k, (dae.M @ y - rhs) / hgamma)


def _scalar(F, J):
    return nDAE(csc_array(np.ones((1, 1))), F, J, {})


def _linear_F(a):
    def F(t, y, p, out=None):
        out = np.empty(1) if out is None else out
        np.multiply(y, -a, out=out)
        return out
    return F


def _constant_J(b):
    def J(t, y, p):
        return csc_array(np.full((1, 1), -b))
    return J


@pytest.mark.parametrize('a, b, reason', [
    # W = 2 against 1 + h a = 1: every iteration halves the error, too slowly for ten iterations
    (0.0, 1.0, 'the Newton iteration converges too slowly'),
    # W = 1 against 1 + h a = 3: the error doubles in size and alternates in sign
    (2.0, 0.0, 'the Newton iteration diverged'),
    (np.nan, 1.0, 'the Newton iteration produced a non-finite value')])
def test_a_failed_newton_iteration_rejects_the_attempt(a, b, reason):
    def body(s):
        return _zero_error(s.implicit(s.t + s.h, 1.0, 2.0 * s.y0), s)

    F = _Counted(_linear_F(a))
    integ = Integrator(_scalar(F, _constant_J(b)), [0, 1], np.ones(1), _Formula(body), Opt())
    eta = integ._nl_eta
    _attempt(integ, 0.0, 1.0)
    assert integ.force_stepfail and integ._stepfail_reason == reason
    # a failure leaves the rate estimate of the last successful call
    assert integ._nl_eta == eta
    assert F.calls - 1 == integ.stats.nsolve <= 10


def test_an_exact_newton_matrix_converges_at_the_second_residual():
    def body(s):
        return _zero_error(s.implicit(s.t + s.h, 1.0, 2.0 * s.y0), s)

    F = _Counted(_linear_F(3.0))
    integ = Integrator(_scalar(F, _constant_J(3.0)), [0, 1], np.ones(1), _Formula(body), Opt())
    calls = F.calls
    _attempt(integ, 0.0, 0.5)
    assert not integ.force_stepfail
    np.testing.assert_allclose(integ.u, [2.0 / 2.5], rtol=1e-15)
    assert F.calls - calls == integ.stats.nsolve == 2


def test_the_rate_estimate_of_a_call_starts_the_next_one():
    """``W = 2`` against ``1 + h a = 1.5`` contracts the Newton error by
    ``theta = 1/4`` per iteration towards ``y = 4/3``, so a converged call
    leaves ``eta = theta/(1 - theta)`` near 1/3. A second call from ``4/3 +
    2.8e-5``, whose first correction is 0.016 in the weighted norm, stops
    after one residual with that estimate, ``(1/3)**0.8 * 0.016 < KAPPA``,
    and needs two from the estimate 1 of a fresh Integrator. The
    modification protocol resets it."""
    got = {}

    def body(s):
        rhs = 2.0 * s.y0
        y = s.implicit(s.t + s.h, 1.0, rhs)
        got['eta'] = integ._nl_eta
        near = np.array([4.0 / 3.0 + 2.8e-5])
        for name in ('carried', 'reset'):
            if name == 'reset':
                integ._nl_eta = 1.0
            calls = F.calls
            s.implicit(s.t + s.h, 1.0, rhs, y=near)
            got[name] = F.calls - calls
        return _zero_error(y, s)

    F = _Counted(_linear_F(0.5))
    integ = Integrator(_scalar(F, _constant_J(1.0)), [0, 1], np.ones(1), _Formula(body), Opt())
    _attempt(integ, 0.0, 1.0)
    assert not integ.force_stepfail
    assert abs(got['eta'] - 1 / 3) <= 0.02
    assert (got['carried'], got['reset']) == (1, 2)
    assert integ._nl_eta != 1.0
    integ.model_modified()
    assert integ._nl_eta == 1.0


def test_the_newton_weight_takes_the_larger_of_the_iterate_and_the_start():
    """From ``uprev = 100`` towards ``y = 1``, the weight ``atol + rtol
    max(|y|, |uprev|)`` is about 0.1, so a first correction of ``7.5e-5`` is
    ``7.5e-4`` in the weighted norm and ends the iteration at one residual;
    a weight of ``|y|`` alone would make it 0.075 and take two."""
    got = {}

    def body(s):
        calls = F.calls
        got['y'] = s.implicit(s.t + s.h, 1.0, np.array([1.5]), y=np.array([1.0 + 1e-4]))
        got['calls'] = F.calls - calls
        return _zero_error(got['y'], s)

    F = _Counted(_linear_F(0.5))
    integ = Integrator(_scalar(F, _constant_J(1.0)), [0, 1], np.array([100.0]), _Formula(body), Opt())
    _attempt(integ, 0.0, 1.0)
    assert not integ.force_stepfail and got['calls'] == 1
    np.testing.assert_allclose(got['y'], [1.0 + 2.5e-5], rtol=1e-14)


# -- W, F0, J0 and dF/dt ----------------------------------------------------------


def test_W_is_factorized_once_per_gamma_per_attempt(model):
    dae, y0 = model('vdp')
    seen = []

    def body(s):
        W1, W2, W3 = s.W(0.5), s.W(0.5), s.W(1.0)
        y = s.implicit(s.t + s.h, 0.5, s.M @ s.y0)
        y = s.implicit(s.t + s.h, 0.5, s.M @ s.y0, y=y)
        seen.append((W1, W2, W3, s.W(0.5), s.W(1.0), s.h))
        return _zero_error(y, s)

    integ = Integrator(dae, [0, 1], y0, _Formula(body), Opt())
    n = _counters(integ)
    _attempt(integ, 0.0, 0.01)
    W1, W2, W3, W4, W5, h = seen[-1]
    assert W1 is W2 is W4 and W3 is W5 and W3 is not W1
    assert (W1.dtgamma, W3.dtgamma) == (h * 0.5, h * 1.0)
    assert integ.stats.ndecomp == n[2] + 2 and integ.stats.nJeval == n[1] + 1
    # a retry factorizes anew, with its own step, from the same Jacobian
    _attempt(integ, 0.0, 0.005, new_step=False)
    V1, _, V3, _, _, h = seen[-1]
    assert V1 is not W1 and V3 is not W3 and V1.dtgamma == 0.005 * 0.5
    assert integ.stats.ndecomp == n[2] + 4 and integ.stats.nJeval == n[1] + 1


def test_the_start_of_a_step_is_evaluated_once_and_kept_on_its_retries(model):
    dae, y0 = model('forced')
    seen = []

    def body(s):
        F0, J0, ft = s.F0, s.J0, s.dFdt()
        assert s.F0 is F0 and s.J0 is J0 and s.dFdt() is ft
        out = np.full(s.n, np.nan)
        assert s.dFdt(out=out) is out and _byte_equal(out, ft)
        seen.append((s.t, s.h, s.new_step, s.y0.copy(), F0.copy(), J0, ft.copy(), _counters(integ)))
        assert not (F0.flags.writeable or ft.flags.writeable or s.y0.flags.writeable)
        return _zero_error(s.y0 + 0.0, s)

    integ = Integrator(dae, [0, 1], y0, _Formula(body), Opt())
    n = _counters(integ)
    _attempt(integ, 0.3, 0.01)
    t, h, new, y, F0, J0, ft, c = seen[-1]
    assert (t, h, new) == (0.3, 0.01, True)
    assert _byte_equal(F0, dae.F(0.3, integ.uprev, dae.p))
    # F0 and the probe of dF/dt, and one Jacobian
    assert c[:2] == (n[0] + 2, n[1] + 1)
    for dt in (0.005, 0.0025):
        _attempt(integ, 0.3, dt, new_step=False)
        t2, h2, new2, y2, F02, J02, ft2, c2 = seen[-1]
        assert (t2, h2, new2) == (0.3, dt, False) and _byte_equal(y2, y)
        assert _byte_equal(F02, F0) and J02 is J0 and _byte_equal(ft2, ft) and c2[:2] == c[:2]
    # a new step evaluates them at its own start
    np.copyto(integ.uprev, integ.u + 0.25)
    _attempt(integ, 0.4, 0.01)
    t3, _, new3, _, F03, J03, ft3, c3 = seen[-1]
    assert (t3, new3) == (0.4, True) and J03 is not J0 and c3[:2] == (c[0] + 2, c[1] + 1)
    assert _byte_equal(F03, dae.F(0.4, integ.uprev, dae.p))
    np.testing.assert_allclose(ft3, [0.0, -np.cos(0.4)], rtol=1e-6)
    # J evaluates at every call
    J = integ.ctx.J
    J(0.4, integ.uprev)
    J(0.4, integ.uprev)
    assert integ.stats.nJeval == c3[1] + 2


# -- D ----------------------------------------------------------------------------


def _three_rows(M):
    """Three variables; row 1 of ``M`` is algebraic, and ``F`` is consistent at ones."""
    def F(t, y, p, out=None):
        out = np.empty(3) if out is None else out
        out[0] = -y[0]
        out[1] = y[2] - y[0]
        out[2] = -y[1]
        return out

    def J(t, y, p):
        return csc_array(np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 0.0]]))

    return nDAE(M, F, J, {})


def _stored_zero_M():
    """Row 0 holds 1.0, row 1 a stored zero, row 2 holds -3.0 in column 1."""
    M = csc_array((np.array([1.0, 0.0, -3.0]), (np.array([0, 1, 2]), np.array([0, 1, 1]))), shape=(3, 3))
    assert M.nnz == 3
    return M


def _D_of(dae, y0):
    got = []

    def body(s):
        got.append(s.D)
        return _zero_error(s.y0 + 0.0, s)

    integ = Integrator(dae, [0, 1], y0, _Formula(body), Opt())
    _attempt(integ, 0.0, 0.01)
    return integ, got[-1]


@pytest.mark.parametrize('dense', [False, True])
def test_D_marks_the_algebraic_rows_a_stored_zero_included(dense):
    M = _stored_zero_M()
    integ, D = _D_of(_three_rows(M.toarray() if dense else M), np.ones(3))
    assert _byte_equal(D, np.array([1.0, 0.0, 1.0])) and not D.flags.writeable
    assert integ.D() is D and integ.ctx.D is D


def test_D_of_the_models(model):
    for name, expected in (('dae_test', [1.0, 0.0]), ('permuted', [0.0, 1.0]), ('vdp', [1.0, 1.0])):
        dae, y0 = model(name)
        _, D = _D_of(dae, y0)
        assert _byte_equal(D, np.array(expected)), name


def test_D_is_rebuilt_when_the_model_changes():
    M = _stored_zero_M()
    integ, D = _D_of(_three_rows(M), np.ones(3))
    M.data[np.flatnonzero(M.data == 0.0)] = 2.0
    # the cache follows model_epoch, which only the modification protocol advances
    assert integ.D() is D
    integ.model_modified()
    assert _byte_equal(integ.ctx.D, np.ones(3))


# -- f ----------------------------------------------------------------------------


def _model_E():
    """``x' = -x + k s``, ``s' = c``, ``c' = -s``, declared as ``c'``, ``x'``,
    ``s'``, so that ``M`` is a permutation."""
    m = Model()
    m.x = Var('x', 0.0)
    m.s = Var('s', 0.0)
    m.c = Var('c', 1.0)
    m.k = Param('k', 1.0)
    m.fc = Ode('fc', -m.s, m.c)
    m.fx = Ode('fx', -m.x + m.k * m.s, m.x)
    m.fs = Ode('fs', m.c, m.s)
    sdae, y0 = m.create_instance()
    return made_numerical(sdae, y0, sparse=True), np.array(y0.array, dtype=np.float64)


def _f_of(dae, y0, t, y):
    got = []

    def body(s):
        got.append(s.f(t, y))
        return _zero_error(s.y0 + 0.0, s)

    integ = Integrator(dae, [0, 1], y0, _Formula(body), Opt())
    n = integ.stats.nfeval
    _attempt(integ, 0.0, 0.01)
    assert integ.stats.nfeval == n + 1
    return got[-1]


def test_f_is_the_derivative_on_a_permuted_mass_matrix():
    dae, y0 = _model_E()
    M = dae.M.toarray()
    assert not np.array_equal(M, np.eye(3)) and np.array_equal(np.sort(M, axis=None), [0] * 6 + [1] * 3)
    y = np.array([0.3, -0.2, 0.7])
    f = _f_of(dae, y0, 0.0, y)
    F = dae.F(0.0, y, dae.p)
    rows, cols = np.nonzero(M)
    expected = np.empty(3)
    expected[cols] = F[rows] / M[rows, cols]
    assert _byte_equal(f, expected)
    np.testing.assert_allclose(f, [-y[0] + y[1], y[2], -y[1]], rtol=1e-15)


def test_f_divides_by_the_entries_of_a_scaled_permutation():
    M = csc_array(np.array([[0.0, 2.0], [-0.5, 0.0]]))

    def F(t, y, p, out=None):
        out = np.empty(2) if out is None else out
        out[0] = y[0] + t
        out[1] = y[1] ** 2
        return out

    def J(t, y, p):
        return csc_array(np.array([[1.0, 0.0], [0.0, 2 * y[1]]]))

    dae = nDAE(M, F, J, {})
    y = np.array([0.3, -1.7])
    f = _f_of(dae, np.ones(2), 0.25, y)
    r = F(0.25, y, None)
    assert _byte_equal(f, np.array([r[1] / -0.5, r[0] / 2.0]))
    np.testing.assert_allclose(f, np.linalg.solve(M.toarray(), r), rtol=1e-15)


def test_the_pairing_of_f_is_rebuilt_when_the_model_changes():
    """``f`` divides by the entries of ``M`` that the modification protocol
    read last, and raises once a row of ``M`` holds only a stored zero."""
    M = csc_array(np.array([[0.0, 2.0], [-0.5, 0.0]]))

    def F(t, y, p, out=None):
        out = np.empty(2) if out is None else out
        out[0] = y[0] + t
        out[1] = y[1] ** 2 - 1.0
        return out

    def J(t, y, p):
        return csc_array(np.array([[1.0, 0.0], [0.0, 2 * y[1]]]))

    integ = Integrator(nDAE(M, F, J, {}), [0, 1], np.array([0.3, 1.0]), _Formula(lambda s: _zero_error(s.y0, s)),
                       Opt())
    y = np.array([0.3, -1.7])
    r = F(0.25, y, None)
    assert _byte_equal(integ.ctx.f(0.25, y), np.array([r[1] / -0.5, r[0] / 2.0]))
    # column 1 holds the entry of row 0, the second value of M.data
    M.data[1] = 4.0
    integ.model_modified()
    assert _byte_equal(integ.ctx.f(0.25, y), np.array([r[1] / -0.5, r[0] / 4.0]))
    # row 1 becomes 0 = y[1]**2 - 1, which the state satisfies
    M.data[0] = 0.0
    integ.model_modified()
    assert not integ.failed
    with pytest.raises(TypeError, match='f = M\\^-1 F needs a mass matrix that pairs every row'):
        integ.ctx.f(0.25, y)


def test_f_raises_on_a_model_with_algebraic_equations(model):
    dae, y0 = model('dae_test')

    def body(s):
        return _zero_error(s.f(s.t, s.y0), s)

    integ = Integrator(dae, [0, 1], y0, _Formula(body), Opt())
    n = integ.stats.nfeval
    with pytest.raises(TypeError, match='formula: f = M\\^-1 F needs a mass matrix that pairs every row'):
        _attempt(integ, 0.0, 0.01)
    assert integ.stats.nfeval == n


class _ExplicitEuler(Algorithm):
    scheme = 'explicit_euler'
    order = 1
    explicit = True

    def perform_step(self, s):
        return s.y0 + s.h * s.f(s.t, s.y0)


def test_an_explicit_algorithm_integrates_a_permuted_model():
    dae, y0 = _model_E()
    errors = []
    for k in (7, 8):
        sol = solve(dae, [0, 1], y0, alg=_ExplicitEuler(), opt=Opt(hinit=2.0 ** -k))
        assert sol.stats.ret == 'success' and sol.T[-1] == 1.0 and sol.T.size == 2 ** k + 1
        errors.append(abs(sol.Y[-1, 0] - 0.5 * (np.exp(-1.0) + np.sin(1.0) - np.cos(1.0))))
    assert 1.8 < errors[0] / errors[1] < 2.2


# -- out= and allocations ---------------------------------------------------------


N_BIG = 20000


def _big():
    """A diagonal ODE of ``N_BIG`` variables whose residual allocates nothing."""
    M = diags_array(np.linspace(1.0, 2.0, N_BIG), format='csc')
    Jc = diags_array(np.full(N_BIG, -1.0), format='csc')

    def F(t, y, p, out=None):
        out = np.empty(N_BIG) if out is None else out
        np.negative(y, out=out)
        return out

    def J(t, y, p):
        return Jc

    return nDAE(M, F, J, {}), np.linspace(0.5, 1.5, N_BIG)


def _peak(fn):
    """The peak of the memory traced while ``fn()`` runs, above the memory before it."""
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        fn()
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


def test_F_and_f_into_out_allocate_no_array():
    dae, y0 = _big()
    integ = Integrator(dae, [0, 1], y0, _Formula(lambda s: None), Opt())
    s, y, out = integ.ctx, y0 * 1.5, np.empty(N_BIG)
    size = 8 * N_BIG
    for service in (s.F, s.f):
        assert _byte_equal(service(0.5, y, out=out), service(0.5, y))
        # the control: an array of the size of the state is seen
        assert _peak(lambda: service(0.5, y)) >= size
        peak = _peak(lambda: service(0.5, y, out=out))
        print(f"{service.__name__}: peak {peak} bytes into out, against {size} bytes of a state")
        assert peak < size // 4


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('matching', [True, False], ids=['matching', 'plain'])
def test_a_klu_solve_into_out_allocates_no_array(matching):
    """Under KLU a solve into ``out`` allocates no array, also when the
    analysis holds a row matching, through which the right-hand side is
    gathered into ``out``; ``np.take`` in its default mode ``'raise'``
    buffers the whole vector there."""
    saved = (klu_backend._MATCHING, klu_backend.MATCHING_MIN_N)
    set_klu_matching(matching, min_n=2)
    try:
        dae, y0 = _big()
        with linsolver('klu'):
            integ = Integrator(dae, [0, 1], y0, _Formula(lambda s: None), Opt())
        W = integ.W(0.5)
        # the guard: the analysis holds a matching exactly when it is asked for
        assert isinstance(W.lu, klu_decomposition) and (W.lu.symbolic.perm is not None) == matching
        b, out = y0 * 1.5, np.empty(N_BIG)
        size = 8 * N_BIG
        assert _byte_equal(W.solve(b, out=out), W.solve(b))
        # the control: the solve into a new array allocates the result
        assert _peak(lambda: W.solve(b)) >= size
        peak = _peak(lambda: W.solve(b, out=out))
        print(f"matching {matching}: peak {peak} bytes into out, against {size} bytes of a state")
        assert peak < size // 4
    finally:
        set_klu_matching(*saved)


def test_out_gives_the_bytes_of_the_out_of_place_call(model):
    dae, y0 = model('vdp')
    checked = []

    def body(s):
        t1, rhs, b = s.t + s.h, s.M @ s.y0, s.F0 + 1.0
        out, out2 = np.full(s.n, np.nan), np.full(s.n, np.nan)
        assert s.F(t1, s.y0, out=out) is out and _byte_equal(out, s.F(t1, s.y0))
        assert s.F(t1, s.y0) is not s.F(t1, s.y0)
        W = s.W(0.5)
        assert W.solve(b, out=out) is out and _byte_equal(out, W.solve(b))
        eta = integ._nl_eta
        y = s.implicit(t1, 0.5, rhs)
        integ._nl_eta = eta
        assert s.implicit(t1, 0.5, rhs, out=out) is out and _byte_equal(out, y)
        integ._nl_eta = eta
        y, k = s.implicit(t1, 0.5, rhs, slope=True)
        integ._nl_eta = eta
        res = s.implicit(t1, 0.5, rhs, out=(out, out2), slope=True)
        assert res[0] is out and res[1] is out2 and _byte_equal(out, y) and _byte_equal(out2, k)
        # the result is a copy, not the iteration's buffer
        assert not np.shares_memory(y, integ._nl_y) and not np.shares_memory(k, integ._nl_y)
        checked.append(True)
        return _zero_error(y, s)

    integ = Integrator(dae, [0, 1], y0, _Formula(body), Opt())
    _attempt(integ, 0.0, 0.01)
    assert checked and not integ.force_stepfail


def test_out_of_f_on_a_permuted_model():
    dae, y0 = _model_E()
    integ = Integrator(dae, [0, 1], y0, _ExplicitEuler(), Opt(hinit=0.1))
    y, out = np.array([0.3, -0.2, 0.7]), np.full(3, np.nan)
    assert integ.ctx.f(0.1, y, out=out) is out and _byte_equal(out, integ.ctx.f(0.1, y))


# -- the counters -----------------------------------------------------------------


def test_the_services_count_every_evaluation(model):
    dae0, y0 = model('dae_test')
    F = _Counted(dae0.F)
    dae = nDAE(dae0.M, F, dae0.J, dae0.p)
    record = {}

    def body(s):
        c0, f0 = _counters(integ), F.calls
        y = s.implicit(s.t + s.h, 1.0, s.M @ s.y0)
        c1, f1 = _counters(integ), F.calls
        y, k = s.implicit(s.t + s.h, 1.0, s.M @ s.y0, y=y, slope=True)
        c2, f2 = _counters(integ), F.calls
        s.F(s.t, s.y0)
        c3 = _counters(integ)
        record.update(c=(c0, c1, c2, c3), f=(f0, f1, f2))
        return y, np.zeros(s.n)

    integ = Integrator(dae, [0, 1], y0, _Formula(body), Opt())
    _attempt(integ, 0.0, 0.1)
    (c0, c1, c2, c3), (f0, f1, f2) = record['c'], record['f']
    # the first call: one Jacobian and one factorization, and per iteration one residual and one solve
    iterations = f1 - f0
    assert iterations >= 1 and c1 == (c0[0] + iterations, c0[1] + 1, c0[2] + 1, c0[3] + iterations)
    # the second call reuses both; its slope costs no residual
    iterations = f2 - f1
    assert iterations >= 1 and c2 == (c1[0] + iterations, c1[1], c1[2], c1[3] + iterations)
    assert c3 == (c2[0] + 1,) + c2[1:]


# -- ImplicitEuler and Trapezoid ----------------------------------------------------


def _model_A(z0=0.0):
    """``x' = -x + z``, ``0 = z - k s``, ``s' = c``, ``c' = -s`` from ``x = s = 0``,
    ``c = 1`` and ``z = z0``; ``x(t) = (exp(-t) + sin t - cos t)/2``."""
    m = Model()
    m.x = Var('x', 0.0)
    m.z = Var('z', z0)
    m.s = Var('s', 0.0)
    m.c = Var('c', 1.0)
    m.k = Param('k', 1.0)
    m.fx = Ode('fx', -m.x + m.z, m.x)
    m.gz = Eqn('gz', m.z - m.k * m.s)
    m.fs = Ode('fs', m.c, m.s)
    m.fc = Ode('fc', -m.s, m.c)
    sdae, y0 = m.create_instance()
    return made_numerical(sdae, y0, sparse=True), np.array(y0.array, dtype=np.float64)


def _x_exact(t):
    return 0.5 * (np.exp(-t) + np.sin(t) - np.cos(t))


@pytest.mark.parametrize('method', [ImplicitEuler, Trapezoid])
def test_the_formula_algorithms_converge_with_their_order(method):
    """The error is the largest over the steps. The first-order error
    coefficient of ``ImplicitEuler`` passes near zero at ``t = 1``, 0.017
    against 0.125 at ``t = 0.5``, so the error at ``t = 1`` alone is ruled by
    its ``h**2`` term on these steps and its slopes read 3.8, -1.4 and 0.5."""
    dae, y0 = _model_A()
    errors = []
    for k in range(3, 7):
        h = 2.0 ** -k
        sol = solve(dae, [0, 1], y0, alg=method(), opt=Opt(fix_h=True, hinit=h, rtol=1e-12, atol=1e-14))
        assert sol.stats.ret == 'success' and sol.T[-1] == 1.0 and sol.stats.nstep == 2 ** k
        errors.append(np.max(np.abs(sol.Y[:, 0] - _x_exact(sol.T))))
    slopes = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print(f"{method.scheme}: slopes {slopes}")
    assert np.mean(slopes[-2:]) >= method.order - 0.3


def _without_D(method):
    """``method`` with ``F(t0, y0)`` in place of ``D F(t0, y0)``."""
    class NoD(method):
        scheme = method.scheme + '_without_D'

        def perform_step(self, s):
            if method is ImplicitEuler:
                y = s.implicit(s.t + s.h, 1.0, s.M @ s.y0)
                return y, s.W(1.0).solve(0.5 * (s.M @ (y - s.y0) - s.h * s.F0))
            y = s.implicit(s.t + s.h, 0.5, s.M @ s.y0 + 0.5 * s.h * s.F0)
            return y, s.W(0.5).solve(s.M @ (y - s.y0) - s.h * s.F0)

    return NoD


@pytest.mark.parametrize('method', [ImplicitEuler, Trapezoid])
def test_a_residual_left_by_the_initialization_is_not_an_error(method, capsys):
    """``z = 1e-7`` leaves the algebraic residual ``1e-7``, below the threshold
    of ``DaeIc``, in place. ``D`` keeps it out of the error estimate; without
    ``D`` it enters the estimate at every step size, and at ``atol = 1e-10`` the
    run fails."""
    dae, y0 = _model_A(z0=1e-7)
    opt = Opt(rtol=1e-6, atol=1e-10)
    sol = solve(dae, [0, 1], y0, alg=method(), opt=opt)
    assert sol.Y[0, 1] == 1e-7
    assert sol.stats.ret == 'success' and sol.T[-1] == 1.0
    assert abs(sol.Y[-1, 0] - _x_exact(1.0)) < 1e-3
    capsys.readouterr()
    control = solve(dae, [0, 1], y0, alg=_without_D(method)(), opt=opt)
    assert control.stats.ret == 'failed' and control.T[-1] < 0.5
    assert capsys.readouterr().out.count('\n') == 1


@pytest.mark.parametrize('method', [ImplicitEuler, Trapezoid])
def test_the_formula_algorithms_on_a_stiff_model(model, method):
    """Van der Pol with ``mu = 10``, whose error test rejects attempts,
    against Rodas4 at tight tolerances."""
    dae, y0 = model('vdp')
    sol = solve(dae, [0, 5], y0, alg=method(), opt=Opt(rtol=1e-5, atol=1e-7))
    ref = solve(dae, [0, 5], y0, opt=Opt(rtol=1e-10, atol=1e-12))
    print(f"{method.scheme}: {sol.stats.nstep} steps, {sol.stats.nreject} rejected, "
          f"deviation {np.max(np.abs(sol.Y[-1] - ref.Y[-1])):.3e}")
    assert sol.stats.ret == 'success' and sol.T[-1] == 5.0 and sol.stats.nreject > 0
    np.testing.assert_allclose(sol.Y[-1], ref.Y[-1], rtol=0.05, atol=0.05)
