"""The ``dF/dt`` policies.

I2: ``'legacy'`` is legacy Rodas' ``dfdt`` bit for bit; ``'ode23s'`` is its
formula, checked against an increment the test computes itself; on
``forced``, whose ``dF/dt = [0, -cos t]`` is exact, the error that
``'ode23s'`` hands the stages, ``dt * (ft - F_t)``, stays within
``4 * SQRT_EPS`` of the magnitude of the state, where ``'legacy'`` at
``t = 0`` exceeds it on the algebraic row by more than three orders of
magnitude.

I4: within a run, ``F(t, y0)`` and ``dF/dt`` are evaluated once per step
and kept on its retries, which the residuals ``vdp`` receives show.

I5: the first step of a run on ``forced`` hands its stages the ``'ode23s'``
quotient in the default configuration, within the same bound, and the
legacy quotient in the legacy-compatible one, which exceeds it.
"""
import numpy as np
import pytest

from Solverz.integrator import Rodas4, init
from Solverz.integrator.derivative import DFDT_POLICIES, SQRT_EPS, dfdt_legacy, dfdt_ode23s
from Solverz.num_api.num_eqn import nDAE
from Solverz.solvers.daesolver.rodas.rodas import dfdt
from Solverz.solvers.option import Opt


def _byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class _Counted:
    """The residual service ``F(t, y, out=None)`` of a model, counting its calls."""

    def __init__(self, dae):
        self.dae = dae
        self.calls = 0

    def __call__(self, t, y, out=None):
        self.calls += 1
        return self.dae.F(t, y, self.dae.p, out=out)


def _apply(policy, dae, t, dt, y):
    """``(ft, number of residual calls)``, from buffers that start as ``NaN``."""
    F = _Counted(dae)
    f0 = dae.F(t, y, dae.p)
    ft = np.full(y.size, np.nan)
    scratch = np.full(y.size, np.nan)
    assert policy(F, t, dt, y, f0, ft, scratch) is ft
    return ft, F.calls


MODELS = [('forced', 'inline_sparse'),
          ('trace', 'inline_sparse'),
          ('trace', 'inline_dense'),
          ('trace', 'rendered')]


def _state(y0):
    return y0 * 1.1 + 0.05


@pytest.mark.i2
def test_the_policies_are_registered():
    assert DFDT_POLICIES == {'ode23s': dfdt_ode23s, 'legacy': dfdt_legacy}
    assert SQRT_EPS == 2.0 ** -26 and type(SQRT_EPS) is float


@pytest.mark.i2
@pytest.mark.parametrize('t', [0, np.int64(0), 0.3, 17], ids=['0', 'int64', '0.3', '17'])
@pytest.mark.parametrize('name, variant', MODELS, ids=[f"{m[0]}-{m[1]}" for m in MODELS])
def test_legacy_is_legacy_dfdt(model, name, variant, t):
    dae, y0 = model(name, variant)
    y = _state(y0)
    ft, calls = _apply(dfdt_legacy, dae, t, 0.5, y)
    assert _byte_equal(ft, np.asarray(dfdt(dae, t, y), dtype=np.float64))
    assert calls == 1


@pytest.mark.i2
@pytest.mark.parametrize('t, dt', [(0.0, 1e-3), (0.0, 5e-8), (0.3, 1e-3), (0.3, 5e-8), (0.3, 2.0),
                                   (17.0, 0.5), (0.05, 1e-6)])
@pytest.mark.parametrize('name, variant', MODELS, ids=[f"{m[0]}-{m[1]}" for m in MODELS])
def test_ode23s_follows_its_formula(model, name, variant, t, dt):
    dae, y0 = model(name, variant)
    y = _state(y0)
    ft, calls = _apply(dfdt_ode23s, dae, t, dt, y)
    delt = 2.0 ** -26 * max(abs(t), abs(t + dt))
    tdel = (t + min(delt, abs(dt))) - t
    assert 0.0 < tdel <= dt
    f0 = dae.F(t, y, dae.p)
    expected = (dae.F(t + tdel, y, dae.p) - f0) / tdel
    assert _byte_equal(ft, expected)
    assert calls == 1


@pytest.mark.i2
@pytest.mark.parametrize('name, variant', MODELS, ids=[f"{m[0]}-{m[1]}" for m in MODELS])
def test_ode23s_is_zero_when_the_increment_vanishes(model, name, variant):
    dae, y0 = model(name, variant)
    t, dt = 1e20, 1e-10
    assert t + dt == t
    ft, calls = _apply(dfdt_ode23s, dae, t, dt, _state(y0))
    assert _byte_equal(ft, np.zeros(y0.size))
    assert calls == 0


def _bound_error(policy, dae, y, t, dt):
    ft, _ = _apply(policy, dae, t, dt, y)
    exact = np.array([0.0, -np.cos(t)])
    return np.abs(dt * (ft - exact)), 4 * SQRT_EPS * max(1.0, np.max(np.abs(y)))


@pytest.mark.i2
@pytest.mark.parametrize('dt', [1e-3, 5e-8])
@pytest.mark.parametrize('t', [0.0, 0.3])
def test_ode23s_is_accurate_at_every_t_and_dt(model, t, dt):
    dae, y = model('forced')
    err, bound = _bound_error(dfdt_ode23s, dae, y, t, dt)
    assert np.all(err <= bound), (err, bound)


@pytest.mark.i2
def test_legacy_is_inaccurate_at_t0(model):
    """At ``t = 0`` the legacy increment is ``1.49e-16``, below the spacing of
    the terms of the algebraic residual, so the quotient is wrong by half."""
    dae, y = model('forced')
    err, bound = _bound_error(dfdt_legacy, dae, y, 0.0, 1e-3)
    assert err[0] <= bound
    assert err[1] > 1e3 * bound, (err, bound)
    err, bound = _bound_error(dfdt_ode23s, dae, y, 0.0, 1e-3)
    assert np.all(err <= bound)


@pytest.mark.i4
def test_F0_and_dFdt_are_evaluated_once_per_step(model):
    """Each step evaluates ``F`` at its start state twice, at ``t`` and at
    ``t + ddt``, however many attempts it takes; the first step also carries
    the residual of ``DaeIc``. Every other residual is a stage residual."""
    dae, y0 = model('vdp')
    calls = []

    def F(t, y, p, out=None):
        calls.append((t, y.copy()))
        return dae.F(t, y, p, out=out)

    alg = Rodas4(legacy_compat=True)
    sol = alg(nDAE(dae.M, F, dae.J, dae.p), [0, 20], y0, Opt(rtol=1e-6, atol=1e-9))
    st = sol.stats
    assert st.ret == 'success' and st.nreject > 0
    s = alg.tableau.s
    assert st.nfeval == len(calls) == 1 + st.nstep * (s + 1) + st.nreject * (s - 1)
    starts = 0
    for k in range(sol.T.size - 1):
        t = sol.T[k]
        at_start = sorted(tc for tc, y in calls if _byte_equal(y, sol.Y[k]))
        assert len(at_start) == (3 if k == 0 else 2), k
        tscale = np.maximum(0.1 * np.abs(t), 1e-8)
        ddt = t + np.sqrt(np.spacing(1)) * tscale - t
        assert at_start[0] == t and at_start[-1] == t + ddt
        starts += len(at_start)
    assert starts == 1 + 2 * st.nstep


@pytest.mark.i5
@pytest.mark.parametrize('legacy_compat', [False, True], ids=['default', 'legacy_compat'])
def test_each_configuration_takes_its_own_policy_in_a_run(model, legacy_compat):
    """The bound of ``test_legacy_is_inaccurate_at_t0`` on the quotient that
    the first step of a run at ``t = 0`` used, with the step it was taken
    for, which the first attempt sets when it is accepted. The legacy
    quotient is wrong by half on the algebraic row, which the short first
    step of ``rtol = 1e-6`` scales to eight times the bound."""
    dae, y0 = model('forced')
    integ = init(dae, [0, 1], y0, alg=Rodas4(legacy_compat=legacy_compat), opt=Opt(rtol=1e-6, atol=1e-8))
    assert integ.step() and integ.tprev == 0.0 and integ.stats.nreject == 0
    err = np.abs(integ.dt_step * (integ.dFdt() - np.array([0.0, -1.0])))
    bound = 4 * SQRT_EPS * max(1.0, np.max(np.abs(integ.uprev)))
    if legacy_compat:
        assert err[1] > bound, (err, bound)
    else:
        assert np.all(err <= bound), (err, bound)
