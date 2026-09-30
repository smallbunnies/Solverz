"""The ``dF/dt`` policies.

I2: ``'legacy'`` is legacy Rodas' ``dfdt`` bit for bit; ``'ode23s'`` is its
formula, checked against an increment the test computes itself; on
``forced``, whose ``dF/dt = [0, -cos t]`` is exact, the error that
``'ode23s'`` hands the stages, ``dt * (ft - F_t)``, stays within
``4 * SQRT_EPS`` of the magnitude of the state, where ``'legacy'`` at
``t = 0`` exceeds it on the algebraic row by more than three orders of
magnitude.
"""
import numpy as np
import pytest

from Solverz.integrator.derivative import DFDT_POLICIES, SQRT_EPS, dfdt_legacy, dfdt_ode23s
from Solverz.solvers.daesolver.rodas.rodas import dfdt


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
