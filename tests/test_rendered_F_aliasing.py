"""Regression: a rendered ``F_`` must return a fresh array on every call.

``print_module_code`` used to emit one module-level ``_F_`` buffer that
``F_`` wrote into and returned, so two residuals were never valid at the
same time. Every solver that DIFFERENCES two residuals then read a zero
derivative, silently:

- ``Rodas`` takes dF/dt as ``(F(t + ddt) - F(t)) / ddt``
  (``solvers/daesolver/rodas/rodas.py``), which became exactly 0 on every
  step. A Rosenbrock method needs that term on a non-autonomous problem, so
  the method dropped from order 4 to order 1 while still converging, and the
  accepted-step count scaled as ``rtol**(-1/2)`` instead of ``rtol**(-1/5)``.
- ``Radau`` holds ``ff1``, ``ff2`` and ``ff3`` together.
- ``ode15s`` and ``adams_bdf`` pass a residual into ``numjac``, which
  differences against it.

``made_numerical`` was never affected: its printer allocates ``_F_`` inside
the function. The two paths must agree, which is what the tests below
assert. ``J_`` was never affected either: ``CooToCsc.__call__`` gathers the
values with fancy indexing, which copies.
"""
import numpy as np
import pytest

from Solverz import (Model, Var, Param, TimeSeriesParam, Ode, Eqn,
                     made_numerical, module_printer, Rodas, Opt)


def _forced_model():
    """dx/dt = -x + u(t), one state, with u a genuinely time-varying input.

    The forcing is what makes dF/dt non-zero, so a model without it could
    not detect the defect at all.
    """
    m = Model()
    m.x = Var('x', [0.0])
    m.u = TimeSeriesParam('u', v_series=[0.0, 1.0, 0.0],
                          time_series=[0.0, 0.5, 1.0],
                          index=np.arange(1), value=np.zeros(1))
    m.decay = Ode('decay', f=-m.x + m.u, diff_var=m.x)
    return m.create_instance()


@pytest.fixture(scope='module')
def rendered(tmp_path_factory):
    sdae, y0 = _forced_model()
    d = tmp_path_factory.mktemp('alias')
    module_printer(sdae, y0, 'alias_mdl', directory=str(d), jit=True).render()
    import sys
    sys.path.insert(0, str(d))
    from alias_mdl import mdl
    return (mdl, made_numerical(sdae, y0, sparse=True),
            np.array(y0.array, dtype=float))


def test_two_residuals_do_not_alias(rendered):
    """The defect itself: ``F`` called twice must give two usable values."""
    mdl, _, y = rendered
    f0 = mdl.F(0.0, y, mdl.p)
    f1 = mdl.F(0.6, y, mdl.p)
    assert f0 is not f1, '`F_` returned the same object twice'
    # The forcing differs between t = 0 and t = 0.6, so the two residuals
    # must differ. Aliasing made this difference exactly zero.
    assert np.max(np.abs(np.asarray(f1) - np.asarray(f0))) > 1e-3


def test_dFdt_is_not_zero(rendered):
    """What the aliasing cost Rodas, measured the way Rodas measures it."""
    from Solverz.solvers.daesolver.rodas.rodas import dfdt
    mdl, inline, y = rendered
    for t in (0.0, 0.25, 0.75):
        d_mdl = np.asarray(dfdt(mdl, t, y), dtype=float)
        d_inl = np.asarray(dfdt(inline, t, y), dtype=float)
        assert np.max(np.abs(d_mdl)) > 0.0, f'dF/dt vanished at t = {t}'
        np.testing.assert_allclose(d_mdl, d_inl, rtol=1e-6, atol=1e-8)


def test_rendered_matches_inline_trajectory(rendered):
    """The consequence: the two paths must integrate to the same answer."""
    mdl, inline, y = rendered
    tspan = np.linspace(0.0, 1.0, 21)
    opt = dict(rtol=1e-8, atol=1e-10, hmax=1e-2)
    a = Rodas(inline, tspan, y.copy(), Opt(**opt))
    b = Rodas(mdl, tspan, y.copy(), Opt(**opt))
    # a rendered ``nDAE`` carries no variable address, so its solution comes
    # back as a bare array rather than name-indexable
    ya = np.asarray(a.Y['x'] if hasattr(a.Y, 'var_list') else a.Y).reshape(-1)
    yb = np.asarray(b.Y).reshape(-1)
    np.testing.assert_allclose(yb, ya, rtol=1e-6, atol=1e-8)
    # Order, not just the answer: the step counts must be comparable. The
    # order-1 behaviour the defect caused showed up here as a factor of
    # hundreds.
    assert b.stats.nstep < 5 * a.stats.nstep + 20


def test_residual_sized_by_the_equations(rendered):
    """The buffer was ``zeros_like(y__)``, i.e. sized by the VARIABLE count.
    It must be sized by the equation count, which is what the inline path
    uses; the two agree only on a square system."""
    mdl, inline, y = rendered
    assert (np.asarray(mdl.F(0.0, y, mdl.p)).shape
            == np.asarray(inline.F(0.0, y, inline.p)).shape)
