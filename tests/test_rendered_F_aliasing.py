"""Regression: a rendered ``F_`` must not hand back one shared array.

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

The contract now is the one of SciML's ``f!(du, u, p, t)``: the residual is
written into the caller's ``out`` array when one is given, and a fresh array
is returned when none is. ``made_numerical`` and the rendered module both
print that form, and ``nDAE`` gives any other residual the keyword by
copying. ``J_`` was never affected: ``CooToCsc.__call__`` gathers the values
with fancy indexing, which copies.
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


# ---------------------------------------------------------------------------
# The in-place contract: ``F(t, y, p, out=buf)`` writes into the caller's
# array, the form of SciML's ``f!(du, u, p, t)`` in NumPy's ``out=`` spelling.


def test_out_is_written_in_place(rendered):
    """Both printers honour ``out``: the result IS the caller's array and it
    equals the out-of-place value."""
    mdl, inline, y = rendered
    for dae in (mdl, inline):
        expected = np.asarray(dae.F(0.6, y, dae.p), dtype=float)
        buf = np.full(expected.shape, np.nan)
        r = dae.F(0.6, y, dae.p, out=buf)
        assert r is buf
        np.testing.assert_array_equal(buf, expected)


def test_legacy_F_gets_out_by_copy():
    """A residual written without ``out``, a user's lambda or a module
    rendered by an older Solverz, receives the keyword from ``nDAE``."""
    from Solverz.num_api.num_eqn import nDAE
    calls = []

    def F(t, y, p):
        calls.append(t)
        return np.array([-y[0] + t])

    dae = nDAE(np.eye(1), F, lambda t, y, p: np.array([[-1.0]]), {})
    y = np.array([0.5])
    plain = dae.F(0.3, y, dae.p)
    buf = np.empty(1)
    r = dae.F(0.3, y, dae.p, out=buf)
    assert r is buf
    np.testing.assert_array_equal(buf, plain)
    assert calls == [0.3, 0.3]


def test_kwargs_catch_all_is_not_an_out_parameter():
    """``**kwargs`` would take the keyword and ignore it, so it does not
    count as support; the adapter copies instead."""
    from Solverz.num_api.num_eqn import _accepts_out, nAE

    def F(y, p, **kwargs):
        return np.array([y[0] - 1.0])

    assert not _accepts_out(F)
    ae = nAE(F, lambda y, p: np.array([[1.0]]), {})
    buf = np.empty(1)
    assert ae.F(np.array([3.0]), ae.p, out=buf) is buf
    assert buf[0] == 2.0


def test_in_place_and_out_of_place_integrate_identically(rendered):
    """A solver may not read a buffer after overwriting it. Integrating the
    same residual once through its own ``out`` and once through a wrapper
    that drops the keyword must give the same trajectory bit for bit."""
    from Solverz.num_api.num_eqn import nDAE
    mdl, _, y = rendered
    F = mdl.F
    wrapped = nDAE(mdl.M, lambda t, y_, p: F(t, y_, p), mdl.J, mdl.p)
    tspan = np.linspace(0.0, 1.0, 11)
    opt = dict(rtol=1e-8, atol=1e-10, hmax=1e-2)
    a = Rodas(mdl, tspan, y.copy(), Opt(**opt))
    b = Rodas(wrapped, tspan, y.copy(), Opt(**opt))
    assert a.stats.nstep == b.stats.nstep
    np.testing.assert_array_equal(np.asarray(a.Y), np.asarray(b.Y))
