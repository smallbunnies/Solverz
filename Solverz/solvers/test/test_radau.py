"""
Tests for the Radau IIA(5) DAE solver
(``Solverz.solvers.daesolver.radau``).

The reference is Solverz's own Rodas integrator at a tighter tolerance,
so the comparison stays inside the same DAE infrastructure (no scipy
translation, no mass-matrix-to-explicit-ODE conversion).
"""

import numpy as np
import pytest

from Solverz import (
    Eqn, Ode, Var, Opt, Rodas, Radau, Model, made_numerical, TimeVars,
)


def _vdp_model(mu=10.0):
    m = Model()
    m.x = Var('x', [2.0, 0.0])
    m.f1 = Ode('f1', m.x[1], m.x[0])
    m.f2 = Ode('f2', mu * (1 - m.x[0] ** 2) * m.x[1] - m.x[0], m.x[1])
    vdp, y0 = m.create_instance()
    return made_numerical(vdp, y0, sparse=True), y0


def _index1_dae():
    """Solverz docs' canonical index-1 DAE (test_dae.py)."""
    m = Model()
    m.x = Var('x', 1)
    m.y = Var('y', 1)
    m.f = Ode(name='f', f=-m.x ** 3 + 0.5 * m.y ** 2, diff_var=m.x)
    m.g = Eqn(name='g', eqn=m.x ** 2 + m.y ** 2 - 2)
    dae, y0 = m.create_instance()
    return made_numerical(dae, y0, sparse=True), y0


def test_radau_vdp_matches_rodas():
    """Mildly-stiff Van der Pol: Radau5 ≈ Rodas3 at t = 20."""
    ndae, y0 = _vdp_model(mu=10.0)
    sol_radau = Radau(ndae, [0.0, 20.0], y0, Opt(rtol=1e-6, atol=1e-9))
    sol_ref = Rodas(ndae, [0.0, 20.0], y0,
                    Opt(rtol=1e-8, atol=1e-10, scheme='rodas3'))
    np.testing.assert_allclose(sol_radau.Y[-1],
                               sol_ref.Y[-1],
                               rtol=1e-5,
                               atol=1e-6)


def test_radau_dense_output():
    """Dense-output mode uses the Radau collocation interpolant."""
    ndae, y0 = _index1_dae()
    tspan = np.linspace(0.0, 20.0, 201)
    sol_radau = Radau(ndae, tspan, y0,
                      Opt(rtol=1e-6, atol=1e-9, hinit=0.1))
    sol_ref = Rodas(ndae, tspan, y0,
                    Opt(rtol=1e-8, atol=1e-10))
    np.testing.assert_allclose(sol_radau.T, tspan, atol=1e-12)
    np.testing.assert_allclose(sol_radau.Y.array,
                               sol_ref.Y.array,
                               rtol=1e-4,
                               atol=1e-4)


def test_radau_index1_dae():
    """Stiff-accurate behaviour on the index-1 DAE from test_dae.py."""
    ndae, y0 = _index1_dae()
    sol_radau = Radau(ndae, [0.0, 20.0], y0,
                      Opt(rtol=1e-6, atol=1e-9, hinit=0.1))
    sol_ref = Rodas(ndae, [0.0, 20.0], y0,
                    Opt(rtol=1e-8, atol=1e-10))
    np.testing.assert_allclose(sol_radau.Y[-1],
                               sol_ref.Y[-1],
                               rtol=1e-5,
                               atol=1e-6)


def test_radau_event_bounceball():
    """Event detection: bouncing ball, mirrors test_rodas_event.py."""
    m = Model()
    m.x = Var('x', [0, 20])
    m.f1 = Ode('f1', m.x[1], m.x[0])
    m.f2 = Ode('f2', -9.8, m.x[1])
    bball, y0 = m.create_instance()
    nbball = made_numerical(bball, y0, sparse=True)

    def events(t, y):
        value = np.array([y[0]])
        isterminal = np.array([1])
        direction = np.array([-1])
        return value, isterminal, direction

    opt = Opt(event=events, rtol=1e-6, atol=1e-9)
    teout = np.array([])
    tstart = 0.0
    tend = 30.0
    for _ in range(10):
        sol = Radau(nbball, np.linspace(tstart, tend, 100), y0, opt)
        teout = np.concatenate([teout, sol.te])
        y0['x'][0] = 0
        y0['x'][1] = -0.9 * sol.Y[-1]['x'][1]
        tstart = sol.T[-1]

    expected = np.array([
        4.081633047365558, 7.755103268847604, 11.061226369910107,
        14.03673684129459, 16.714695668619328, 19.124858513456523,
        21.294005283692695, 23.246237159791765, 25.003245923211466,
        26.584554164220997,
    ])
    np.testing.assert_allclose(teout, expected, rtol=1e-5, atol=0)
