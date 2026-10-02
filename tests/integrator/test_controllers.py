"""The step-size controllers, against values computed by hand.

``IController`` and ``PIController`` return the inverse growth factor ``q``
and propose ``dt / q``; after a rejection neither grows the step until the
next acceptance. ``LegacyRodasController`` evaluates the expressions of
legacy Rodas with its ``facmax`` state. Each case includes ``EEst`` of 0,
``inf`` and ``NaN``, and the bounds of the proposed step.

Every ``rodas.py:N`` refers to legacy Rodas at commit ``056e87a``, before its
deprecation warning moved its lines.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from Solverz.integrator import (Algorithm, Controller, IController, LegacyRodasController, PIController,
                                Rodas4, Rodas5P)
from Solverz.integrator.options import IntegratorOptions
from Solverz.solvers.option import Opt

pytestmark = pytest.mark.i4


def _opts(**kwargs):
    base = dict(safety=0.9, qmin=0.2, qmax=6, qmax_init=6, t0=0.0, dtmax=1.0)
    base.update(kwargs)
    return SimpleNamespace(**base)


def _alg(order, error_order=None):
    return SimpleNamespace(order=order, error_order=error_order)


def _attempt(ctrl, EEst, dt):
    """``(q, accepted)`` of one attempt with the error ``EEst`` and the step ``dt``."""
    integ = SimpleNamespace(EEst=EEst, dt=dt)
    return ctrl.stepsize(integ), ctrl.accepts(integ), integ


def test_the_protocol():
    c = Controller(_opts(), _alg(1))
    integ = SimpleNamespace(EEst=1.0, dt=0.1)
    assert c.accepts(integ) and not c.accepts(SimpleNamespace(EEst=1.0000001, dt=0.1))
    for call in (lambda: c.stepsize(integ), lambda: c.on_accept(integ, 1.0), lambda: c.on_reject(integ, 1.0)):
        with pytest.raises(NotImplementedError):
            call()


def test_the_controller_an_algorithm_selects():
    opts = IntegratorOptions.from_opt(Opt(), Rodas4(), [0, 1])
    assert type(Rodas4().controller(opts)) is IController
    legacy_opts = IntegratorOptions.from_opt(Opt(), Rodas4(legacy_compat=True), [0, 1])
    assert type(Rodas4(legacy_compat=True).controller(legacy_opts)) is LegacyRodasController

    class Toy(Algorithm):
        order = 2

    assert type(Toy().controller(opts)) is IController
    # error_order defaults to order
    assert Toy().controller(opts).expo == 1 / 2
    assert IController(_opts(), _alg(2, 3)).expo == 1 / 3


def test_icontroller():
    c = IController(_opts(), _alg(4))
    assert (c.safety, c.qmin, c.qmax, c.qmax_now, c.expo) == (0.9, 0.2, 6, 6, 0.25)

    # a zero error grows by qmax
    q, ok, integ = _attempt(c, 0.0, 0.1)
    assert q == 1 / 6 and ok
    assert c.on_accept(integ, q) == 0.1 / (1 / 6)
    # an ordinary accepted error
    E = np.float64(0.5)
    q, ok, integ = _attempt(c, E, 0.1)
    assert q == 0.5 ** 0.25 / 0.9 and ok
    assert c.on_accept(integ, q) == 0.1 / (0.5 ** 0.25 / 0.9)
    # a tiny error is capped at the growth factor qmax
    q, ok, _ = _attempt(c, 1e-20, 0.1)
    assert q == 1 / 6 and ok
    # a large error is capped at the reduction factor qmin
    q, ok, integ = _attempt(c, 1e8, 0.1)
    assert q == 5.0 and not ok
    c.on_reject(integ, q)
    assert integ.dt == 0.1 / 5.0 and c.qmax_now == 1.0


def test_icontroller_on_non_finite_errors():
    for E in (np.inf, np.nan):
        c = IController(_opts(), _alg(4))
        q, ok, integ = _attempt(c, np.float64(E), 0.1)
        # inf is capped at 1/qmin, and a NaN never wins a comparison
        assert q == 5.0 and not ok
        c.on_reject(integ, q)
        assert integ.dt == 0.1 / 5.0


def test_icontroller_keeps_the_cap_after_a_rejection_until_an_acceptance():
    c = IController(_opts(), _alg(4))
    q, ok, integ = _attempt(c, 4.0, 0.1)
    assert not ok
    c.on_reject(integ, q)
    assert integ.dt == 0.1 / (4.0 ** 0.25 / 0.9) and c.qmax_now == 1.0
    # a second rejection, and then an accepted tiny error: no growth
    q, ok, integ = _attempt(c, 2.0, 0.05)
    assert not ok
    c.on_reject(integ, q)
    assert c.qmax_now == 1.0
    q, ok, integ = _attempt(c, 1e-12, 0.04)
    assert ok and q == 1.0
    assert c.on_accept(integ, q) == 0.04
    # the acceptance restores the cap qmax
    assert c.qmax_now == 6
    q, ok, integ = _attempt(c, 1e-12, 0.04)
    assert q == 1 / 6 and c.on_accept(integ, q) == 0.04 / (1 / 6)


def test_icontroller_steady_band():
    c = IController(_opts(safety=1.0), _alg(4))
    q, ok, integ = _attempt(c, 1.0, 0.1)
    assert q == 1.0 and ok
    assert c.on_accept(integ, q) == 0.1
    # outside the band of [1, 1] the step changes
    q, ok, integ = _attempt(c, 0.9, 0.1)
    assert q < 1.0 and c.on_accept(integ, q) == 0.1 / q


def test_picontroller():
    c = PIController(_opts(), _alg(4))
    assert (c.beta1, c.beta2) == (7 / 40, 2 / 20)
    assert (c.q11, c.errold, c.qmax_now) == (1.0, 1e-4, 6)

    q, ok, integ = _attempt(c, 0.0, 0.1)
    assert q == 1 / 6 and ok and c.q11 == 1.0

    E = np.float64(0.5)
    q, ok, integ = _attempt(c, E, 0.1)
    expected = max(1 / 6, min(5.0, 0.5 ** (7 / 40) / 1e-4 ** (2 / 20) / 0.9))
    assert c.q11 == 0.5 ** (7 / 40) and q == expected and ok
    assert c.on_accept(integ, q) == 0.1 / q
    assert c.errold == 0.5 and c.qmax_now == 6

    # the next error is weighed against the last accepted one
    q, ok, integ = _attempt(c, np.float64(0.25), 0.1)
    assert q == max(1 / 6, min(5.0, 0.25 ** (7 / 40) / 0.5 ** (2 / 20) / 0.9))
    c.on_accept(integ, q)
    # an accepted error below qoldinit is remembered as qoldinit
    q, ok, integ = _attempt(c, np.float64(1e-9), 0.1)
    c.on_accept(integ, q)
    assert c.errold == 1e-4


def test_picontroller_rejection():
    for E in (4.0, np.inf, np.nan):
        c = PIController(_opts(), _alg(4))
        q, ok, integ = _attempt(c, np.float64(E), 0.1)
        assert not ok
        c.on_reject(integ, q)
        q11 = np.float64(E) ** (7 / 40)
        assert integ.dt == 0.1 / min(5.0, q11 / 0.9)
        assert c.qmax_now == 1.0
    # no growth after a rejection until an acceptance
    q, ok, integ = _attempt(c, 1e-12, 0.05)
    assert ok and q == 1.0 and c.on_accept(integ, q) == 0.05 and c.qmax_now == 6


def test_picontroller_steady_band():
    c = PIController(_opts(safety=1.0), _alg(4))
    c.errold = 1.0
    q, ok, integ = _attempt(c, 1.0, 0.1)
    assert q == 1.0 and c.on_accept(integ, q) == 0.1


def _legacy_rodas4_opts(**kwargs):
    return IntegratorOptions.from_opt(Opt(**kwargs), Rodas4(legacy_compat=True), [0, 20])


def test_legacy_controller_parameters():
    opts = _legacy_rodas4_opts(f_savety=0.8, fac1=0.3, fac2=5, facmax=4, hmax=0.5)
    c = LegacyRodasController(opts, Rodas4(legacy_compat=True))
    assert (c.safety, c.fac1, c.fac2, c.facmax, c.pord, c.hmax) == (0.8, 0.3, 5, 4, 4, 0.5)
    # hmin from t0 in its tspan dtype, np.int64 for [0, 20]
    assert c.hmin == 16 * np.spacing(np.int64(0)) and c.hmin > 0
    opts = _legacy_rodas4_opts()
    c = LegacyRodasController(opts, Rodas5P(legacy_compat=True))
    assert c.pord == 5 and c.hmax == 20 and c.facmax == 6


def test_legacy_controller_is_the_legacy_rule():
    opts = _legacy_rodas4_opts()
    c = LegacyRodasController(opts, Rodas4(legacy_compat=True))
    hmin, hmax = c.hmin, c.hmax

    # rodas.py:213-216 with facmax = opt.facmax = 6
    E = np.float64(0.5)
    fac, ok, integ = _attempt(c, E, 0.1)
    ref = np.minimum(6, np.maximum(0.2, 0.9 / (np.maximum(E, 1.0e-6) ** (1 / 4))))
    assert ok and type(fac) is np.float64 and fac == ref and c.dtnew == 0.1 * ref
    new = c.on_accept(integ, fac)
    assert new == np.min([hmax, np.max([hmin, 0.1 * ref])]) and c.facmax == 6

    # the floor of 1e-6: a zero error grows by facmax
    fac, ok, integ = _attempt(c, np.float64(0.0), 0.1)
    assert ok and fac == 6 and c.dtnew == 0.1 * 6

    # a rejection sets facmax = 1 until the next acceptance
    fac, ok, integ = _attempt(c, np.float64(16.0), 0.1)
    assert not ok and fac == np.maximum(0.2, 0.9 / 16.0 ** 0.25)
    c.on_reject(integ, fac)
    assert integ.dt == 0.1 * fac and c.facmax == 1
    fac, ok, integ = _attempt(c, np.float64(1e-12), 0.05)
    assert ok and fac == 1 and c.dtnew == 0.05
    c.on_accept(integ, fac)
    assert c.facmax == 6

    # inf and the 1e6 of a non-finite state reject by fac1; NaN rejects and
    # gives a NaN step, as in legacy
    for E in (np.inf, np.float64(1.0e6)):
        fac, ok, integ = _attempt(c, np.float64(E), 0.1)
        assert not ok and fac == 0.2
    fac, ok, integ = _attempt(c, np.float64(np.nan), 0.1)
    assert not ok and np.isnan(fac)
    c.on_reject(integ, fac)
    assert np.isnan(integ.dt)


def test_legacy_controller_bounds():
    opts = _legacy_rodas4_opts(hmax=0.15)
    c = LegacyRodasController(opts, Rodas4(legacy_compat=True))
    fac, ok, integ = _attempt(c, np.float64(1e-12), 0.1)
    assert c.dtnew == 0.1 * 6 and c.on_accept(integ, fac) == 0.15
    # below hmin the step is raised to hmin
    fac, ok, integ = _attempt(c, np.float64(1e3), 1e-322)
    assert c.dtnew < c.hmin
    c.on_reject(integ, fac)
    assert integ.dt == c.hmin
