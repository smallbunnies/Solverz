"""The defaults of ``Algorithm``, the style check, ``StepFailure`` and the
members of ``StepContext`` that read the Integrator."""
from types import SimpleNamespace

import numpy as np
import pytest

from Solverz.integrator import Algorithm, StepFailure
from Solverz.integrator.algorithm import StepContext, check_style


def test_step_failure_is_its_own_type():
    # The core catches RuntimeError, ArithmeticError and ValueError elsewhere,
    # each with its own meaning, so a StepFailure must be none of them.
    assert issubclass(StepFailure, Exception)
    for other in (RuntimeError, ArithmeticError, ValueError):
        assert not issubclass(StepFailure, other)


def test_traits():
    a = Algorithm()
    assert (a.scheme, a.order, a.error_order, a.interp_order) == (None, None, None, 1)
    assert (a.adaptive, a.explicit, a.inplace, a.legacy_compat) == (False, False, False, False)
    assert a.norm == 'rms'


def test_default_interpolant_is_linear_and_reads_only_the_step():
    rng = np.random.default_rng(0)
    uprev, u_step = rng.standard_normal(5), rng.standard_normal(5)
    integ = SimpleNamespace(uprev=uprev, u_step=u_step)
    out = np.full(5, np.nan)
    for theta in (0.0, 0.3, 1.0):
        assert Algorithm().interpolant(integ, None, theta, out) is None
        assert out.tobytes() == (uprev + theta * (u_step - uprev)).tobytes()


def test_default_hooks():
    a = Algorithm()
    c1, c2 = a.alloc(None), a.alloc(None)
    assert isinstance(c1, SimpleNamespace) and c1 is not c2 and vars(c1) == {}
    assert a.addsteps(None, c1) is None
    assert a.reset_history(None, c1) is None


def test_default_initial_dt():
    opts = SimpleNamespace(dt0=None, t0=0.25, tend=2.0)
    assert Algorithm().initial_dt(SimpleNamespace(opts=opts)) == 1e-6 * (2.0 - 0.25)
    opts = SimpleNamespace(dt0=0.1, t0=0.25, tend=2.0)
    assert Algorithm().initial_dt(SimpleNamespace(opts=opts)) == 0.1


class _Formula(Algorithm):
    def perform_step(self, s):
        return s.y0


class _InPlace(Algorithm):
    inplace = True

    def perform_step(self, integ, cache):
        pass


class _FormulaOfInPlace(_InPlace):
    inplace = False

    def perform_step(self, s):
        return s.y0


class _WrongInPlace(Algorithm):
    inplace = True

    def perform_step(self, s):
        return s.y0


def test_style_check():
    for alg in (_Formula(), _InPlace(), _FormulaOfInPlace()):
        check_style(alg)
        check_style(alg)
    with pytest.raises(TypeError, match='does not define perform_step'):
        check_style(Algorithm())
    for _ in range(2):
        with pytest.raises(TypeError, match=r'takes 1 parameters after self, but inplace = True needs 2'):
            check_style(_WrongInPlace())
    alg = _Formula()
    alg.inplace = True
    with pytest.raises(TypeError, match='needs 2'):
        check_style(alg)
    check_style(_Formula())


SERVICES = ('F', 'f', 'J', 'dFdt', 'W', 'implicit', 'error_norm')


def test_step_context_reads_the_integrator():
    uprev = np.array([1.0, 2.0])
    opts = SimpleNamespace(rtol=1e-3, atol=1e-6, adaptive=True)
    services = {name: (lambda *args, name=name: name) for name in SERVICES}
    integ = SimpleNamespace(n=2, uprev=uprev, t=0.0, dt=0.1, new_step=True,
                            M='M', p={'k': 1}, opts=opts, cache='cache', **services)
    s = StepContext(integ)
    # the services are the Integrator's own, bound once
    assert all(getattr(s, name) is services[name] for name in SERVICES)
    assert (s.n, s.t, s.h, s.new_step, s.M, s.p) == (2, 0.0, 0.1, True, 'M', {'k': 1})
    assert (s.rtol, s.atol, s.adaptive, s.cache) == (1e-3, 1e-6, True, 'cache')
    integ.t, integ.dt, integ.new_step, integ.M = 0.1, 0.05, False, 'M2'
    assert (s.t, s.h, s.new_step, s.M) == (0.1, 0.05, False, 'M2')
    # the modification protocol rebinds p, so every member reads the Integrator again
    integ.p, integ.cache = {'k': 2}, 'cache2'
    integ.opts = SimpleNamespace(rtol=1e-5, atol=1e-9, adaptive=False)
    assert (s.p, s.cache) == ({'k': 2}, 'cache2')
    assert (s.rtol, s.atol, s.adaptive) == (1e-5, 1e-9, False)
    assert np.shares_memory(s.y0, uprev) and not s.y0.flags.writeable
    uprev[0] = 3.0
    assert s.y0[0] == 3.0
    with pytest.raises(ValueError):
        s.y0[0] = 4.0
    with pytest.raises(AttributeError):
        s.extra = 1
