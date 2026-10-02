"""D3b: the legacy-compatible configuration reproduces legacy Rodas trajectories.

``RodasX(legacy_compat=True)(dae, tspan, y0, opt)`` must give ``T`` and ``Y``
byte-equal to legacy ``Rodas(dae, tspan, y0, opt)`` on event-free runs, over
the matrix of ``test_legacy_transcription.py``: the four schemes, both kinds
of ``tspan``, sparse and dense Jacobians, rendered and inline models, a
non-autonomous residual, a run with many rejected attempts, both backends,
the value-dependent row matching of KLU, and a model of 1000 unknowns; and
beyond that matrix an inconsistent start, which ``DaeIc`` completes at
entry, a grid that does not increase, the controller options, a first
step above ``hmax``, and spans whose last stretch misses ``tend``. Each
call gets its own ``Opt``, since legacy writes ``hmax`` and ``facmax`` into
the one it receives. The step counts agree too, since both runs take the
same attempts; the residual counts differ by design and are not compared.
Where legacy never ends, at a fixed step that does not divide the span, the
configuration stretches its last step to ``tend``.

The negative control runs two calls on one model from different initial
states and compares each with its own legacy call, so that anything one call
leaves behind for the next shows as a difference; under the row matching of
KLU the two first analyses differ, so that a carried analysis shows too.
"""
import numpy as np
import pytest

from Solverz.integrator import Rosenbrock, init
from Solverz.solvers.daesolver.rodas.rodas import Rodas
from Solverz.solvers import klu_backend
from Solverz.solvers.klu_backend import KLU_AVAILABLE, set_klu_matching
from Solverz.solvers.laesolver import linsolver
from Solverz.solvers.option import Opt

from tests.integrator import models
from tests.integrator.legacy_rodas import legacy_run
from tests.integrator.test_legacy_transcription import CASES, SCHEMES, _first_matching, _span

pytestmark = pytest.mark.i4


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _parity(dae, y0, tspan, kwargs):
    ref = Rodas(dae, tspan, y0.copy(), Opt(**kwargs))
    assert ref.stats.ret is None, 'the legacy run failed'
    alg = Rosenbrock.from_scheme(kwargs['scheme'], legacy_compat=True)
    y = y0.copy()
    sol = alg(dae, tspan, y, Opt(**kwargs))
    assert _byte_equal(y, y0)
    assert sol.stats.ret == 'success' and sol.stats.succeed is True
    assert _byte_equal(sol.T, ref.T), f"T: {sol.T.size} rows against {ref.T.size}"
    assert _byte_equal(sol.Y, ref.Y)
    assert (sol.stats.nstep, sol.stats.nreject) == (ref.stats.nstep, ref.stats.nreject)
    assert sol.te is None and sol.ye is None and sol.ie is None
    return sol


@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('name, variant, tspan, kwargs', CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_the_compatible_configuration_is_legacy_rodas(model, backend, name, variant, tspan, kwargs,
                                                      scheme, grid):
    dae, y0 = model(name, variant)
    sol = _parity(dae, y0, _span(tspan, grid), dict(kwargs, scheme=scheme))
    if name == 'vdp':
        assert sol.stats.nreject > 0, 'no rejected attempt on vdp'


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('name, args, tspan, kwargs', [
    ('permuted', (), [0, 20], dict(hinit=0.1)),
    ('ladder', (40,), [0, 2], dict(rtol=1e-6, atol=1e-8)),
], ids=['permuted', 'ladder40'])
def test_under_the_row_matching(model, klu_matching_low, name, args, tspan, kwargs, scheme, grid):
    dae, y0 = model(name, 'inline_sparse', *args)
    with linsolver('klu'):
        _parity(dae, y0, _span(tspan, grid), dict(kwargs, scheme=scheme))


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
def test_on_rendered_alloc_1000(model, grid):
    dae, y0 = model('alloc', 'rendered', 1000)
    with linsolver('klu'):
        _parity(dae, y0, _span([0.0, 1.0], grid), dict(scheme='rodas4', rtol=1e-6, atol=1e-8, hmax=1e-2))


@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('name, variant', [('dae_test', 'inline_sparse'), ('dae_test', 'inline_dense'),
                                           ('dae_test', 'rendered'), ('permuted', 'inline_sparse')],
                         ids=['dae_test-inline_sparse', 'dae_test-inline_dense', 'dae_test-rendered',
                              'permuted-inline_sparse'])
def test_an_inconsistent_start(model, backend, name, variant, scheme, grid):
    """``DaeIc`` at entry, which the core runs on counted services, moves the
    algebraic variable onto the circle as it does in legacy Rodas, also when
    the algebraic equation is the first row of ``M``."""
    dae, _ = model(name, variant)
    y0 = np.array([0.5, 1.0])
    sol = _parity(dae, y0, _span([0, 20], grid), dict(scheme=scheme, hinit=0.1))
    # DaeIc stops on a Newton step below 1e-3 * rtol, so the residual of the
    # default rtol = 1e-3 is of the order 1e-6
    assert sol.Y[0, 1] != 1.0 and abs(sol.Y[0, 0] ** 2 + sol.Y[0, 1] ** 2 - 2) < 1e-5


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('tspan', [[0, 2, 2, 5, 20], [0, 5, 3, 8, 20], [0, 0, 10, 20]],
                         ids=['repeated', 'decreasing', 'repeated_start'])
def test_a_grid_that_does_not_increase(model, backend, tspan, scheme):
    """Legacy saves a node while ``t >= node > tprev``, so a repeated node in
    one step is saved twice and a node at or before the start of the step
    that reaches it ends the saving; the integer nodes keep their dtype."""
    dae, y0 = model('dae_test')
    _parity(dae, y0, tspan, dict(scheme=scheme, hinit=0.1))


@pytest.mark.parametrize('scheme', SCHEMES)
def test_the_controller_options_and_an_array_atol(model, backend, scheme):
    """Every field of ``Opt`` the controller reads, with ``facmax = 1`` at
    entry as an earlier legacy call leaves it after a rejection."""
    dae, y0 = model('vdp')
    sol = _parity(dae, y0, [0, 20], dict(scheme=scheme, rtol=1e-6, atol=np.array([1e-9, 1e-8]),
                                         f_savety=0.8, fac1=0.3, fac2=4, facmax=1, hmax=0.5))
    assert sol.stats.nreject > 0


@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('scheme', SCHEMES)
def test_a_first_step_above_hmax(model, backend, scheme, grid):
    """Legacy clamps the first step to ``hmax`` and nothing later does, so
    ``hinit = 0.5`` above ``hmax = 0.1`` exercises the clamp. A span from 0
    to ``1e6 * hmax`` or more, over which the default first step exceeds
    ``hmax``, would take a million steps."""
    dae, y0 = model('dae_test')
    sol = _parity(dae, y0, _span([0, 20], grid), dict(scheme=scheme, hinit=0.5, hmax=0.1))
    if not grid:
        assert sol.T[1] == 0.1, 'the first step was not clamped to hmax'


@pytest.mark.parametrize('solver, tend, last', [('klu', 6543.21, -9.09e-13), ('klu', 1618.03, 2.27e-13),
                                                 ('superlu', 6886.87, -9.09e-13), ('superlu', 7137.71, 9.09e-13)],
                         ids=['klu-overshoot', 'klu-undershoot', 'superlu-overshoot', 'superlu-undershoot'])
def test_a_last_stretch_that_misses_tend(model, solver, tend, last):
    """Legacy stretches its last step to ``dt = tend - t``, and ``t + dt`` can
    miss ``tend`` by an ulp. Its end test ``|tend - t| < spacing(1)`` then
    fails, and it takes one more step of that ulp, backwards after an
    overshoot; the compatible configuration takes the same steps. The two
    backends differ in the last bits of the steps, so each has its own
    spans."""
    if solver == 'klu' and not KLU_AVAILABLE:
        pytest.skip('libklu is not available')
    dae, y0 = model('dae_test')
    with linsolver(solver):
        sol = _parity(dae, y0, [0, tend], dict(scheme='rodas4', rtol=1e-3, atol=1e-6))
    assert sol.T[-1] == tend
    np.testing.assert_allclose(sol.T[-1] - sol.T[-2], last, rtol=0.01)


def _second_state():
    """A consistent initial state ``(x, y)`` of ``dae_test`` and ``permuted``
    other than ``x = y = 1``, on the circle ``x**2 + y**2 = 2``."""
    return np.array([0.8, np.sqrt(2.0 - 0.8 * 0.8)])


@pytest.mark.parametrize('scheme', SCHEMES)
def test_consecutive_calls_do_not_depend_on_each_other(backend, scheme):
    dae, y0 = models.build('dae_test')
    kwargs = dict(scheme=scheme, rtol=1e-6, atol=1e-8)
    for y in (y0, _second_state(), y0):
        _parity(dae, y, [0, 20], kwargs)


def _first_perm(dae, y0, tspan, kwargs):
    """The row permutation of the first KLU analysis of a legacy call."""
    attempts = []
    legacy_run(dae, tspan, y0.copy(), Opt(**kwargs), attempts)
    return _first_matching(dae, attempts[0], kwargs['scheme'])


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('scheme', SCHEMES)
def test_consecutive_calls_under_the_row_matching(klu_matching_low, scheme):
    """The row matching of a KLU analysis depends on the values of the matrix
    that triggered it, so an analysis carried from one call into the next
    would factorize with a permutation chosen for other values. With a
    first step of 100, the first iteration matrix from ``x = 1/6`` on the
    circle has the identity as its matching, and the one from ``x = y = 1``
    the anti-diagonal, so the second call fails on a carried analysis."""
    dae, y0 = models.build('permuted')
    x = 1.0 / 6.0
    states = (np.array([x, np.sqrt(2.0 - x * x)]), y0)
    kwargs = dict(scheme=scheme, hinit=100.0)
    with linsolver('klu'):
        first, second = (_first_perm(dae, y, [0, 200], kwargs) for y in states)
        assert not np.array_equal(first, second), 'the two first analyses have one row matching'
        for y in states:
            _parity(dae, y, [0, 200], kwargs)


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('scheme', SCHEMES)
def test_a_call_after_one_under_the_row_matching(scheme):
    """A call under the default matching setting, where this model of two
    unknowns gets no row matching, after one under the low threshold, whose
    analysis carries the anti-diagonal."""
    dae, y0 = models.build('permuted')
    kwargs = dict(scheme=scheme, rtol=1e-6, atol=1e-8)
    saved = (klu_backend._MATCHING, klu_backend.MATCHING_MIN_N)
    with linsolver('klu'):
        set_klu_matching(True, min_n=2)
        try:
            assert _first_perm(dae, y0, [0, 20], kwargs) is not None
            _parity(dae, y0, np.linspace(0, 20, 201), kwargs)
        finally:
            set_klu_matching(*saved)
        assert _first_perm(dae, y0, [0, 20], kwargs) is None
        _parity(dae, y0, np.linspace(0, 20, 201), kwargs)


@pytest.mark.parametrize('scheme', SCHEMES)
def test_a_fixed_step_that_does_not_divide_the_span(model, scheme):
    """Beyond parity: legacy's fixed-step form steps past ``tend`` when the
    step does not divide the span, and never ends. The compatible
    configuration stretches its last fixed step to ``tend``, so that the run
    ends there. The run is driven step by step, so that a run without the
    stretch fails after ten steps instead of running on."""
    dae, y0 = model('dae_test')
    integ = init(dae, [0, 1], y0, alg=Rosenbrock.from_scheme(scheme, legacy_compat=True),
                 opt=Opt(fix_h=True, hinit=0.3))
    for _ in range(10):
        if not integ.step():
            break
    assert integ.finished and not integ.failed
    sol = integ.postamble()
    assert sol.stats.ret == 'success'
    assert sol.T.tolist() == [0.0, 0.3, 0.6, 0.8999999999999999, 1.0]
