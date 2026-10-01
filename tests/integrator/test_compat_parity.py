"""D3b: the legacy-compatible configuration reproduces legacy Rodas trajectories.

``RodasX(legacy_compat=True)(dae, tspan, y0, opt)`` must give ``T`` and ``Y``
byte-equal to legacy ``Rodas(dae, tspan, y0, opt)`` on event-free runs, over
the matrix of ``test_legacy_transcription.py``: the four schemes, both kinds
of ``tspan``, sparse and dense Jacobians, rendered and inline models, a
non-autonomous residual, a run with many rejected attempts, both backends,
the value-dependent row matching of KLU, and a model of 1000 unknowns; and
beyond that matrix an inconsistent start, which ``DaeIc`` completes at
entry, a grid that does not increase, and the controller options. Each
call gets its own ``Opt``, since legacy writes ``hmax`` and ``facmax`` into
the one it receives. The step counts agree too, since both runs take the
same attempts; the residual counts differ by design and are not compared.
Where legacy never ends, at a fixed step that does not divide the span, the
configuration stretches its last step to ``tend``.

The negative control runs two calls on one model from different initial
states and compares each with its own legacy call, so that anything one call
leaves behind for the next shows as a difference.
"""
import numpy as np
import pytest

from Solverz.integrator import Rosenbrock, init
from Solverz.solvers.daesolver.rodas.rodas import Rodas
from Solverz.solvers.klu_backend import KLU_AVAILABLE
from Solverz.solvers.laesolver import linsolver
from Solverz.solvers.option import Opt

from tests.integrator import models
from tests.integrator.test_legacy_transcription import CASES, SCHEMES, _span

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


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('scheme', SCHEMES)
def test_consecutive_calls_under_the_row_matching(klu_matching_low, scheme):
    """The row matching of a KLU analysis depends on the values of the matrix
    that triggered it, so an analysis carried from one call into the next
    would factorize with a permutation chosen for other values."""
    dae, y0 = models.build('permuted')
    kwargs = dict(scheme=scheme, rtol=1e-6, atol=1e-8)
    with linsolver('klu'):
        for y in (y0, _second_state(), y0):
            _parity(dae, y, np.linspace(0, 20, 201), kwargs)


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
