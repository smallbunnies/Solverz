"""D3a: the Rosenbrock step of the integrator is legacy Rodas, attempt by attempt.

For every attempt that ``legacy_run`` records on the matrix of
``test_legacy_transcription.py``, the Integrator in the legacy-compatible
configuration is put in the state of that attempt, namely ``t`` and ``dt``
the recorded values unconverted, ``uprev`` the recorded state, ``new_step``
true on the first attempt of a step, and the recorded Jacobian, and it takes
the attempt through its dispatch. Its own ``IterationMatrix`` receives the
factorizations in legacy's order, so the KLU analysis and the SuperLU
ordering it carries evolve as legacy's do. ``u`` must be byte-equal to
``ynew`` and ``EEst`` to the raw error, rejected attempts included, and the
counters must show ``s + 1`` residuals and one Jacobian on the first attempt
of a step and ``s - 1`` residuals on a retry.

On a grid, every accepted step is then interpolated at the nodes it covers,
as the legacy-compatible saving does it, through ``addsteps`` and
``interpolant``, and every row must be byte-equal to legacy's saved row.
"""
import numpy as np
import pytest

from Solverz.integrator import Rosenbrock
from Solverz.integrator.integrator import Integrator
from Solverz.solvers.klu_backend import KLU_AVAILABLE
from Solverz.solvers.laesolver import linsolver
from Solverz.solvers.option import Opt

from tests.integrator.legacy_rodas import legacy_run
from tests.integrator.test_legacy_transcription import CASES, SCHEMES, _span

pytestmark = pytest.mark.i3


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class _Recorded:
    """The model with the Jacobian of the recorded attempt, which checks that
    it is asked for at the start of the step."""

    def __init__(self, dae):
        self.M, self.p, self.F = dae.M, dae.p, dae.F
        self.attempt = None
        self.calls = 0

    def J(self, t, y, p):
        a = self.attempt
        assert type(t) is type(a.t) and t == a.t
        assert _byte_equal(y, a.y0) and p is self.p
        self.calls += 1
        return a.J


def _save_nodes(integ, a, nodes, inext, Y, row):
    """The legacy-compatible saving of the accepted attempt ``a``: every node
    in ``(t, t + dt]``, through ``addsteps`` once and then ``interpolant``.
    Returns the index of the next node."""
    alg = integ.alg
    told, tnew = a.t, a.t + a.dt
    integ.tprev, integ.t, integ.t_step, integ.dt_step = told, tnew, tnew, a.dt
    integ.u_step = integ.u
    ready = False
    while inext < nodes.size and tnew >= nodes[inext] > told:
        if not ready:
            alg.addsteps(integ, integ.cache)
            ready = True
        row.fill(np.nan)
        assert alg.interpolant(integ, integ.cache, (nodes[inext] - told) / a.dt, row) is None
        assert _byte_equal(row, Y[inext]), f"node {inext} at t = {nodes[inext]!r}"
        inext += 1
    return inext


def _lockstep(dae, y0, tspan, kwargs):
    """Run legacy on its own ``Opt`` and replay every attempt on the Integrator."""
    attempts = []
    T, Y = legacy_run(dae, tspan, y0.copy(), Opt(**kwargs), attempts)
    model = _Recorded(dae)
    alg = Rosenbrock.from_scheme(kwargs['scheme'], legacy_compat=True)
    integ = Integrator(model, tspan, y0, alg, Opt(**kwargs))
    assert integ.cache.K.flags.c_contiguous
    s, stats = alg.tableau.s, integ.stats
    nodes = np.array(tspan)
    inext = 1
    row = np.empty(y0.size)
    for k, a in enumerate(attempts):
        integ.t, integ.dt = a.t, a.dt
        np.copyto(integ.uprev, a.y0)
        integ.new_step = a.reject == 0
        integ.force_stepfail = False
        model.attempt = a
        before = (stats.nfeval, stats.nJeval, stats.ndecomp, stats.nsolve, model.calls)
        integ.perform_step()
        assert not integ.force_stepfail, integ._stepfail_reason
        assert _byte_equal(integ.u, a.ynew), f"attempt {k}: u"
        assert type(integ.EEst) is np.float64, f"attempt {k}: EEst is {type(integ.EEst)}"
        assert _byte_equal(integ.EEst, a.err_raw), f"attempt {k}: EEst {integ.EEst!r} != {a.err_raw!r}"
        first = a.reject == 0
        after = (stats.nfeval, stats.nJeval, stats.ndecomp, stats.nsolve, model.calls)
        assert tuple(x - y for x, y in zip(after, before)) == (s + 1 if first else s - 1, int(first), 1, s,
                                                               int(first))
        if a.err_raw <= 1.0 and nodes.size > 2:
            inext = _save_nodes(integ, a, nodes, inext, Y, row)
    if nodes.size > 2:
        assert inext == nodes.size == Y.shape[0]
    return attempts


@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('name, variant, tspan, kwargs', CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_every_attempt_is_legacy(model, backend, name, variant, tspan, kwargs, scheme, grid):
    dae, y0 = model(name, variant)
    attempts = _lockstep(dae, y0, _span(tspan, grid), dict(kwargs, scheme=scheme))
    if name == 'vdp':
        assert any(a.reject > 0 for a in attempts), 'no rejected attempt on vdp'


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('name, args, tspan, kwargs', [
    ('permuted', (), [0, 20], dict(hinit=0.1)),
    ('ladder', (40,), [0, 2], dict(rtol=1e-6, atol=1e-8)),
], ids=['permuted', 'ladder40'])
def test_every_attempt_under_the_row_matching(model, klu_matching_low, name, args, tspan, kwargs,
                                              scheme, grid):
    dae, y0 = model(name, 'inline_sparse', *args)
    with linsolver('klu'):
        _lockstep(dae, y0, _span(tspan, grid), dict(kwargs, scheme=scheme))


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
def test_every_attempt_on_rendered_alloc_1000(model, grid):
    dae, y0 = model('alloc', 'rendered', 1000)
    with linsolver('klu'):
        _lockstep(dae, y0, _span([0.0, 1.0], grid), dict(scheme='rodas4', rtol=1e-6, atol=1e-8, hmax=1e-2))
