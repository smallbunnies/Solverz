"""``legacy_run`` reproduces whole legacy Rodas trajectories bit for bit.

The transcription in ``legacy_rodas.py`` is the reference against which the
integrator's Rosenbrock step is compared attempt by attempt, so it must first
be shown to be legacy Rodas itself: ``T`` and ``Y`` byte-equal to those of
``Rodas`` for the four schemes, both kinds of ``tspan``, sparse and dense
Jacobians, rendered and inline models, a non-autonomous residual, a run with
many rejected attempts, and both backends. Each call gets its own ``Opt``,
since Rodas writes ``hmax`` and ``facmax`` into the one it receives; the two
``Opt`` objects must end in the same state as well.

The recorded attempts are checked against the same run, field by field,
since the lockstep of I3 consumes every field: the accepted ones reproduce
the saved states, the saved times and the step counts of ``Stats``; every
attempt starts from the time and the state the last accepted one ended in;
the first attempt of a step carries the Jacobian at its start, and a retry
the same object; ``trace`` records an accepted error below the floor of
``1e-6``, which only an error taken before the floor can show.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import issparse

from Solverz.solvers import klu_backend
from Solverz.solvers.daesolver.rodas.param import Rodas_param
from Solverz.solvers.daesolver.rodas.rodas import Rodas
from Solverz.solvers.klu_backend import KLU_AVAILABLE, klu_decomposition
from Solverz.solvers.laesolver import linsolver
from Solverz.solvers.option import Opt

from tests.integrator.legacy_rodas import legacy_iteration_matrix, legacy_run

pytestmark = pytest.mark.i2

SCHEMES = ['rodas3', 'rodas4', 'rodasp', 'rodas5p']

# name, variant, tspan of two entries, Opt arguments
CASES = [
    ('dae_test', 'inline_sparse', [0, 20], dict(hinit=0.1)),
    ('dae_test', 'inline_dense', [0, 20], dict(hinit=0.1)),
    ('dae_test', 'rendered', [0, 20], dict(hinit=0.1)),
    ('forced', 'inline_sparse', [0, 1], dict(rtol=1e-6, atol=1e-8)),
    ('forced', 'inline_dense', [0, 1], dict(rtol=1e-6, atol=1e-8)),
    ('trace', 'inline_sparse', [0, 1], dict(rtol=1e-6, atol=1e-8)),
    ('trace', 'inline_dense', [0, 1], dict(rtol=1e-6, atol=1e-8)),
    ('trace', 'rendered', [0, 1], dict(rtol=1e-6, atol=1e-8)),
    ('vdp', 'inline_sparse', [0, 20], dict(rtol=1e-6, atol=1e-9)),
]


def _byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _same_matrix(A, B):
    if issparse(A) or issparse(B):
        return (type(A) is type(B) and A.shape == B.shape and _byte_equal(A.data, B.data)
                and _byte_equal(A.indices, B.indices) and _byte_equal(A.indptr, B.indptr))
    return type(A) is type(B) and _byte_equal(np.asarray(A), np.asarray(B))


def _span(tspan, grid):
    return np.linspace(tspan[0], tspan[-1], 201) if grid else tspan


def _check(dae, y0, tspan, kwargs):
    """Run legacy Rodas and the transcription on the same problem and compare."""
    ref_opt = Opt(**kwargs)
    sol = Rodas(dae, tspan, y0.copy(), ref_opt)
    opt = Opt(**kwargs)
    attempts = []
    T, Y = legacy_run(dae, tspan, y0.copy(), opt, attempts)

    assert sol.stats.ret is None, 'the legacy run failed'
    assert _byte_equal(T, sol.T)
    assert _byte_equal(Y, sol.Y)
    assert vars(opt) == vars(ref_opt)

    accepted = [a for a in attempts if a.err_raw <= 1.0]
    rejected = [a for a in attempts if not a.err_raw <= 1.0]
    assert len(accepted) == sol.stats.nstep
    assert len(rejected) == sol.stats.nreject
    if len(tspan) == 2:
        assert len(accepted) == len(T) - 1
        for k, a in enumerate(accepted):
            assert _byte_equal(a.ynew, Y[k + 1])
            # the recorded dt is the one after the stretch to tend
            assert T[k + 1] == a.t + a.dt
    assert _byte_equal(attempts[0].y0, Y[0])
    assert attempts[0].t == T[0] and type(attempts[0].t) is type(np.asarray(tspan)[0])
    t, y, first = attempts[0].t, attempts[0].y0, None
    for a in attempts:
        assert type(a.t) is type(t) and a.t == t
        assert _byte_equal(a.y0, y)
        if a.reject == 0:
            first = a
            assert _same_matrix(a.J, dae.J(a.t, a.y0, dae.p))
        else:
            assert a.J is first.J
        if a.err_raw <= 1.0:
            t, y = a.t + a.dt, a.ynew
    return attempts


@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('name, variant, tspan, kwargs', CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_transcription_is_legacy_rodas(model, backend, name, variant, tspan, kwargs, scheme, grid):
    dae, y0 = model(name, variant)
    attempts = _check(dae, y0, _span(tspan, grid), dict(kwargs, scheme=scheme))
    if name == 'vdp':
        assert any(a.reject > 0 for a in attempts), 'no rejected attempt on vdp'
    if name == 'trace':
        assert any(a.err_raw < 1e-6 for a in attempts), 'no error below the floor on trace'


def _first_matching(dae, attempt, scheme):
    """The row permutation of the KLU analysis of the iteration matrix of
    ``attempt``, under the current matching setting."""
    Wm, _ = legacy_iteration_matrix(dae.M, attempt.J, attempt.dt, Rodas_param(scheme).gamma,
                                    SimpleNamespace(rscale_to_dense=None))
    return klu_decomposition(Wm).symbolic.perm


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('name, args, tspan, kwargs', [
    ('permuted', (), [0, 20], dict(hinit=0.1)),
    ('ladder', (40,), [0, 2], dict(rtol=1e-6, atol=1e-8)),
], ids=['permuted', 'ladder40'])
def test_transcription_under_the_row_matching(model, klu_matching_low, name, args, tspan, kwargs,
                                              scheme, grid):
    dae, y0 = model(name, 'inline_sparse', *args)
    with linsolver('klu'):
        attempts = _check(dae, y0, _span(tspan, grid), dict(kwargs, scheme=scheme))
        perm = _first_matching(dae, attempts[0], scheme)
    assert perm is not None, 'the analysis holds no row matching'
    if name == 'permuted':
        assert not np.array_equal(perm, np.arange(perm.size)), 'the row matching is the identity'


@pytest.mark.skipif(not KLU_AVAILABLE, reason='libklu is not available')
@pytest.mark.parametrize('grid', [False, True], ids=['span', 'grid'])
def test_transcription_on_rendered_alloc_1000(model, grid):
    dae, y0 = model('alloc', 'rendered', 1000)
    with linsolver('klu'):
        attempts = _check(dae, y0, _span([0.0, 1.0], grid),
                          dict(scheme='rodas4', rtol=1e-6, atol=1e-8, hmax=1e-2))
        perm = _first_matching(dae, attempts[0], 'rodas4')
    if klu_backend.klu_matching_enabled() and y0.size >= klu_backend.MATCHING_MIN_N:
        assert perm is not None, 'the analysis holds no row matching'
