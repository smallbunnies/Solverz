"""Fixtures of the integrator tests.

Every test of a gate from I2 on carries the marker of its milestone, ``i2``
to ``i8``, so that ``-m i2`` selects the gate of I2: a file of one milestone
through ``pytestmark``, a file shared by two milestones on each test. The
files of I1 predate the markers.
"""
import pytest

from Solverz.solvers import klu_backend
from Solverz.solvers.klu_backend import KLU_AVAILABLE, set_klu_matching
from Solverz.solvers.laesolver import linsolver

from tests.integrator import models

MILESTONES = ('i2', 'i3', 'i4', 'i5', 'i6a', 'i6b', 'i7a', 'i7b', 'i8')


def pytest_configure(config):
    for name in MILESTONES:
        config.addinivalue_line('markers', f"{name}: a test of the gate of milestone {name.upper()}")


@pytest.fixture(params=['klu', 'superlu'])
def backend(request):
    """The backend of the test, selected through the ``linsolver`` context
    manager; the ``'klu'`` case is skipped without libklu."""
    if request.param == 'klu' and not KLU_AVAILABLE:
        pytest.skip('libklu is not available')
    with linsolver(request.param):
        yield request.param


@pytest.fixture
def klu_matching_low():
    """The row matching of the KLU analysis from two unknowns up, so that small
    models exercise the value-dependent permutation; the previous setting is
    restored afterwards."""
    saved = (klu_backend._MATCHING, klu_backend.MATCHING_MIN_N)
    set_klu_matching(True, min_n=2)
    try:
        yield
    finally:
        set_klu_matching(*saved)


@pytest.fixture(scope='session')
def model(tmp_path_factory):
    """``model(name, variant='inline_sparse', *args) -> (dae, y0)``, built once
    per session, a rendered one compiled once; ``y0`` is a new array at every
    call.

    The ``dae`` is shared by every test of the session, so a test that changes
    the model, its parameters or its ``model_cache`` builds its own with
    ``models.build``.
    """
    built = {}

    def get(name, variant='inline_sparse', *args):
        key = (name, variant, args)
        if key not in built:
            directory = tmp_path_factory.mktemp('rendered') if variant == 'rendered' else None
            built[key] = models.build(name, variant, *args, directory=directory)
        dae, y0 = built[key]
        return dae, y0.copy()

    return get
