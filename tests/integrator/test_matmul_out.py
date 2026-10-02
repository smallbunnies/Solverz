"""``np.matmul(K, v, out=buf)`` gives the bits of ``K @ v``.

The Rosenbrock step writes its stage products into buffers that it keeps
across attempts, so a buffer can hold ``NaN`` or ``inf`` from a failed
attempt when the next product is written into it. A BLAS that implements
``beta = 0`` in ``y = alpha A x + beta y`` by scaling ``y`` would carry
``0 * NaN`` into the result. The products are those of the step: ``K`` is
``(n, s)`` and C-contiguous, and ``v`` is a column of a transposed table of
``Rodas_param``, such as ``alpha[:, j]``, or a contiguous vector such as
``dt * b``.
"""
import numpy as np
import pytest

from Solverz.solvers.daesolver.rodas.param import Rodas_param


def _byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _stage_matrices(n, s, rng):
    """``K`` as the step holds it: full, and zero from column ``j`` on as
    after stage ``j - 1``; the full one has a zero row and signed zeros, so
    that the sign of a zero result is compared too."""
    full = np.zeros((n, s))
    full[...] = rng.standard_normal((n, s)) * 10.0 ** rng.integers(-8, 4, size=(n, s))
    full[0, :] = 0.0
    full[-1, ::2] = -0.0
    yield full
    for j in range(1, s):
        K = np.zeros((n, s))
        K[:, :j] = full[:, :j]
        yield K


def _vectors(tab, dt):
    for j in range(tab.s):
        yield tab.alpha[:, j]
        yield tab.gammatilde[:, j]
    yield np.multiply(tab.b, dt)
    yield np.multiply(tab.bd, dt)
    yield -np.multiply(tab.b, dt)


@pytest.mark.parametrize('fill', [np.nan, np.inf], ids=['nan', 'inf'])
@pytest.mark.parametrize('scheme', ['rodas3', 'rodas4', 'rodas5p'])
@pytest.mark.parametrize('n', [2, 28, 1000, 10000])
def test_matmul_into_a_dirty_buffer_is_byte_equal(n, scheme, fill):
    tab = Rodas_param(scheme)
    s = tab.s
    assert s == {'rodas3': 4, 'rodas4': 6, 'rodas5p': 8}[scheme]
    assert tab.alpha[:, 1].base is not None
    rng = np.random.default_rng(n * 100 + s)
    buf = np.empty(n)
    clean = np.empty(n)
    ncase = 0
    for K in _stage_matrices(n, s, rng):
        assert K.flags.c_contiguous
        for v in _vectors(tab, 1.25e-3):
            expected = K @ v
            clean.fill(0.0)
            np.matmul(K, v, out=clean)
            buf.fill(fill)
            assert np.matmul(K, v, out=buf) is buf
            assert _byte_equal(buf, expected), (n, s, ncase)
            assert _byte_equal(clean, expected), (n, s, ncase)
            ncase += 1
    assert ncase == s * (2 * s + 3)
