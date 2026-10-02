"""The models the integrator tests share.

``build(name, variant, *args)`` returns ``(dae, y0)`` with ``y0`` a new
float64 array, for the variants ``'inline_sparse'``, ``'inline_dense'`` and
``'rendered'``. A rendered model needs ``directory``, a fresh directory, and
is compiled with ``jit=True`` under a module name unique in the session;
``module_printer`` has no dense option, so there is no rendered dense
variant. The builders ending in ``_model`` return the symbolic model and its
``Vars``, for tests that need a ``Vars``.

- ``dae_test``: ``x' = -x**3 + 0.5 y**2``, ``0 = x**2 + y**2 - 2`` from
  ``x = y = 1``, the model of ``tests/test_dae.py``.
- ``forced``: a hand-built ``nDAE`` with ``x' = -x + z``,
  ``0 = z - (1 + sin t)`` from ``x = 0``, ``z = 1``, ``M`` a ``csc_array`` and
  ``J`` a ``csc_array`` (``'inline_sparse'``) or an ndarray
  (``'inline_dense'``). Solverz has no symbol for time, so explicit time
  reaches a generated residual only through a ``TimeSeriesParam``, and this
  model exists inline only; ``dF/dt = [0, -cos t]`` exactly.
- ``trace``: ``x' = -x + u(t)``, ``0 = z - x u(t)`` from ``x = 0.5``,
  ``z = 0``, with the ``TimeSeriesParam`` ``u`` through ``(0, 0)``,
  ``(0.1, 1)``, ``(0.5, 1)``, ``(1, 0.5)``; the non-autonomous model of the
  rendered variant.
- ``vdp``: Van der Pol with ``mu = 10`` from ``x = [2, 0]``, whose many
  rejected attempts exercise the retry path.
- ``alloc(n)``: ``x' = -k x + sin x`` from ``x = linspace(0.5, 1.5, n)``,
  ``k = linspace(0.9, 1.1, n)``; its Jacobian is diagonal.
- ``ladder(n)``: ``x' = A x + 0.1 sin x`` from ``x = linspace(0.5, 1.5, n)``,
  ``A`` the sparse tridiagonal matrix of ``-2`` on the diagonal and ``1``
  beside it, so that ``W`` is not diagonal.
- ``ball``: the bouncing ball ``x0' = x1``, ``x1' = -9.8`` from ``[0, 20]``.
- ``orbit``: the restricted three-body orbit of ``test_rodas_event.py``.
- ``permuted``: ``dae_test`` with the algebraic equation declared first, so
  that the rows of ``M`` are not aligned with the variables.
"""
import importlib
import itertools
import sys

import numpy as np
from scipy.sparse import csc_array, diags_array

from Solverz import (Eqn, Mat_Mul, Model, Ode, Param, TimeSeriesParam, Var, made_numerical,
                     module_printer, sin)
from Solverz.num_api.num_eqn import nDAE

VARIANTS = ('inline_sparse', 'inline_dense', 'rendered')

_module_ids = itertools.count()


def dae_test_model():
    m = Model()
    m.x = Var('x', 1)
    m.y = Var('y', 1)
    m.f = Ode(name='f', f=-m.x ** 3 + 0.5 * m.y ** 2, diff_var=m.x)
    m.g = Eqn(name='g', eqn=m.x ** 2 + m.y ** 2 - 2)
    return m.create_instance()


def permuted_model():
    m = Model()
    m.x = Var('x', 1)
    m.y = Var('y', 1)
    m.g = Eqn(name='g', eqn=m.x ** 2 + m.y ** 2 - 2)
    m.f = Ode(name='f', f=-m.x ** 3 + 0.5 * m.y ** 2, diff_var=m.x)
    return m.create_instance()


def trace_model():
    m = Model()
    m.x = Var('x', [0.5])
    m.z = Var('z', [0.0])
    m.u = TimeSeriesParam('u', v_series=[0.0, 1.0, 1.0, 0.5], time_series=[0.0, 0.1, 0.5, 1.0],
                          index=np.arange(1), value=np.zeros(1))
    m.fx = Ode('fx', f=-m.x + m.u, diff_var=m.x)
    m.gz = Eqn('gz', m.z - m.x * m.u)
    return m.create_instance()


def vdp_model(mu=10.0):
    m = Model()
    m.x = Var('x', [2.0, 0.0])
    m.f1 = Ode('f1', m.x[1], m.x[0])
    m.f2 = Ode('f2', mu * (1 - m.x[0] ** 2) * m.x[1] - m.x[0], m.x[1])
    return m.create_instance()


def alloc_model(n):
    m = Model()
    m.x = Var('x', np.linspace(0.5, 1.5, n))
    m.k = Param('k', np.linspace(0.9, 1.1, n))
    m.decay = Ode('decay', f=-m.k * m.x + sin(m.x), diff_var=m.x)
    return m.create_instance()


def ladder_model(n):
    A = diags_array([np.ones(n - 1), np.full(n, -2.0), np.ones(n - 1)], offsets=[-1, 0, 1],
                    shape=(n, n), format='csc')
    m = Model()
    m.x = Var('x', np.linspace(0.5, 1.5, n))
    m.A = Param('A', csc_array(A), dim=2, sparse=True)
    m.f = Ode('f', f=Mat_Mul(m.A, m.x) + 0.1 * sin(m.x), diff_var=m.x)
    return m.create_instance()


def ball_model():
    m = Model()
    m.x = Var('x', [0, 20])
    m.f1 = Ode('f1', m.x[1], m.x[0])
    m.f2 = Ode('f2', -9.8, m.x[1])
    return m.create_instance()


def orbit_model():
    mu = 1 / 82.45
    mustar = 1 - mu
    m = Model()
    m.y = Var('y', [1.2, 0, 0, -1.04935750983031990726])
    m.f1 = Ode('f1', m.y[2], m.y[0])
    m.f2 = Ode('f2', m.y[3], m.y[1])
    r13 = ((m.y[0] + mu) ** 2 + m.y[1] ** 2) ** 1.5
    r23 = ((m.y[0] - mustar) ** 2 + m.y[1] ** 2) ** 1.5
    m.f3 = Ode('f3',
               2 * m.y[3] + m.y[0] - mustar * ((m.y[0] + mu) / r13) - mu * ((m.y[0] - mustar) / r23),
               m.y[2])
    m.f4 = Ode('f4',
               -2 * m.y[2] + m.y[1] - mustar * (m.y[1] / r13) - mu * (m.y[1] / r23),
               m.y[3])
    return m.create_instance()


SYMBOLIC = {
    'dae_test': dae_test_model,
    'permuted': permuted_model,
    'trace': trace_model,
    'vdp': vdp_model,
    'alloc': alloc_model,
    'ladder': ladder_model,
    'ball': ball_model,
    'orbit': orbit_model,
}


def forced(variant='inline_sparse'):
    """The hand-built ``forced`` model; ``variant`` selects a sparse or a dense ``J``."""
    if variant not in ('inline_sparse', 'inline_dense'):
        raise ValueError(f"forced exists inline only, not as {variant!r}")
    M = csc_array(([1.0], ([0], [0])), shape=(2, 2))

    def F(t, y, p, out=None):
        if out is None:
            out = np.empty(2)
        out[0] = -y[0] + y[1]
        out[1] = y[1] - (1.0 + np.sin(t))
        return out

    if variant == 'inline_sparse':
        def J(t, y, p):
            return csc_array(([-1.0, 1.0, 1.0], ([0, 0, 1], [0, 1, 1])), shape=(2, 2))
    else:
        def J(t, y, p):
            return np.array([[-1.0, 1.0], [0.0, 1.0]])

    return nDAE(M, F, J, {}), np.array([0.0, 1.0])


def render(sdae, y0, directory, name):
    """Render ``sdae`` with ``jit=True`` into ``directory`` and import it."""
    module_printer(sdae, y0, name, directory=str(directory), jit=True).render()
    sys.path.insert(0, str(directory))
    try:
        mod = importlib.import_module(name)
    finally:
        sys.path.remove(str(directory))
    return mod.mdl


def build(name, variant='inline_sparse', *args, directory=None):
    """``(dae, y0)`` of the model ``name`` in ``variant``; ``args`` are the
    model's own arguments, such as ``n``."""
    if name == 'forced':
        return forced(variant)
    sdae, y0 = SYMBOLIC[name](*args)
    y = np.array(y0.array, dtype=np.float64)
    if variant == 'inline_sparse':
        return made_numerical(sdae, y0, sparse=True), y
    if variant == 'inline_dense':
        return made_numerical(sdae, y0, sparse=False), y
    if variant == 'rendered':
        if directory is None:
            raise ValueError("a rendered model needs a directory")
        tag = '_'.join([name] + [str(a) for a in args])
        return render(sdae, y0, directory, f"sz_integ_{tag}_{next(_module_ids)}"), y
    raise ValueError(f"unknown variant {variant!r}")
