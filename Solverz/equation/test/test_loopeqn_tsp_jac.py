"""Regression tests for the LoopEqn-with-TimeSeriesParam Jacobian bug.

The bug
-------
Solverz commit ``141ea3a`` (LoopEqn Jacobian pipeline Phase J1,
2026-04-15, shipped to ``dev`` via PR #135 on 2026-04-25) introduced
``is_constant_matrix_deri`` to classify ``DeriExpr`` blocks as
constant when they have no Var / IdxVar / iAliasVar / IdxAliasVar
free symbols, baking the value into ``setting["data"]`` at module
build time and skipping per-step re-evaluation.

The predicate **did not inspect whether a ``Para`` free symbol
corresponded to a ``TimeSeriesParam``**. As a result, any LoopEqn
body multiplying a ``Var`` by a ``TimeSeriesParam`` factor (e.g.
``... - G_shunt[i] * ux[i]``, the standard three-phase short-circuit
fault-injection shape) produced a Jacobian that:

* evaluated F correctly via ``p_["G_shunt"].get_v_t(t)`` (the F
  code path was untouched),
* but FROZE J at the build-time ``G_shunt = 0`` default and never
  updated it at runtime.

Rodas's modified-Newton iterations then drove the algebraic system
with the wrong Jacobian during faults, producing silent integration
failures (100 consecutive step rejections at the fault inception)
and physically meaningless trajectories on any simulation that uses
``TimeSeriesParam`` for shunt-fault injection, leak rates, or any
other parameter-coupled time-varying coefficient.

These tests guard against a recurrence in two layers:

1. ``is_constant_matrix_deri`` must return ``False`` when any
   ``Para`` free symbol corresponds to a ``TimeSeriesParam`` in the
   passed PARAM dict (jac.py-level unit test).

2. The full LoopEqn pipeline must route a body containing a
   ``TimeSeriesParam`` factor through the ``LoopEqnDiff``
   dense-kernel fallback (loop_jac.py-level integration test): the
   resulting JacBlock must carry ``_loop_eqn_diff`` (so the module
   printer emits a dynamic kernel taking the TimeSeriesParam as
   argument), and ``mdl.J(t, y, p)`` evaluated at distinct times
   inside / outside the fault window must return distinct values.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
import uuid

import numpy as np
import pytest
from scipy.sparse import csc_array

from Solverz import (
    Eqn,
    Idx,
    LoopEqn,
    Model,
    Ode,
    Param,
    Sum,
    TimeSeriesParam,
    Var,
    module_printer,
)
from Solverz.equation.jac import is_constant_matrix_deri
from Solverz.equation.param import TimeSeriesParam as TSPClass
from Solverz.sym_algebra.functions import Diag
from Solverz.sym_algebra.symbols import Para


_TEST_TEMPDIRS = []


def _mdl_from_module(spf, y0, jit: bool = False):
    """Render ``spf`` with ``module_printer`` into a tempdir and
    import the generated module. The inline ``made_numerical``
    rejects LoopEqn (issue #132), so LoopEqn tests must go through
    a rendered module."""
    mod_name = f'_sz_tsp_test_{uuid.uuid4().hex[:8]}'
    d = tempfile.mkdtemp()
    _TEST_TEMPDIRS.append(d)
    printer = module_printer(spf, y0, mod_name, directory=d, jit=jit)
    printer.render()
    sys.path.insert(0, d)
    mod = importlib.import_module(mod_name)
    return mod.mdl, mod.y


# ---------------------------------------------------------------------------
# Layer 1 — predicate unit tests
# ---------------------------------------------------------------------------


def test_is_constant_matrix_deri_flags_timeseries_param_diag():
    """``Diag(G_shunt)`` must be reported as non-constant when
    ``G_shunt`` is a ``TimeSeriesParam`` in PARAM."""
    Gs = Para("G_shunt")
    expr = -Diag(Gs)
    PARAM = {
        "G_shunt": TSPClass(
            "G_shunt", v_series=[0.0, 0.0], time_series=[0.0, 1.0]
        ),
    }
    assert is_constant_matrix_deri(expr) is True, (
        "without PARAM the predicate is permissive"
    )
    assert is_constant_matrix_deri(expr, PARAM=PARAM) is False, (
        "with PARAM the predicate must see G_shunt as TimeSeriesParam"
    )


def test_is_constant_matrix_deri_flags_mixed_static_plus_tsp():
    """The bug shape: ``-Gbus - Diag(G_shunt)``. Both factors are
    ``Para`` (so the old predicate said 'constant'); the new
    predicate must detect the TimeSeriesParam mixed in."""
    Gbus = Para("Gbus", dim=2)
    Gs = Para("G_shunt")
    expr = -Gbus - Diag(Gs)
    G_arr = np.array([[1.0]])
    PARAM = {
        "Gbus": Param("Gbus", value=G_arr, dim=2),
        "G_shunt": TSPClass(
            "G_shunt", v_series=[0.0, 0.0], time_series=[0.0, 1.0]
        ),
    }
    assert is_constant_matrix_deri(expr) is True, (
        "regression: old predicate is permissive without PARAM"
    )
    assert is_constant_matrix_deri(expr, PARAM=PARAM) is False, (
        "regression: predicate must flag the TimeSeriesParam contribution"
    )


def test_is_constant_matrix_deri_preserves_pure_static_para():
    """Pure-static blocks (no TimeSeriesParam) must STILL be
    classified as constant — otherwise we lose the fast-path
    optimisation for ordinary Mat_Mul Jacobians."""
    Gbus = Para("Gbus", dim=2)
    expr = -Gbus
    G_arr = np.array([[1.0]])
    PARAM = {"Gbus": Param("Gbus", value=G_arr, dim=2)}
    assert is_constant_matrix_deri(expr, PARAM=PARAM) is True


# ---------------------------------------------------------------------------
# Layer 2 — LoopEqn end-to-end
# ---------------------------------------------------------------------------


def _build_eps_loopeqn_model(nb: int = 3):
    """Build the minimal LoopEqn EPS shape that triggered the bug:

        ix_inj[i] := ix[i] - Sum(Gbus[i,j]*ux[j], j) - G_shunt[i]*ux[i]

    plus enough balancing ``Eqn``s to make the system square. The
    shape mirrors the rectangular current-balance LoopEqn that
    SolMuseum's ``eps_network.mdl(dyn=True, loopeqn=True)`` emits.

    A trivial ``Ode`` is added so the model compiles as a DAE
    (which produces ``F_(t, y, p)`` taking the time argument the
    TimeSeriesParam's ``get_v_t(t)`` reads); an AE-only model
    renders ``F_(y, p)`` without ``t`` and crashes at runtime on
    the first ``G_shunt.get_v_t(t)`` call."""
    G = np.array(
        [
            [2.0, -1.0, -1.0],
            [-1.0, 3.0, -2.0],
            [-1.0, -2.0, 4.0],
        ]
    )
    m = Model()
    m.ix = Var("ix", np.zeros(nb))
    m.ux = Var("ux", np.ones(nb))
    m.Gbus = Param("Gbus", csc_array(G), dim=2, sparse=True)
    m.G_shunt = TimeSeriesParam(
        "G_shunt",
        v_series=[0.0, 0.0],
        time_series=[0.0, 1.0e9],
        index=0,
        value=np.zeros(nb),
    )
    # Unique Idx names per test invocation — sympy ``Idx`` is name-keyed
    # globally and other tests in the suite that use ``Idx('i')`` /
    # ``Idx('j')`` would otherwise cross-contaminate substitution state.
    suffix = uuid.uuid4().hex[:8]
    i = Idx(f"_sz_tsp_i_{suffix}", nb)
    j = Idx(f"_sz_tsp_j_{suffix}", nb)
    m.ix_inj = LoopEqn(
        "ix_inj",
        outer_index=i,
        body=(
            m.ix[i]
            - Sum(m.Gbus[i, j] * m.ux[j], j)
            - m.G_shunt[i] * m.ux[i]
        ),
        model=m,
    )
    for k in range(nb):
        m.add(Eqn(f"P_{k}", m.ix[k] - m.ux[k]))
    # Trivial dummy state to make the model a DAE (required so the
    # rendered F_/J_ wrappers take ``t`` as their first arg).
    m.dummy = Var("dummy", np.array([0.0]))
    m.dummy_ode = Ode("dummy_ode", -m.dummy, diff_var=m.dummy)
    return m, G


def test_loopeqn_with_tsp_routes_through_loopeqndiff():
    """The bug shape's JacBlock for ``d ix_inj / d ux`` must end up
    in the ``LoopEqnDiff`` fallback so the module printer emits a
    dynamic kernel. Prior to the fix the Phase J2 classifier
    happily translated ``-Gbus - Diag(G_shunt)`` and the block
    landed in the mutable-matrix path that bakes its data at
    build time."""
    m, _ = _build_eps_loopeqn_model()
    sdae, y0 = m.create_instance()
    sdae.FormJac(y0)
    jb_ux = sdae.jac.blocks["ix_inj"][m.ux.symbol]
    assert hasattr(jb_ux, "_loop_eqn_diff"), (
        "regression: ix_inj/ux must route through LoopEqnDiff so a "
        "runtime kernel re-evaluates the TimeSeriesParam every J call"
    )


def test_loopeqn_with_tsp_jac_updates_at_runtime():
    """``mdl.J(t, y, p)`` must reflect the time-varying
    ``G_shunt[i]`` value. We override ``G_shunt`` on the compiled
    model with a step profile (zero outside the fault, 100 inside)
    and check that the diagonal entry ``d(ix_inj_0)/d(ux[0])`` of
    the rendered J differs between t = pre-fault and t = mid-fault
    by exactly the fault delta. Pre-fix the J entry was baked at
    the build-time TimeSeriesParam value (zero) and never updated."""
    m, G = _build_eps_loopeqn_model()
    sdae, y0 = m.create_instance()
    mdl, y = _mdl_from_module(sdae, y0, jit=False)

    nb = G.shape[0]
    mdl.p["G_shunt"] = TimeSeriesParam(
        "G_shunt",
        v_series=[0.0, 0.0, 100.0, 100.0, 0.0, 0.0],
        time_series=[0.0, 1.0, 1.001, 2.0, 2.001, 10.0],
        index=0,
        value=np.zeros(nb),
    )

    # ix and ux are LoopEqn-managed Vars; their address ranges
    # tell us the J row/column for d(ix_inj_0)/d(ux[0]).
    row = y.a["ix"].start  # ix_inj_0 row
    col = y.a["ux"].start  # ux[0]  col

    J_pre = mdl.J(0.5, y.array, mdl.p).toarray()
    J_mid = mdl.J(1.5, y.array, mdl.p).toarray()
    J_post = mdl.J(3.0, y.array, mdl.p).toarray()

    # Pre-fault: d(ix_inj_0)/d(ux[0]) = -Gbus[0,0] - 0 = -Gbus[0,0].
    np.testing.assert_allclose(
        J_pre[row, col], -G[0, 0], rtol=0, atol=1e-12
    )
    # Mid-fault: -Gbus[0,0] - 100.
    np.testing.assert_allclose(
        J_mid[row, col], -G[0, 0] - 100.0, rtol=0, atol=1e-12
    )
    # Post-fault: back to pre-fault value (regression: any non-zero
    # leftover here would mean the kernel mutates global state).
    np.testing.assert_allclose(
        J_post[row, col], -G[0, 0], rtol=0, atol=1e-12
    )


def test_loopeqn_with_tsp_jac_off_diagonal_unchanged():
    """The off-diagonal ``d(ix_inj_0)/d(ux[1]) = -Gbus[0,1]`` must
    stay constant across the fault — only the diagonal entry
    receives the G_shunt contribution. Guards against an over-eager
    kernel that touches the wrong slots."""
    m, G = _build_eps_loopeqn_model()
    sdae, y0 = m.create_instance()
    mdl, y = _mdl_from_module(sdae, y0, jit=False)
    nb = G.shape[0]
    mdl.p["G_shunt"] = TimeSeriesParam(
        "G_shunt",
        v_series=[0.0, 0.0, 100.0, 100.0, 0.0, 0.0],
        time_series=[0.0, 1.0, 1.001, 2.0, 2.001, 10.0],
        index=0,
        value=np.zeros(nb),
    )
    row = y.a["ix"].start          # ix_inj_0
    col_other = y.a["ux"].start + 1  # ux[1]
    J_pre = mdl.J(0.5, y.array, mdl.p).toarray()
    J_mid = mdl.J(1.5, y.array, mdl.p).toarray()
    np.testing.assert_allclose(
        J_pre[row, col_other], -G[0, 1], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        J_mid[row, col_other], -G[0, 1], rtol=0, atol=1e-12
    )
