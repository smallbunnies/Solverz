"""Phase 0 smoke tests for ``LoopEqn`` — declare an equation as a
parameterised scalar template that prints to a Python ``for``-loop in
the generated ``inner_F`` / ``inner_J`` code.

The substrate for the body of a ``LoopEqn`` is **sympy's native
``IndexedBase`` + ``Idx`` + ``Sum``**, NOT Solverz's own
``IdxVar``/``IdxPara`` (those are opaque to sympy ``subs`` /
``Sum.doit()``, see ``PHASE0_FINDINGS.md`` Iteration 0).
The user provides a ``var_map`` linking each ``IndexedBase`` name to a
real Solverz ``Var`` / ``Param``. Solverz's printer translates the
``IndexedBase`` references back to concrete Var/Param accesses at
code-gen time, and replaces ``Sum`` nodes with Python ``for``-loops.
"""

import os
import sys
import tempfile

import numpy as np
import sympy as sp
from scipy.sparse import csc_array

from Solverz import (
    Abs, Eqn, Idx, LoopEqn, Mat_Mul, Model, Param, Sign, Sum, Var,
    cos, heaviside, made_numerical, module_printer, nr_method, sin,
)


def test_loop_eqn_eps_minimal():
    """Phase 0 minimum smoke test.

    A 3-bus EPS-style current-balance model expressed as a single
    ``LoopEqn``. The math: for each bus ``i``, enforce

        ix[i] - sum_j (G[i, j] * ux[j] - B[i, j] * uy[j]) = 0
        iy[i] - sum_j (G[i, j] * uy[j] + B[i, j] * ux[j]) = 0

    where ``G`` and ``B`` are the real and imaginary parts of the bus
    admittance matrix.  We pin ``ix`` and ``iy`` to known values and
    expect the Newton solve to recover the corresponding ``ux`` / ``uy``.

    The two equations are declared as **two separate** ``LoopEqn``s so
    each emits one ``inner_F`` sub-function with a for-loop body, NOT
    ``2 * nb`` scalar Eqns.

    Stock Solverz 0.8.3 does not have ``LoopEqn`` — this import will
    fail. That ImportError is the first observation in
    ``PHASE0_FINDINGS.md``, and it drives Hypothesis A in step 0.3.
    """
    # The import is the first thing that fails on stock Solverz —
    # putting it inside the function so test discovery still picks up
    # the test file even when LoopEqn is not defined yet.
    from Solverz import LoopEqn  # type: ignore[attr-defined]

    nb = 3
    # G and B are the real / imag parts of a 3-bus admittance matrix
    # WITH non-trivial diagonal shunts — without the shunts, both
    # matrices have zero row sums (the structural ``Y @ 1 = 0``
    # property of a bus-admittance matrix that has no ground branches),
    # which makes the resulting Newton Jacobian rank-deficient.
    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    B_dense = np.array([
        [ 0.7, -0.2, -0.3],
        [-0.2,  0.6, -0.2],
        [-0.3, -0.2,  0.6],
    ])

    # A known voltage profile we want the Newton solve to recover.
    ux_target = np.array([1.0,  0.95, 0.92])
    uy_target = np.array([0.0, -0.05, -0.02])
    # The corresponding current injections that the equations enforce.
    ix_pin = G_dense @ ux_target - B_dense @ uy_target
    iy_pin = G_dense @ uy_target + B_dense @ ux_target

    m = Model()
    m.ux = Var('ux', np.ones(nb))           # flat-start initial guess
    m.uy = Var('uy', np.zeros(nb))
    # Use DENSE matrix params for the Phase 0 prototype — sparse
    # CSC printing is a separate concern that can be addressed in 0.4.x.
    m.G = Param('G', G_dense, dim=2, sparse=False)
    m.B = Param('B', B_dense, dim=2, sparse=False)
    m.ix_pin = Param('ix_pin', ix_pin)
    m.iy_pin = Param('iy_pin', iy_pin)

    # === Build the LoopEqn body using sympy native primitives ===
    i, j = sp.Idx('i'), sp.Idx('j')
    ux_sym = sp.IndexedBase('ux')
    uy_sym = sp.IndexedBase('uy')
    G_sym = sp.IndexedBase('G')
    B_sym = sp.IndexedBase('B')
    ix_pin_sym = sp.IndexedBase('ix_pin')
    iy_pin_sym = sp.IndexedBase('iy_pin')

    # Body for the real-part current balance row i:
    #   ix_pin[i] - sum_j (G[i, j] * ux[j] - B[i, j] * uy[j])
    body_re = ix_pin_sym[i] - sp.Sum(
        G_sym[i, j] * ux_sym[j] - B_sym[i, j] * uy_sym[j],
        (j, 0, nb - 1),
    )

    body_im = iy_pin_sym[i] - sp.Sum(
        G_sym[i, j] * uy_sym[j] + B_sym[i, j] * ux_sym[j],
        (j, 0, nb - 1),
    )

    var_map = {
        'ux': m.ux, 'uy': m.uy,
        'G': m.G, 'B': m.B,
        'ix_pin': m.ix_pin, 'iy_pin': m.iy_pin,
    }

    m.eqn_re = LoopEqn(
        'eqn_re',
        outer_index=i,
        n_outer=nb,
        body=body_re,
        var_map=var_map,
    )
    m.eqn_im = LoopEqn(
        'eqn_im',
        outer_index=i,
        n_outer=nb,
        body=body_im,
        var_map=var_map,
    )

    # === Stage 1: create_instance ===
    spf, y0 = m.create_instance()

    # === Stage 2: made_numerical ===
    mdl = made_numerical(spf, y0, sparse=True)

    # === Stage 3: F evaluation at flat-start ===
    F_val = mdl.F(y0, mdl.p)
    assert F_val.shape == (2 * nb,)
    # At flat-start ux=1, uy=0:
    #   F_re[i] = ix_pin[i] - (G @ 1 - B @ 0)[i] = ix_pin[i] - G.sum(axis=1)[i]
    #   F_im[i] = iy_pin[i] - (G @ 0 + B @ 1)[i] = iy_pin[i] - B.sum(axis=1)[i]
    expected_F_re = ix_pin - G_dense.sum(axis=1)
    expected_F_im = iy_pin - B_dense.sum(axis=1)
    np.testing.assert_allclose(
        F_val,
        np.concatenate([expected_F_re, expected_F_im]),
        atol=1e-12,
    )

    # === Stage 4: Newton solve ===
    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y['ux'], ux_target, atol=1e-8)
    np.testing.assert_allclose(sol.y['uy'], uy_target, atol=1e-8)


def test_loop_eqn_eps_dyn_with_var_residual():
    """Phase 0.4.1: outer-index Var residual term (identity Jacobian).

    The actual SolMuseum ``eps_network.mdl(dyn=True)`` model
    (``SolMuseum/ae/eps_network.py:88-97``) writes the rectangular
    current-balance row as

        ``ix[i] - sum_j (G[i,j] * ux[j] - B[i,j] * uy[j]) = 0``

    where **``ix`` is a Var**, NOT a parameter. Differentiating
    ``ix[i]`` w.r.t. ``ix[k]`` produces ``KroneckerDelta(i, k)`` — i.e.
    the per-(eqn-row, var-col) Jacobian block is the identity matrix.

    Phase 0 minimum doesn't handle this (only handles the simpler
    ``±IndexedBase[outer, k]`` → ``±Para(dim=2)`` pattern). This test
    drives the next translator extension.

    Setup: keep ``ix_pin`` and ``iy_pin`` as Params for the bus-side
    forcing, but introduce ``ix`` and ``iy`` as Vars whose residual is
    ``ix[i] - (the right-hand side)``. Add two extra "anchor" Eqns
    that enforce ``ix == ix_pin`` and ``iy == iy_pin`` so the system is
    square (4*nb equations in 4*nb variables).
    """
    from Solverz import LoopEqn

    nb = 3
    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    B_dense = np.array([
        [ 0.7, -0.2, -0.3],
        [-0.2,  0.6, -0.2],
        [-0.3, -0.2,  0.6],
    ])

    ux_target = np.array([1.0, 0.95, 0.92])
    uy_target = np.array([0.0, -0.05, -0.02])
    ix_pin = G_dense @ ux_target - B_dense @ uy_target
    iy_pin = G_dense @ uy_target + B_dense @ ux_target

    m = Model()
    m.ux = Var('ux', np.ones(nb))
    m.uy = Var('uy', np.zeros(nb))
    m.ix = Var('ix', np.zeros(nb))   # NEW — Var, not Param
    m.iy = Var('iy', np.zeros(nb))   # NEW — Var, not Param
    m.G = Param('G', G_dense, dim=2, sparse=False)
    m.B = Param('B', B_dense, dim=2, sparse=False)
    m.ix_pin = Param('ix_pin', ix_pin)
    m.iy_pin = Param('iy_pin', iy_pin)

    i, j = sp.Idx('i'), sp.Idx('j')
    ux_sym = sp.IndexedBase('ux')
    uy_sym = sp.IndexedBase('uy')
    ix_sym = sp.IndexedBase('ix')
    iy_sym = sp.IndexedBase('iy')
    G_sym = sp.IndexedBase('G')
    B_sym = sp.IndexedBase('B')

    body_re = ix_sym[i] - sp.Sum(
        G_sym[i, j] * ux_sym[j] - B_sym[i, j] * uy_sym[j],
        (j, 0, nb - 1),
    )
    body_im = iy_sym[i] - sp.Sum(
        G_sym[i, j] * uy_sym[j] + B_sym[i, j] * ux_sym[j],
        (j, 0, nb - 1),
    )

    var_map = {
        'ux': m.ux, 'uy': m.uy,
        'ix': m.ix, 'iy': m.iy,
        'G': m.G, 'B': m.B,
    }

    m.eqn_re = LoopEqn('eqn_re', outer_index=i, n_outer=nb,
                       body=body_re, var_map=var_map)
    m.eqn_im = LoopEqn('eqn_im', outer_index=i, n_outer=nb,
                       body=body_im, var_map=var_map)

    # Anchor eqns to make the system square: ix == ix_pin, iy == iy_pin.
    m.anchor_ix = Eqn('anchor_ix', m.ix - m.ix_pin)
    m.anchor_iy = Eqn('anchor_iy', m.iy - m.iy_pin)

    spf, y0 = m.create_instance()
    mdl = made_numerical(spf, y0, sparse=True)

    # F-side smoke: residual at flat-start.
    F_val = mdl.F(y0, mdl.p)
    assert F_val.shape == (4 * nb,)

    # Newton solve.
    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y['ux'], ux_target, atol=1e-8)
    np.testing.assert_allclose(sol.y['uy'], uy_target, atol=1e-8)
    np.testing.assert_allclose(sol.y['ix'], ix_pin, atol=1e-8)
    np.testing.assert_allclose(sol.y['iy'], iy_pin, atol=1e-8)


def test_loop_eqn_mass_continuity():
    """Phase 0.4.2: 1-D Var inside Sum at inner dummy + 2-D incidence Param.

    Models the heat / gas network mass-continuity equation:

        ``f_inj[node] + sum_p (V_in[node, p] - V_out[node, p]) * m[p] = 0``

    where:

    - ``f_inj[node]`` is a 1-D Param (net injection per node) — Phase 0
      already handles this via the ``IndexedBase('M')[outer]`` 1-D
      Param branch (used in the EPS smoke test).
    - ``m[p]`` is a 1-D Var (pipe mass-flow) accessed at the inner
      sum dummy — this is the NEW pattern.
    - ``V_in[node, p]`` and ``V_out[node, p]`` are 2-D incidence Params
      (1 wherever pipe ``p`` enters / leaves node).

    The Jacobian:

    - ``∂F[node]/∂m[k] = V_in[node, k] - V_out[node, k]``
      → translates to ``Para('V_in', dim=2) - Para('V_out', dim=2)``,
      which JacBlock should classify as a (mutable, but
      free-symbol-only-on-Paras) matrix block. Phase 0 might still
      exercise the constant-matrix path because the result is ``Para
      + (-Para)``, which sympy may simplify.

    Topology (4 nodes, 3 pipes; linear chain):

        f_inj=+6 ─[p0]─→ ─[p1]─→ ─[p2]─→ f_inj=-6
          node 0    node 1    node 2    node 3

    All three nodes other than the sink carry flow 6 (mass conserved).
    The 4-node mass continuity has rank 3 (rows sum to 0), so we use
    ``n_outer = 3`` to skip the redundant 4th row. Square system: 3
    pipe vars, 3 LoopEqn rows.
    """
    from Solverz import LoopEqn

    n_node_eqn = 3   # n_outer for the LoopEqn — skip the redundant 4th
    n_pipe = 3
    V_in_full = np.array([
        [0.0, 0.0, 0.0],   # node 0: no inflow
        [1.0, 0.0, 0.0],   # node 1: pipe 0 in
        [0.0, 1.0, 0.0],   # node 2: pipe 1 in
    ])
    V_out_full = np.array([
        [1.0, 0.0, 0.0],   # node 0: pipe 0 out
        [0.0, 1.0, 0.0],   # node 1: pipe 1 out
        [0.0, 0.0, 1.0],   # node 2: pipe 2 out
    ])
    f_inj_full = np.array([6.0, 0.0, 0.0])

    m = Model()
    m.m_pipe = Var('m_pipe', np.zeros(n_pipe))     # flat-start
    m.V_in = Param('V_in', V_in_full, dim=2, sparse=False)
    m.V_out = Param('V_out', V_out_full, dim=2, sparse=False)
    m.f_inj = Param('f_inj', f_inj_full)

    i, p = sp.Idx('i'), sp.Idx('p')
    f_inj_sym = sp.IndexedBase('f_inj')
    V_in_sym = sp.IndexedBase('V_in')
    V_out_sym = sp.IndexedBase('V_out')
    m_pipe_sym = sp.IndexedBase('m_pipe')

    body_mass = f_inj_sym[i] + sp.Sum(
        (V_in_sym[i, p] - V_out_sym[i, p]) * m_pipe_sym[p],
        (p, 0, n_pipe - 1),
    )
    var_map_mass = {
        'm_pipe': m.m_pipe,
        'V_in': m.V_in,
        'V_out': m.V_out,
        'f_inj': m.f_inj,
    }
    m.mass_eqn = LoopEqn(
        'mass_eqn',
        outer_index=i,
        n_outer=n_node_eqn,
        body=body_mass,
        var_map=var_map_mass,
    )

    spf, y0 = m.create_instance()
    mdl = made_numerical(spf, y0, sparse=True)

    sol = nr_method(mdl, y0)
    m_target = np.array([6.0, 6.0, 6.0])
    np.testing.assert_allclose(sol.y['m_pipe'], m_target, atol=1e-8)


def _build_eps_minimal_model(nb=3):
    """Shared model factory for the Phase 0 minimum + JIT path tests."""
    from Solverz import LoopEqn

    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    B_dense = np.array([
        [ 0.7, -0.2, -0.3],
        [-0.2,  0.6, -0.2],
        [-0.3, -0.2,  0.6],
    ])
    ux_target = np.array([1.0, 0.95, 0.92])
    uy_target = np.array([0.0, -0.05, -0.02])
    ix_pin_v = G_dense @ ux_target - B_dense @ uy_target
    iy_pin_v = G_dense @ uy_target + B_dense @ ux_target

    m = Model()
    m.ux = Var('ux', np.ones(nb))
    m.uy = Var('uy', np.zeros(nb))
    m.G = Param('G', G_dense, dim=2, sparse=False)
    m.B = Param('B', B_dense, dim=2, sparse=False)
    m.ix_pin = Param('ix_pin', ix_pin_v)
    m.iy_pin = Param('iy_pin', iy_pin_v)

    i, j = sp.Idx('i'), sp.Idx('j')
    ux_sym = sp.IndexedBase('ux')
    uy_sym = sp.IndexedBase('uy')
    G_sym = sp.IndexedBase('G')
    B_sym = sp.IndexedBase('B')
    ix_pin_sym = sp.IndexedBase('ix_pin')
    iy_pin_sym = sp.IndexedBase('iy_pin')
    body_re = ix_pin_sym[i] - sp.Sum(
        G_sym[i, j] * ux_sym[j] - B_sym[i, j] * uy_sym[j],
        (j, 0, nb - 1),
    )
    body_im = iy_pin_sym[i] - sp.Sum(
        G_sym[i, j] * uy_sym[j] + B_sym[i, j] * ux_sym[j],
        (j, 0, nb - 1),
    )
    var_map = {
        'ux': m.ux, 'uy': m.uy,
        'G': m.G, 'B': m.B,
        'ix_pin': m.ix_pin, 'iy_pin': m.iy_pin,
    }
    m.eqn_re = LoopEqn('eqn_re', outer_index=i, n_outer=nb,
                       body=body_re, var_map=var_map)
    m.eqn_im = LoopEqn('eqn_im', outer_index=i, n_outer=nb,
                       body=body_im, var_map=var_map)

    return m, ux_target, uy_target


def test_loop_eqn_eps_minimal_jit_module():
    """Phase 0.4.3: end-to-end ``module_printer(jit=True)`` path.

    The whole point of LoopEqn is to reduce numba LLVM compile time on
    the IES benchmark. This requires the JIT path to actually work:

    1. ``module_printer(jit=True).render()`` must succeed (no
       ``PrintMethodNotImplementedError`` from sympy's ``pycode`` on
       ``Sum`` over ``Idx``) — handled by emitting a hand-built source
       string for each LoopEqn's ``inner_F<N>`` instead of going
       through the AST + pycode path.

    2. The generated ``@njit(cache=True)``-decorated ``inner_F<N>``
       must contain explicit nested ``for`` loops (NOT Python
       ``sum(...)`` generators, which numba rejects).

    3. The compiled module must produce the same numerical result as
       the inline ``made_numerical`` path: Newton solve converges to
       the analytical answer.

    The eps minimum case has a constant Jacobian (all derivatives are
    bare Params), so ``inner_J`` reduces to ``return _data_`` with the
    full COO data precomputed at module-build time. This is the
    best-case scenario for the JIT path — no runtime Jacobian
    computation, just a constant-data lookup.
    """
    nb = 3
    m, ux_target, uy_target = _build_eps_minimal_model(nb=nb)
    spf, y0 = m.create_instance()

    with tempfile.TemporaryDirectory() as d:
        printer = module_printer(spf, y0, 'phase04_jit_eps',
                                 directory=d, jit=True)
        printer.render()

        sys.path.insert(0, d)
        try:
            import phase04_jit_eps
            mdl = phase04_jit_eps.mdl
            y0_loaded = phase04_jit_eps.y

            sol = nr_method(mdl, y0_loaded)
            np.testing.assert_allclose(sol.y['ux'], ux_target,
                                       atol=1e-8)
            np.testing.assert_allclose(sol.y['uy'], uy_target,
                                       atol=1e-8)

            # Sanity check: the inner_F<N> source must contain an
            # explicit ``for`` loop over the outer index AND the inner
            # sum dummy. If it falls back to a generator expression,
            # numba would reject it long before we get here, but be
            # explicit about what we expect.
            module_path = os.path.join(d, 'phase04_jit_eps',
                                       'num_func.py')
            with open(module_path) as f:
                src = f.read()
            assert 'def inner_F0' in src
            assert 'for i in range(3):' in src
            assert 'for j in range(0, 3):' in src
            assert '_sz_loop_acc_0' in src
            # No generator expression
            assert 'sum(' not in src
        finally:
            sys.path.remove(d)
            for mod_name in list(sys.modules):
                if mod_name.startswith('phase04_jit_eps'):
                    del sys.modules[mod_name]


def test_loop_eqn_sparse_walker_inline():
    """Phase 1 sparse-walker smoke test (inline path).

    Reuses the 3-bus EPS minimum geometry but declares ``G`` and ``B``
    as sparse 2-D ``Param``s. Each current-balance row is a separate
    ``LoopEqn`` whose Sum body contains exactly ONE sparse walker
    (``G`` for the real part, ``B`` for the imag part, split across
    FOUR LoopEqns). The translator should emit CSR-walking code (not
    a dense ``for j in range(0, 3)`` loop), and the generated function
    should close over the CSR arrays via the ``exec`` namespace.

    The expected numerics match the dense test exactly because the
    test matrices have no structural zeros being lost.
    """
    from Solverz import LoopEqn

    nb = 3
    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    B_dense = np.array([
        [ 0.7, -0.2, -0.3],
        [-0.2,  0.6, -0.2],
        [-0.3, -0.2,  0.6],
    ])

    ux_target = np.array([1.0,  0.95, 0.92])
    uy_target = np.array([0.0, -0.05, -0.02])
    ix_pin = G_dense @ ux_target - B_dense @ uy_target
    iy_pin = G_dense @ uy_target + B_dense @ ux_target

    m = Model()
    m.ux = Var('ux', np.ones(nb))
    m.uy = Var('uy', np.zeros(nb))
    # Sparse 2-D Params — stored as csc_array in Solverz's default
    # convention. LoopEqn detects these and switches to CSR walking.
    m.G = Param('G', csc_array(G_dense), dim=2, sparse=True)
    m.B = Param('B', csc_array(B_dense), dim=2, sparse=True)
    m.ix_pin = Param('ix_pin', ix_pin)
    m.iy_pin = Param('iy_pin', iy_pin)

    i, j = sp.Idx('i'), sp.Idx('j')
    ux_sym = sp.IndexedBase('ux')
    uy_sym = sp.IndexedBase('uy')
    G_sym = sp.IndexedBase('G')
    B_sym = sp.IndexedBase('B')
    ix_pin_sym = sp.IndexedBase('ix_pin')
    iy_pin_sym = sp.IndexedBase('iy_pin')

    # Phase 1 sparse walker supports ONE sparse walker per Sum. Split
    # the real-part eqn into two Sums (one per walker) and combine via
    # arithmetic — this is the "Case A" the plan locks us to until
    # shared-skeleton Case B lands.
    body_re = (
        ix_pin_sym[i]
        - sp.Sum(G_sym[i, j] * ux_sym[j], (j, 0, nb - 1))
        + sp.Sum(B_sym[i, j] * uy_sym[j], (j, 0, nb - 1))
    )
    body_im = (
        iy_pin_sym[i]
        - sp.Sum(G_sym[i, j] * uy_sym[j], (j, 0, nb - 1))
        - sp.Sum(B_sym[i, j] * ux_sym[j], (j, 0, nb - 1))
    )

    var_map = {
        'ux': m.ux, 'uy': m.uy,
        'G': m.G, 'B': m.B,
        'ix_pin': m.ix_pin, 'iy_pin': m.iy_pin,
    }

    m.eqn_re = LoopEqn('eqn_re', outer_index=i, n_outer=nb,
                       body=body_re, var_map=var_map)
    m.eqn_im = LoopEqn('eqn_im', outer_index=i, n_outer=nb,
                       body=body_im, var_map=var_map)

    # Pre-init checks: the translator should have registered CSR
    # caches for both walkers, keyed by IndexedBase name.
    assert set(m.eqn_re._sparse_csr.keys()) == {'G', 'B'}
    assert set(m.eqn_im._sparse_csr.keys()) == {'G', 'B'}

    # Generated source for the inline path must contain CSR walk
    # (``_sz_csr_<M>_indptr`` and ``_sz_csr_<M>_indices``) and must
    # NOT contain a dense ``for j in range(0, 3):`` loop.
    src = m.eqn_re.NUM_EQN._loopeqn_source
    assert '_sz_csr_G_indptr' in src
    assert '_sz_csr_B_indptr' in src
    assert '_sz_csr_G_indices' in src
    assert '_sz_csr_G_data' in src
    # The dense form would produce "for j in range(0, 3):"; CSR form
    # binds ``j`` via an assignment "j = _sz_csr_<M>_indices[_sz_kk_N]".
    assert 'for j in range(0, 3):' not in src
    assert 'j = _sz_csr_G_indices' in src or 'j = _sz_csr_B_indices' in src

    spf, y0 = m.create_instance()
    mdl = made_numerical(spf, y0, sparse=True)

    # F-side evaluation at flat-start should match the dense expected.
    F_val = mdl.F(y0, mdl.p)
    assert F_val.shape == (2 * nb,)
    expected_F_re = ix_pin - G_dense.sum(axis=1)   # flat ux = 1, uy = 0
    expected_F_im = iy_pin - B_dense.sum(axis=1)
    np.testing.assert_allclose(
        F_val,
        np.concatenate([expected_F_re, expected_F_im]),
        atol=1e-12,
    )

    # Newton solve recovers the target voltages.
    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y['ux'], ux_target, atol=1e-8)
    np.testing.assert_allclose(sol.y['uy'], uy_target, atol=1e-8)


def test_loop_eqn_sparse_walker_jit_module():
    """Phase 1 sparse walker + ``module_printer(jit=True)`` end-to-end.

    Renders the 3-bus EPS geometry with SPARSE ``G`` and ``B`` through
    the JIT module printer, imports the generated module, and solves
    the Newton system. Asserts:

    1. The rendered ``inner_F<N>`` source contains CSR-walking code
       (``_sz_csr_G_indptr[i]``, ``_sz_csr_G_indices[...]``) and NOT
       a dense ``for j in range(0, 3)`` loop.
    2. The generated module references the CSR arrays at module level
       (``_sz_csr_G_data = setting["_sz_csr_G_data"]``).
    3. Newton converges to the analytical answer — proving the
       module-level constants are correctly loaded, numba accepts
       the @njit-decorated inner_F<N>, and CSR walking produces
       numerically identical results to dense walking.
    """
    from Solverz import LoopEqn

    nb = 3
    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    B_dense = np.array([
        [ 0.7, -0.2, -0.3],
        [-0.2,  0.6, -0.2],
        [-0.3, -0.2,  0.6],
    ])

    ux_target = np.array([1.0,  0.95, 0.92])
    uy_target = np.array([0.0, -0.05, -0.02])
    ix_pin = G_dense @ ux_target - B_dense @ uy_target
    iy_pin = G_dense @ uy_target + B_dense @ ux_target

    m = Model()
    m.ux = Var('ux', np.ones(nb))
    m.uy = Var('uy', np.zeros(nb))
    m.G = Param('G', csc_array(G_dense), dim=2, sparse=True)
    m.B = Param('B', csc_array(B_dense), dim=2, sparse=True)
    m.ix_pin = Param('ix_pin', ix_pin)
    m.iy_pin = Param('iy_pin', iy_pin)

    i, j = sp.Idx('i'), sp.Idx('j')
    ux_sym = sp.IndexedBase('ux')
    uy_sym = sp.IndexedBase('uy')
    G_sym = sp.IndexedBase('G')
    B_sym = sp.IndexedBase('B')
    ix_pin_sym = sp.IndexedBase('ix_pin')
    iy_pin_sym = sp.IndexedBase('iy_pin')

    # One sparse walker per Sum (Case A). Split G/B across two Sums
    # per equation so each Sum contains exactly one sparse Param.
    body_re = (
        ix_pin_sym[i]
        - sp.Sum(G_sym[i, j] * ux_sym[j], (j, 0, nb - 1))
        + sp.Sum(B_sym[i, j] * uy_sym[j], (j, 0, nb - 1))
    )
    body_im = (
        iy_pin_sym[i]
        - sp.Sum(G_sym[i, j] * uy_sym[j], (j, 0, nb - 1))
        - sp.Sum(B_sym[i, j] * ux_sym[j], (j, 0, nb - 1))
    )

    var_map = {
        'ux': m.ux, 'uy': m.uy,
        'G': m.G, 'B': m.B,
        'ix_pin': m.ix_pin, 'iy_pin': m.iy_pin,
    }
    m.eqn_re = LoopEqn('eqn_re', outer_index=i, n_outer=nb,
                       body=body_re, var_map=var_map)
    m.eqn_im = LoopEqn('eqn_im', outer_index=i, n_outer=nb,
                       body=body_im, var_map=var_map)

    spf, y0 = m.create_instance()

    with tempfile.TemporaryDirectory() as d:
        printer = module_printer(spf, y0, 'phase1_jit_sparse',
                                 directory=d, jit=True)
        printer.render()

        sys.path.insert(0, d)
        try:
            import phase1_jit_sparse
            mdl = phase1_jit_sparse.mdl
            y0_loaded = phase1_jit_sparse.y

            sol = nr_method(mdl, y0_loaded)
            np.testing.assert_allclose(sol.y['ux'], ux_target,
                                       atol=1e-8)
            np.testing.assert_allclose(sol.y['uy'], uy_target,
                                       atol=1e-8)

            module_path = os.path.join(d, 'phase1_jit_sparse',
                                       'num_func.py')
            with open(module_path) as f:
                src = f.read()

            # Module-level constant loads emitted before any function.
            assert '_sz_csr_G_data = setting["_sz_csr_G_data"]' in src
            assert '_sz_csr_G_indices = setting["_sz_csr_G_indices"]' in src
            assert '_sz_csr_G_indptr = setting["_sz_csr_G_indptr"]' in src
            assert '_sz_csr_B_data = setting["_sz_csr_B_data"]' in src

            # inner_F<N> bodies contain CSR walks, not dense range loops.
            assert 'def inner_F0' in src
            assert '_sz_csr_G_indptr[i]' in src
            assert '_sz_csr_G_indices[_sz_kk_' in src
            assert '_sz_csr_G_data[_sz_kk_' in src
            # No dense inner loop over the full column range.
            assert 'for j in range(0, 3):' not in src
            # No sparse 2-D Param names should appear in an inner_F<N>
            # signature — they must not cross into the @njit body as
            # arguments (they'd be csc_array, which numba can't type).
            # (Use a conservative substring check: "def inner_F0(B"
            # would mean B showed up as first arg.)
            assert 'def inner_F0(B' not in src
            assert 'def inner_F1(B' not in src
            assert 'def inner_F0(G' not in src
        finally:
            sys.path.remove(d)
            for mod_name in list(sys.modules):
                if mod_name.startswith('phase1_jit_sparse'):
                    del sys.modules[mod_name]


def test_loop_eqn_native_solverz_syntax_inline():
    """New preferred API: write the body using ``m.G[i, j]`` /
    ``m.ux[j]`` directly, and pass ``model=m``. LoopEqn walks the
    body once at construction, rewrites every Solverz ``IdxVar`` /
    ``IdxPara`` to its ``sympy.IndexedBase`` equivalent, and auto-
    builds ``var_map`` by looking up each name on the model.

    Verifies numerical equivalence with the legacy ``var_map`` API
    (``test_loop_eqn_sparse_walker_inline`` uses the same geometry).
    """
    from Solverz import LoopEqn

    nb = 3
    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    B_dense = np.array([
        [ 0.7, -0.2, -0.3],
        [-0.2,  0.6, -0.2],
        [-0.3, -0.2,  0.6],
    ])

    ux_target = np.array([1.0, 0.95, 0.92])
    uy_target = np.array([0.0, -0.05, -0.02])
    ix_pin = G_dense @ ux_target - B_dense @ uy_target
    iy_pin = G_dense @ uy_target + B_dense @ ux_target

    m = Model()
    m.ux = Var('ux', np.ones(nb))
    m.uy = Var('uy', np.zeros(nb))
    m.G = Param('G', csc_array(G_dense), dim=2, sparse=True)
    m.B = Param('B', csc_array(B_dense), dim=2, sparse=True)
    m.ix_pin = Param('ix_pin', ix_pin)
    m.iy_pin = Param('iy_pin', iy_pin)

    # Native syntax: use m.<name>[index] directly in the body. No
    # parallel sympy IndexedBase declarations, no var_map dict.
    i, j = sp.Idx('i'), sp.Idx('j')
    body_re = (
        m.ix_pin[i]
        - sp.Sum(m.G[i, j] * m.ux[j], (j, 0, nb - 1))
        + sp.Sum(m.B[i, j] * m.uy[j], (j, 0, nb - 1))
    )
    body_im = (
        m.iy_pin[i]
        - sp.Sum(m.G[i, j] * m.uy[j], (j, 0, nb - 1))
        - sp.Sum(m.B[i, j] * m.ux[j], (j, 0, nb - 1))
    )

    m.eqn_re = LoopEqn('eqn_re', outer_index=i, n_outer=nb,
                       body=body_re, model=m)
    m.eqn_im = LoopEqn('eqn_im', outer_index=i, n_outer=nb,
                       body=body_im, model=m)

    # Internally the body has been rewritten to sympy IndexedBase
    # form; var_map is auto-populated.
    assert set(m.eqn_re.var_map.keys()) == {
        'ux', 'uy', 'G', 'B', 'ix_pin'}
    assert m.eqn_re.var_map['G'] is m.G
    assert m.eqn_re.var_map['ux'] is m.ux
    assert set(m.eqn_re._sparse_csr.keys()) == {'G', 'B'}

    # The rewritten body no longer contains Solverz IdxVar/IdxPara.
    from Solverz.sym_algebra.symbols import IdxVar, IdxPara
    for atom in m.eqn_re.body.atoms():
        assert not isinstance(atom, (IdxVar, IdxPara))

    spf, y0 = m.create_instance()
    mdl = made_numerical(spf, y0, sparse=True)

    F_val = mdl.F(y0, mdl.p)
    expected_F_re = ix_pin - G_dense.sum(axis=1)
    expected_F_im = iy_pin - B_dense.sum(axis=1)
    np.testing.assert_allclose(
        F_val,
        np.concatenate([expected_F_re, expected_F_im]),
        atol=1e-12,
    )

    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y['ux'], ux_target, atol=1e-8)
    np.testing.assert_allclose(sol.y['uy'], uy_target, atol=1e-8)


def test_loop_eqn_native_solverz_syntax_jit_module():
    """Native syntax end-to-end through the JIT module path.

    Same setup as the native inline test; this one runs the full
    ``module_printer(jit=True)`` → compiled @njit → Newton pipeline
    to confirm the auto-derived var_map flows through unchanged.
    """
    from Solverz import LoopEqn

    nb = 3
    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    B_dense = np.array([
        [ 0.7, -0.2, -0.3],
        [-0.2,  0.6, -0.2],
        [-0.3, -0.2,  0.6],
    ])
    ux_target = np.array([1.0, 0.95, 0.92])
    uy_target = np.array([0.0, -0.05, -0.02])
    ix_pin = G_dense @ ux_target - B_dense @ uy_target
    iy_pin = G_dense @ uy_target + B_dense @ ux_target

    m = Model()
    m.ux = Var('ux', np.ones(nb))
    m.uy = Var('uy', np.zeros(nb))
    m.G = Param('G', csc_array(G_dense), dim=2, sparse=True)
    m.B = Param('B', csc_array(B_dense), dim=2, sparse=True)
    m.ix_pin = Param('ix_pin', ix_pin)
    m.iy_pin = Param('iy_pin', iy_pin)

    i, j = sp.Idx('i'), sp.Idx('j')
    body_re = (
        m.ix_pin[i]
        - sp.Sum(m.G[i, j] * m.ux[j], (j, 0, nb - 1))
        + sp.Sum(m.B[i, j] * m.uy[j], (j, 0, nb - 1))
    )
    body_im = (
        m.iy_pin[i]
        - sp.Sum(m.G[i, j] * m.uy[j], (j, 0, nb - 1))
        - sp.Sum(m.B[i, j] * m.ux[j], (j, 0, nb - 1))
    )
    m.eqn_re = LoopEqn('eqn_re', outer_index=i, n_outer=nb,
                       body=body_re, model=m)
    m.eqn_im = LoopEqn('eqn_im', outer_index=i, n_outer=nb,
                       body=body_im, model=m)

    spf, y0 = m.create_instance()

    with tempfile.TemporaryDirectory() as d:
        printer = module_printer(spf, y0, 'native_jit_sparse',
                                 directory=d, jit=True)
        printer.render()

        sys.path.insert(0, d)
        try:
            import native_jit_sparse
            mdl = native_jit_sparse.mdl
            y0_loaded = native_jit_sparse.y

            sol = nr_method(mdl, y0_loaded)
            np.testing.assert_allclose(sol.y['ux'], ux_target,
                                       atol=1e-8)
            np.testing.assert_allclose(sol.y['uy'], uy_target,
                                       atol=1e-8)

            module_path = os.path.join(d, 'native_jit_sparse',
                                       'num_func.py')
            with open(module_path) as f:
                src = f.read()
            assert '_sz_csr_G_indptr[i]' in src
            assert '_sz_csr_B_indptr[i]' in src
        finally:
            sys.path.remove(d)
            for mod_name in list(sys.modules):
                if mod_name.startswith('native_jit_sparse'):
                    del sys.modules[mod_name]


def test_loop_eqn_ultra_concise_api():
    """Maximally concise API: no ``import sympy``, no
    ``sp.Idx('i')``, no ``sp.Sum(expr, (j, 0, n-1))``, no explicit
    ``n_outer``.

    The user writes a body using Solverz's re-exported
    :func:`Solverz.Idx` / :func:`Solverz.Sum` helpers and
    constructs the ``LoopEqn`` with just an outer ``Idx`` and a
    body expression. Bounds flow through automatically via
    ``Idx('i', n)`` → ``Sum`` range → ``LoopEqn.n_outer``.
    """
    nb = 3
    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    B_dense = np.array([
        [ 0.7, -0.2, -0.3],
        [-0.2,  0.6, -0.2],
        [-0.3, -0.2,  0.6],
    ])
    ux_target = np.array([1.0, 0.95, 0.92])
    uy_target = np.array([0.0, -0.05, -0.02])
    ix_pin = G_dense @ ux_target - B_dense @ uy_target
    iy_pin = G_dense @ uy_target + B_dense @ ux_target

    m = Model()
    m.ux = Var('ux', np.ones(nb))
    m.uy = Var('uy', np.zeros(nb))
    m.G = Param('G', csc_array(G_dense), dim=2, sparse=True)
    m.B = Param('B', csc_array(B_dense), dim=2, sparse=True)
    m.ix_pin = Param('ix_pin', ix_pin)
    m.iy_pin = Param('iy_pin', iy_pin)

    # Bounded indices — range flows automatically.
    i = Idx('i', nb)
    j = Idx('j', nb)

    body_re = (
        m.ix_pin[i]
        - Sum(m.G[i, j] * m.ux[j], j)
        + Sum(m.B[i, j] * m.uy[j], j)
    )
    body_im = (
        m.iy_pin[i]
        - Sum(m.G[i, j] * m.uy[j], j)
        - Sum(m.B[i, j] * m.ux[j], j)
    )

    # No n_outer — LoopEqn auto-infers from i.upper - i.lower + 1.
    m.eqn_re = LoopEqn('eqn_re', outer_index=i, body=body_re, model=m)
    m.eqn_im = LoopEqn('eqn_im', outer_index=i, body=body_im, model=m)

    assert m.eqn_re.n_outer == nb
    assert set(m.eqn_re.var_map.keys()) == {
        'ux', 'uy', 'G', 'B', 'ix_pin'}
    assert set(m.eqn_re._sparse_csr.keys()) == {'G', 'B'}

    spf, y0 = m.create_instance()
    mdl = made_numerical(spf, y0, sparse=True)
    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y['ux'], ux_target, atol=1e-8)
    np.testing.assert_allclose(sol.y['uy'], uy_target, atol=1e-8)


def test_loop_eqn_sum_with_explicit_n():
    """``Sum(expr, j, n)`` form still works with unbounded ``Idx`` —
    user explicitly supplies ``n`` at the Sum call site and
    ``n_outer`` at the LoopEqn call site.
    """
    nb = 3
    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    rhs = G_dense @ np.array([1.0, 2.0, 3.0])

    m = Model()
    m.x = Var('x', np.zeros(nb))
    m.G = Param('G', csc_array(G_dense), dim=2, sparse=True)
    m.rhs = Param('rhs', rhs)

    # Unbounded Idx — user supplies explicit n / n_outer.
    i = Idx('i')
    j = Idx('j')
    body = m.rhs[i] - Sum(m.G[i, j] * m.x[j], j, nb)
    m.eqn = LoopEqn('eqn', outer_index=i, n_outer=nb, body=body, model=m)

    spf, y0 = m.create_instance()
    mdl = made_numerical(spf, y0, sparse=True)
    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y['x'],
                               np.array([1.0, 2.0, 3.0]),
                               atol=1e-8)


def test_loop_eqn_function_dispatch_f_side():
    """Blanket Class C check — every function in ``_FUNCTION_NUMPY_MAP``
    used in a LoopEqn body lowers to the expected ``np.*`` /
    ``SolCF.*`` call and produces the numerically correct F value.

    Picks a handful of unary funcs (``Abs``, ``Sign``, ``heaviside``,
    ``exp``, ``sin``, ``cos``) composed in a single body. F-side only —
    some of these are non-differentiable (``Sign`` / ``heaviside``), so
    Jacobian handling is not exercised here.
    """
    from Solverz import exp as sz_exp

    n = 4
    x_init = np.array([1.0, -2.0, 0.5, -0.1])
    m = Model()
    m.x = Var('x', x_init)
    m.coef = Param('coef', np.array([0.3, 0.4, 0.5, 0.6]))

    i = Idx('i', n)
    # Composite body: |x| + sign(x) * coef + heaviside(x) + exp(coef)
    #               + sin(x) - cos(coef)
    body = (
        Abs(m.x[i])
        + Sign(m.x[i]) * m.coef[i]
        + heaviside(m.x[i])
        + sz_exp(m.coef[i])
        + sin(m.x[i])
        - cos(m.coef[i])
    )
    eqn = LoopEqn('dispatch_eqn', outer_index=i, body=body, model=m)

    # Generated source should contain every mapped call.
    src = eqn.NUM_EQN._loopeqn_source
    assert 'np.abs' in src
    assert 'np.sign' in src
    assert 'SolCF.Heaviside' in src
    assert 'np.exp' in src
    assert 'np.sin' in src
    assert 'np.cos' in src

    # Reference F: hand-computed element-wise using numpy.
    # SolCF.Heaviside convention: H(0) = 1, H(x<0) = 0, H(x>0) = 1.
    def sz_heaviside(x):
        return np.where(x >= 0, 1.0, 0.0)
    coef_v = m.coef.v
    expected = (
        np.abs(x_init)
        + np.sign(x_init) * coef_v
        + sz_heaviside(x_init)
        + np.exp(coef_v)
        + np.sin(x_init)
        - np.cos(coef_v)
    )
    arg_names = sorted(eqn.SYMBOLS.keys())  # ['coef', 'x']
    args = [m.coef.v if n == 'coef' else x_init for n in arg_names]
    F = eqn.NUM_EQN(*args)
    np.testing.assert_allclose(F, expected, atol=1e-12)


def test_loop_eqn_indirect_index_subset_loop():
    """Subset-of-nodes LoopEqn via an int Param index map.

    Pattern: the user wants one equation per source node (not per
    total node). They build an ``int`` Param ``source_idx`` listing
    the source-node indices and write the body using
    ``m.Ts[m.source_idx[i]]`` so the outer index ``i`` walks only the
    source subset.

    This is the enabling primitive for splitting a node-type-
    conditional equation (like the heat-network supply temperature
    mixing, ``heat_network.py:93-120``) into multiple per-type
    LoopEqns — one for source nodes, one for load nodes, one for
    intermediate nodes — each iterating only its own subset. Requires
    the recursive index rewrite in ``_rewrite_solverz_body`` so the
    inner ``IdxPara('source_idx', i)`` flows through as an
    ``IndexedBase`` access.
    """
    from Solverz import LoopEqn

    n_total = 5
    source_idx_v = np.array([0, 3], dtype=int)   # two source nodes
    Ts_init = np.array([305., 315., 320., 335., 340.])
    Tsource_v = np.array([300., 310., 320., 330., 340.])

    m = Model()
    m.Ts = Var('Ts', Ts_init)
    m.Tsource = Param('Tsource', Tsource_v)
    m.source_idx = Param('source_idx', source_idx_v, dim=1)

    # Subset LoopEqn: enforces Ts[source_idx[i]] == Tsource[source_idx[i]]
    i = Idx('i', len(source_idx_v))
    body = m.Ts[m.source_idx[i]] - m.Tsource[m.source_idx[i]]
    eqn = LoopEqn('Ts_source', outer_index=i, body=body, model=m)

    # Generated source should reference the nested index walk.
    src = eqn.NUM_EQN._loopeqn_source
    assert 'Ts[source_idx[i]]' in src
    assert 'Tsource[source_idx[i]]' in src

    arg_names = sorted(eqn.SYMBOLS.keys())  # ['Ts', 'Tsource', 'source_idx']
    arg_values = {'Ts': Ts_init, 'Tsource': Tsource_v,
                  'source_idx': source_idx_v}
    args = [arg_values[n] for n in arg_names]
    F = eqn.NUM_EQN(*args)
    # Expected: [305-300, 335-330] = [5, 5]
    np.testing.assert_allclose(F, Ts_init[source_idx_v]
                               - Tsource_v[source_idx_v],
                               atol=1e-12)


def test_loop_eqn_heat_power_balance_source_split_f_side():
    """Node-type-split port of the heat power balance source-node
    branch (``heat_network.py:180-196``). For a source node ``N``
    the equation is

        phi[N] = Cp / 1e6 * |min[N]| * (Tsource[N] - Tr[N])

    Demonstrates three primitives composing in one body:

    - ``m.Cp`` as a bare (non-indexed) scalar ``Param`` — registered
      in ``var_map`` by the walker without an index, flows into the
      function signature as a normal arg. (Previously a blocker.)
    - ``Abs(m.min[m.source_idx[i]])`` — Class C function dispatch
      stacked on top of recursive indirect indexing.
    - One LoopEqn over the source-node subset via
      ``m.source_idx[i]``, mirroring how the real heat_network
      refactor would split the 5-branch node-type case switch into
      one LoopEqn per node type.

    F-side only — Newton solve on a bilinear body
    (``|min| * (Tsource - Tr)``) needs a J-side extension that's
    still deferred (see issue #128).
    """
    from Solverz import LoopEqn

    n_total = 5
    source_idx_v = np.array([0, 2, 4], dtype=int)
    phi_init = np.array([100., 200., 300., 400., 500.])
    min_init = np.array([5.0, -3.0, 4.0, -7.0, 2.0])
    Tsource_v = np.array([350., 355., 360., 365., 370.])
    Tr_init = np.array([310., 315., 320., 325., 330.])

    m = Model()
    m.phi = Var('phi', phi_init)
    m.min_f = Var('min_f', min_init)
    m.Tr = Var('Tr', Tr_init)
    m.Tsource = Param('Tsource', Tsource_v)
    m.Cp = Param('Cp', 4182.0)
    m.source_idx = Param('source_idx', source_idx_v, dim=1)

    i = Idx('i', len(source_idx_v))
    body = (
        m.phi[m.source_idx[i]]
        - m.Cp / 1e6 * Abs(m.min_f[m.source_idx[i]])
        * (m.Tsource[m.source_idx[i]] - m.Tr[m.source_idx[i]])
    )
    eqn = LoopEqn('phi_source', outer_index=i, body=body, model=m)

    # Every ingredient should have landed in var_map.
    assert set(eqn.var_map.keys()) == {
        'phi', 'min_f', 'Tr', 'Tsource', 'Cp', 'source_idx'}

    # The generated source should emit Cp as a function arg and
    # contain the indirect lookups + np.abs.
    src = eqn.NUM_EQN._loopeqn_source
    assert 'def _loop_eqn_func(Cp,' in src or ', Cp,' in src
    assert 'phi[source_idx[i]]' in src
    assert 'min_f[source_idx[i]]' in src
    assert 'Tsource[source_idx[i]]' in src
    assert 'Tr[source_idx[i]]' in src
    assert 'np.abs' in src

    # F-side check: hand-compute the 3-element residual.
    expected = (
        phi_init[source_idx_v]
        - 4182e-6 * np.abs(min_init[source_idx_v])
        * (Tsource_v[source_idx_v] - Tr_init[source_idx_v])
    )
    arg_names = sorted(eqn.SYMBOLS.keys())
    arg_values = {
        'Cp': m.Cp.v, 'Tr': Tr_init, 'Tsource': Tsource_v,
        'min_f': min_init, 'phi': phi_init,
        'source_idx': source_idx_v,
    }
    args = [arg_values[n] for n in arg_names]
    F = eqn.NUM_EQN(*args)
    np.testing.assert_allclose(F, expected, atol=1e-10)


def test_loop_eqn_sum_with_sign_and_pow_f_side():
    """Mirror of the heat ``loop_pressure`` expression (heat_network.py
    lines 88-90): ``Sum(K[j] * m[j]**2 * Sign(m[j]) * pinloop[j])``.

    The heat loop-pressure equation is actually a SCALAR equation
    (single residual, not one-per-pipe), so LoopEqn doesn't reduce the
    sub-function count for this particular pattern. But it's still a
    useful check that:

    - ``Sign`` dispatch works inside a Sum body.
    - ``Pow`` (``m[j] ** 2``) composes with ``Sign`` through the
      translator.
    - A LoopEqn with ``n_outer = 1`` (single-row) behaves like a
      standard scalar Eqn.
    """
    n_pipe = 3
    m_flow = np.array([6.0, -4.0, 2.0])
    K_val = np.array([0.01, 0.02, 0.015])
    pinloop_val = np.array([1.0, 1.0, 1.0])

    m = Model()
    m.m = Var('m', m_flow)
    m.K = Param('K', K_val)
    m.pinloop = Param('pinloop', pinloop_val)

    # Outer index is unused inside the body (n_outer = 1 — the whole
    # equation is a single scalar). We still need an Idx for the
    # LoopEqn API.
    i = Idx('i', 1)
    j = Idx('j', n_pipe)
    body = Sum(
        m.K[j] * m.m[j] ** 2 * Sign(m.m[j]) * m.pinloop[j],
        j,
    )
    eqn = LoopEqn('loop_pressure', outer_index=i, body=body, model=m)

    src = eqn.NUM_EQN._loopeqn_source
    assert 'np.sign' in src
    assert '** 2.0' in src

    expected = (K_val * m_flow ** 2 * np.sign(m_flow) * pinloop_val).sum()
    arg_names = sorted(eqn.SYMBOLS.keys())  # ['K', 'm', 'pinloop']
    arg_values = {'K': K_val, 'm': m_flow, 'pinloop': pinloop_val}
    args = [arg_values[n] for n in arg_names]
    F = eqn.NUM_EQN(*args)
    assert F.shape == (1,)
    np.testing.assert_allclose(F[0], expected, atol=1e-12)


def test_loop_eqn_pow_body_newton_j3():
    """Phase J3 end-to-end: a body whose Jacobian forces the
    LoopEqnDiff fallback path, and Newton solves to the analytic
    answer.

    Body: ``Vm[i]**2 - rhs[i]``. Differentiating:

        ∂F[i] / ∂Vm[k] = 2 * Vm[i] * δ(k, i)

    which canonicalizes to ``2 * Vm[i] * KroneckerDelta(k, i)`` —
    a ``(delta + 1-D Var factor)`` shape the Phase J2 classifier
    does NOT recognise (Phase J2 handles DiagTermWithSum — the
    inner must be a ``Sum``, not a bare ``Vm[i]``). So the diff
    falls through to the dense LoopEqnDiff kernel built by
    ``build_loop_jac_kernel_source``. The kernel emits a nested
    ``for i: for k:`` with a ``(1.0 if i == k else 0.0)``
    conditional from the ``KroneckerDelta → _translate_loop_body_njit``
    branch added in Phase J3.0.

    Verifies:
    - At least one LoopEqnDiff fallback instance exists after
      ``create_instance``.
    - ``FormJac``'s LoopEqnDiff short-circuit picks up Value0
      from the kernel correctly.
    - Inline ``made_numerical`` + Newton converges to the
      analytic ``Vm = sqrt(rhs)`` within ``atol=1e-8``.

    Polar PF (``Sum(Vm[i]*Vm[j]*cos(Va[i]-Va[j]), j)``) is the
    real target. This test uses a simpler scalar-power body only
    to exercise the pipeline without dragging in gauge-freedom
    issues from the 3-bus PF.
    """
    from Solverz import LoopEqn

    n = 3
    Vm_target = np.array([1.0, 2.0, 3.0])
    rhs_value = Vm_target ** 2   # [1, 4, 9]

    m = Model()
    # Warm-start near but not at target
    m.Vm = Var('Vm', np.array([0.8, 2.3, 2.7]))
    m.rhs = Param('rhs', rhs_value)

    i = Idx('i', n)
    body = m.Vm[i] ** 2 - m.rhs[i]
    m.eqn = LoopEqn('sq_eqn', outer_index=i, body=body, model=m)

    spf, y0 = m.create_instance()

    # At least one derivative should be LoopEqnDiff (Phase J2
    # classifier can't express ``δ(k,i) * 2 * Vm[i]`` as a pure
    # Solverz Mat_Mul/Diag/Para combination).
    from Solverz.equation.eqn import LoopEqnDiff
    loop_diffs = [d for d in m.eqn.derivatives.values()
                  if isinstance(d, LoopEqnDiff)]
    assert len(loop_diffs) == 1, (
        f"Expected 1 LoopEqnDiff fallback, got {len(loop_diffs)}"
    )
    # Kernel source should contain the KroneckerDelta conditional.
    src = loop_diffs[0].kernel_source
    assert 'if i == _sz_loop_dk' in src or 'if _sz_loop_dk == i' in src

    # Structural sparsity must be the 3x3 diagonal (3 positions)
    # and NOT the dense 9-entry fallback. ``compute_loop_jac_sparsity``
    # recognised the ``δ(k,i) * 2 * Vm[i]`` term as contributing
    # only to (0,0), (1,1), (2,2).
    ed = loop_diffs[0]
    assert ed._sparsity_row.tolist() == [0, 1, 2]
    assert ed._sparsity_col.tolist() == [0, 1, 2]

    # The constructed JacBlock should inherit that sparsity —
    # not be marked as a 9-entry dense block.
    spf.FormJac(y0)
    for jbs_row in spf.jac.blocks_sorted.values():
        for var, jb in jbs_row.items():
            if hasattr(jb, '_loop_eqn_diff'):
                assert jb.SpEleSize == 3, (
                    f"LoopEqnDiff JacBlock should have 3 nnz, "
                    f"got {jb.SpEleSize}"
                )
                np.testing.assert_array_equal(jb.CooRow, [0, 1, 2])
                np.testing.assert_array_equal(jb.CooCol, [0, 1, 2])

    mdl = made_numerical(spf, y0, sparse=True)
    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y['Vm'], Vm_target, atol=1e-8)


def test_loop_eqn_pow_body_newton_j3_jit_module():
    """Phase J3.3 end-to-end: render a LoopEqnDiff block through
    the JIT module path and solve via the compiled module.

    Asserts:

    - ``module_printer(jit=True).render()`` succeeds — the
      LoopEqnDiff's dense kernel source lands in ``num_func.py``
      as a top-level ``@njit(cache=True)`` def, and the J_
      wrapper calls it + fancy-indexes the block's structural
      (row, col) positions into ``_data_[addr_slice]``.
    - The generated ``num_func.py`` contains the kernel def line
      and the row/col array loads from ``setting``.
    - Importing the module + running Newton converges to
      ``Vm = sqrt(rhs)``.
    """
    import os as _os
    import sys as _sys
    import tempfile as _tempfile
    from Solverz import LoopEqn

    n = 3
    Vm_target = np.array([1.0, 2.0, 3.0])
    rhs_value = Vm_target ** 2

    m = Model()
    m.Vm = Var('Vm', np.array([0.8, 2.3, 2.7]))
    m.rhs = Param('rhs', rhs_value)

    i = Idx('i', n)
    body = m.Vm[i] ** 2 - m.rhs[i]
    m.eqn = LoopEqn('sq_eqn', outer_index=i, body=body, model=m)

    spf, y0 = m.create_instance()

    with _tempfile.TemporaryDirectory() as d:
        printer = module_printer(spf, y0, 'phase_j3_pow',
                                 directory=d, jit=True)
        printer.render()

        module_path = _os.path.join(d, 'phase_j3_pow', 'num_func.py')
        with open(module_path) as f:
            src = f.read()
        # The sparse kernel is emitted as a @njit def with a flat
        # ``for _sz_idx in range(nnz):`` loop. nnz = 3 because the
        # pow body's Jacobian is the 3-element diagonal — the
        # kernel iterates exactly 3 positions, NOT 9 (dense).
        assert '_sz_loop_jac_kernel_0' in src
        assert 'np.empty(3)' in src
        assert 'for _sz_idx in range(3):' in src
        # Row / col arrays loaded from setting at module level.
        assert '_sz_loop_jac_row_0 = setting["_sz_loop_jac_row_0"]' in src
        assert '_sz_loop_jac_col_0 = setting["_sz_loop_jac_col_0"]' in src
        # The J_ wrapper writes directly to data[addr_slice] with
        # no fancy indexing — the kernel's output is already in
        # csc column-major order matching Value0.tocoo().
        assert '_sz_loop_jac_kernel_0(Vm, _sz_loop_jac_row_0, _sz_loop_jac_col_0)' in src

        _sys.path.insert(0, d)
        try:
            import phase_j3_pow
            mdl = phase_j3_pow.mdl
            y0_loaded = phase_j3_pow.y
            sol = nr_method(mdl, y0_loaded)
            np.testing.assert_allclose(sol.y['Vm'], Vm_target, atol=1e-8)
        finally:
            _sys.path.remove(d)
            for mod_name in list(_sys.modules):
                if mod_name.startswith('phase_j3_pow'):
                    del _sys.modules[mod_name]


def test_loop_eqn_sparsity_is_symbolic_not_numerical():
    """The structural sparsity analysis (``compute_loop_jac_sparsity``)
    must be SYMBOLIC — derived from the canonical diff expression's
    tree structure, not from evaluating the kernel at y0. Otherwise
    a Var starting at zero could trick the analyzer into pruning
    diagonal positions whose derivative value is ``4·x·δ`` and
    happens to be zero at that specific point.

    This test starts ``x = [0, 2, 0]`` for body ``2·x[i]² - rhs[i]``.
    At ``x=0`` positions the diagonal derivative ``4·x`` is
    numerically zero. Verifies:

    - ``ed._sparsity_row / _col`` contain all 3 diagonal
      positions regardless of ``x`` being 0 at (0,0) and (2,2).
    - After ``FormJac``, the JacBlock's ``CooRow`` / ``CooCol`` /
      ``SpEleSize`` still describe the 3-entry diagonal.
    - ``Value0`` is filled via the FormJac perturbation (random
      non-zero Var values) — so none of the diagonal entries
      accidentally collapse to zero in scipy's csc construction,
      but even if they did the structural position would still
      be recorded.
    """
    from Solverz.equation.eqn import LoopEqnDiff

    n = 3
    m = Model()
    # Deliberately zero at rows 0 and 2 so the diagonal derivative
    # 4·x[i]·δ(k,i) is numerically zero at those positions.
    m.x = Var('x', np.array([0.0, 2.0, 0.0]))
    m.rhs = Param('rhs', np.array([-1.0, 4.0, -9.0]))
    i = Idx('i', n)
    body = 2 * m.x[i] ** 2 - m.rhs[i]
    m.eqn = LoopEqn('qq', outer_index=i, body=body, model=m)

    spf, y0 = m.create_instance()
    (ed,) = [d for d in m.eqn.derivatives.values()
             if isinstance(d, LoopEqnDiff)]

    # Structural sparsity is the full diagonal — independent of
    # the Var values ``x = [0, 2, 0]``.
    np.testing.assert_array_equal(ed._sparsity_row, [0, 1, 2])
    np.testing.assert_array_equal(ed._sparsity_col, [0, 1, 2])
    assert ed._nnz == 3

    # After FormJac, the JacBlock inherits the structural
    # sparsity. The perturbed Value0 has all-non-zero diagonal
    # entries because FormJac perturbs Vars to [1, 2).
    spf.FormJac(y0)
    found = False
    for jbs in spf.jac.blocks_sorted.values():
        for var, jb in jbs.items():
            if hasattr(jb, '_loop_eqn_diff'):
                found = True
                np.testing.assert_array_equal(jb.CooRow, [0, 1, 2])
                np.testing.assert_array_equal(jb.CooCol, [0, 1, 2])
                assert jb.SpEleSize == 3
    assert found, "expected at least one LoopEqnDiff-sourced JacBlock"


def test_loop_eqn_trig_diag_newton_j3_j4():
    """Phase J3 fallback + Phase J4 sparse kernel end-to-end on a
    body whose per-row diagonal derivative is a nonlinear trig
    expression.

    Bodies:

        F1[i] = Va[i] * cos(Va[i]) + Vm[i] - r1[i]
        F2[i] = Vm[i] * sin(Vm[i])            - r2[i]

    Two LoopEqns, 3 rows each → 6 residuals in (Va, Vm) ∈ ℝ³×ℝ³.
    No gauge freedom (no angle-differences, no scaling
    invariance), so the 3-bus square system is well-conditioned.

    Interesting derivatives:

    - ``∂F1[i] / ∂Va[k] = (cos(Va[i]) - Va[i]*sin(Va[i])) * δ(k, i)``
      — diagonal block with a nonlinear trig scalar per row.
      Falls through Phase J2's classifier (``other_factors`` in
      the term's anatomy) → LoopEqnDiff fallback → sparse kernel
      with 3 entries, ``for _sz_idx in range(3)``.
    - ``∂F1[i] / ∂Vm[k] = δ(k, i)``
      — the Phase J1 constant identity path.
    - ``∂F2[i] / ∂Vm[k] = (sin(Vm[i]) + Vm[i]*cos(Vm[i])) * δ(k, i)``
      — another Phase J3 fallback diagonal block with trig.
    - ``∂F2[i] / ∂Va[k] = 0`` — pruned by ``is_zero`` before the
      classifier ever runs.

    Newton converges from a nearby warm start to the analytic
    target by construction.
    """
    from Solverz.equation.eqn import LoopEqnDiff

    n = 3
    Va_target = np.array([0.5, 1.0, 1.5])
    Vm_target = np.array([0.8, 1.2, 1.6])
    r1 = Va_target * np.cos(Va_target) + Vm_target
    r2 = Vm_target * np.sin(Vm_target)

    m = Model()
    m.Va = Var('Va', Va_target + np.array([0.1, -0.05, 0.07]))
    m.Vm = Var('Vm', Vm_target + np.array([-0.02, 0.04, -0.03]))
    m.r1 = Param('r1', r1)
    m.r2 = Param('r2', r2)

    i = Idx('i', n)
    body1 = m.Va[i] * cos(m.Va[i]) + m.Vm[i] - m.r1[i]
    body2 = m.Vm[i] * sin(m.Vm[i]) - m.r2[i]
    m.eqn1 = LoopEqn('eqn1', outer_index=i, body=body1, model=m)
    m.eqn2 = LoopEqn('eqn2', outer_index=i, body=body2, model=m)

    spf, y0 = m.create_instance()

    # Sanity: at least one derivative in each equation fell into
    # the Phase J3 LoopEqnDiff fallback (the trig diagonal terms).
    loop_diffs = []
    for eqn in (m.eqn1, m.eqn2):
        for ed in eqn.derivatives.values():
            if isinstance(ed, LoopEqnDiff):
                loop_diffs.append(ed)
    assert len(loop_diffs) >= 2

    # Each fallback block should be 3 entries (the diagonal), not
    # 9 (full dense) — proof that ``compute_loop_jac_sparsity``
    # correctly recognised ``δ(outer, diff) * <scalar>`` as a
    # diagonal-only contribution even when ``<scalar>`` is a
    # nonlinear trig expression.
    for ed in loop_diffs:
        assert ed._nnz == 3, (
            f"Expected 3 nnz for a diagonal block, got {ed._nnz}"
        )
        # The sparse kernel source should iterate exactly 3
        # positions.
        assert 'np.empty(3)' in ed.kernel_source
        assert 'for _sz_idx in range(3):' in ed.kernel_source

    mdl = made_numerical(spf, y0, sparse=True)
    sol = nr_method(mdl, y0)
    # NR default tolerance is 1e-6; allow 1e-5 margin here.
    np.testing.assert_allclose(sol.y['Va'], Va_target, atol=1e-5)
    np.testing.assert_allclose(sol.y['Vm'], Vm_target, atol=1e-5)


def test_loop_eqn_polar_pf_trig_f_side():
    """F-side port of the EPS dyn=False polar power-flow pattern
    (``SolMuseum/ae/eps_network.py:57-75``). Each bus enforces

        P[i] = sum_j Vm[i] Vm[j] (G[i,j] cos(Va[i]-Va[j])
                                  + B[i,j] sin(Va[i]-Va[j]))
        Q[i] = sum_j Vm[i] Vm[j] (G[i,j] sin(Va[i]-Va[j])
                                  - B[i,j] cos(Va[i]-Va[j]))

    **Scope: F-side only.** Verifies ``cos`` / ``sin`` (Solverz's
    ``UniVarFunc`` subclasses) pass through ``_translate_loop_body_njit``
    to ``np.cos`` / ``np.sin`` and the generated callable numerically
    matches numpy's analytical computation. J-side translation of
    bilinear ``Var * Var`` products under trig is a known gap (see
    issue #128) — Newton solve on this pattern is out of scope for
    the current prototype and requires extending
    ``_translate_loop_jac`` to handle 1-D Var references / mixed
    KroneckerDelta terms, which is tracked separately.
    """
    nb = 3
    G_dense = np.array([
        [ 4.5, -1.0, -3.0],
        [-1.0,  3.5, -2.0],
        [-3.0, -2.0,  5.5],
    ])
    B_dense = np.array([
        [ 0.7, -0.2, -0.3],
        [-0.2,  0.6, -0.2],
        [-0.3, -0.2,  0.6],
    ])
    Vm_init = np.array([1.0, 0.98, 0.97])
    Va_init = np.array([0.0, -0.015, -0.030])
    P_offset = np.array([0.5, -0.2, 0.3])
    Q_offset = np.array([0.1, -0.1, 0.05])

    m = Model()
    m.Vm = Var('Vm', Vm_init)
    m.Va = Var('Va', Va_init)
    m.G = Param('G', G_dense, dim=2, sparse=False)
    m.B = Param('B', B_dense, dim=2, sparse=False)
    m.P_inj = Param('P_inj', P_offset)
    m.Q_inj = Param('Q_inj', Q_offset)

    i = Idx('i', nb)
    j = Idx('j', nb)
    body_P = Sum(
        m.Vm[i] * m.Vm[j] * (
            m.G[i, j] * cos(m.Va[i] - m.Va[j])
            + m.B[i, j] * sin(m.Va[i] - m.Va[j])
        ),
        j,
    ) - m.P_inj[i]
    body_Q = Sum(
        m.Vm[i] * m.Vm[j] * (
            m.G[i, j] * sin(m.Va[i] - m.Va[j])
            - m.B[i, j] * cos(m.Va[i] - m.Va[j])
        ),
        j,
    ) - m.Q_inj[i]

    # Build the LoopEqn objects directly without attaching to the
    # Model — the model assembly path will call derive_derivative
    # which currently can't handle the bilinear trig Jacobian.
    eqn_P = LoopEqn('P_eqn', outer_index=i, body=body_P, model=m)
    eqn_Q = LoopEqn('Q_eqn', outer_index=i, body=body_Q, model=m)

    # Generated source must reference np.cos / np.sin.
    src_P = eqn_P.NUM_EQN._loopeqn_source
    assert 'np.cos' in src_P
    assert 'np.sin' in src_P
    src_Q = eqn_Q.NUM_EQN._loopeqn_source
    assert 'np.cos' in src_Q
    assert 'np.sin' in src_Q

    # Call NUM_EQN directly with the Var/Param values and compare
    # against the hand-computed reference.
    expected_P = np.zeros(nb)
    expected_Q = np.zeros(nb)
    for ii in range(nb):
        for jj in range(nb):
            expected_P[ii] += (Vm_init[ii] * Vm_init[jj]
                               * (G_dense[ii, jj]
                                  * np.cos(Va_init[ii] - Va_init[jj])
                                  + B_dense[ii, jj]
                                  * np.sin(Va_init[ii] - Va_init[jj])))
            expected_Q[ii] += (Vm_init[ii] * Vm_init[jj]
                               * (G_dense[ii, jj]
                                  * np.sin(Va_init[ii] - Va_init[jj])
                                  - B_dense[ii, jj]
                                  * np.cos(Va_init[ii] - Va_init[jj])))
    expected_P -= P_offset
    expected_Q -= Q_offset

    # Args are in lex-sorted SYMBOLS order: B, G, P_inj, Va, Vm
    # (for P_eqn) or B, G, Q_inj, Va, Vm (for Q_eqn).
    arg_names = sorted(eqn_P.SYMBOLS.keys())
    arg_values = {
        'B': B_dense, 'G': G_dense,
        'P_inj': P_offset, 'Q_inj': Q_offset,
        'Va': Va_init, 'Vm': Vm_init,
    }
    args_P = [arg_values[n] for n in sorted(eqn_P.SYMBOLS.keys())]
    args_Q = [arg_values[n] for n in sorted(eqn_Q.SYMBOLS.keys())]
    F_P = eqn_P.NUM_EQN(*args_P)
    F_Q = eqn_Q.NUM_EQN(*args_Q)
    np.testing.assert_allclose(F_P, expected_P, atol=1e-12)
    np.testing.assert_allclose(F_Q, expected_Q, atol=1e-12)


def test_loop_eqn_native_rejects_missing_model_attr():
    """If the body references a symbol the model doesn't know about,
    LoopEqn should raise a clear error at construction time pointing
    at the missing attribute.
    """
    from Solverz import LoopEqn

    m = Model()
    m.ux = Var('ux', np.ones(3))
    # m.G NOT defined
    import pytest
    i, j = sp.Idx('i'), sp.Idx('j')

    # Legal IdxVar reference that the model doesn't know about.
    # Create a standalone Var not attached to m.
    stray = Var('stray_coef', np.ones(3))
    body = sp.Sum(stray[j] * m.ux[j], (j, 0, 2))

    with pytest.raises(ValueError, match=r"stray_coef"):
        m.bad_eqn = LoopEqn('bad_eqn', outer_index=i, n_outer=3,
                            body=body, model=m)


def test_canonicalize_kronecker_linear_sum():
    """``canonicalize_kronecker`` must collapse
    ``Sum(δ(k, j) * G[i, j], (j, 0, n-1))`` to ``G[i, k]`` — the
    basic linear-Sum diff pattern.
    """
    from Solverz.equation.loop_jac import canonicalize_kronecker
    import sympy as sp
    from sympy.functions.special.tensor_functions import KroneckerDelta

    i = sp.Idx('i', 5)
    j = sp.Idx('j', 5)
    k = sp.Idx('_sz_loop_dk')
    G = sp.IndexedBase('G')

    raw = sp.Sum(KroneckerDelta(k, j) * G[i, j], (j, 0, 4))
    canonical = canonicalize_kronecker(raw, i, k)
    assert canonical == G[i, k]


def test_canonicalize_kronecker_bare_delta():
    """A bare ``KroneckerDelta(k, outer)`` passes through unchanged —
    no Sum around it, nothing to collapse.
    """
    from Solverz.equation.loop_jac import canonicalize_kronecker
    import sympy as sp
    from sympy.functions.special.tensor_functions import KroneckerDelta

    i = sp.Idx('i', 5)
    k = sp.Idx('_sz_loop_dk')

    raw = KroneckerDelta(k, i)
    canonical = canonicalize_kronecker(raw, i, k)
    assert canonical == KroneckerDelta(k, i)


def test_canonicalize_kronecker_negated_sum():
    """Negation on a Sum: ``-Sum(δ(k,j) * (V_in[i,j] - V_out[i,j]),
    (j, 0, n-1))`` → ``-(V_in[i,k] - V_out[i,k])``. Exercises:

    - Sum over Add linearity splitting into two Sums
    - dummy-matched δ collapse in each sub-Sum
    - Leading ``-1`` propagation through the expression tree
    """
    from Solverz.equation.loop_jac import canonicalize_kronecker
    import sympy as sp
    from sympy.functions.special.tensor_functions import KroneckerDelta

    i = sp.Idx('i', 5)
    j = sp.Idx('j', 5)
    k = sp.Idx('_sz_loop_dk')
    V_in = sp.IndexedBase('V_in')
    V_out = sp.IndexedBase('V_out')

    raw = -sp.Sum((V_in[i, j] - V_out[i, j]) * KroneckerDelta(k, j),
                  (j, 0, 4))
    canonical = canonicalize_kronecker(raw, i, k)
    expected = -(V_in[i, k] - V_out[i, k])
    assert sp.simplify(canonical - expected) == 0


def test_loop_eqn_bilinear_newton_j2():
    """Phase J2 end-to-end: LoopEqn body ``Sum(A[i,j] * x[i] * x[j],
    (j, 0, n-1)) - b[i]`` Newton-solves.

    This is the minimum bilinear body that exercises both new
    classifier shapes at once:

    - ``∂F[i]/∂x[k]`` canonicalizes to
      ``δ(k,i) * Sum(A[i,j]*x[j], j) + A[i,k]*x[i]``
    - First term → ``Diag(Mat_Mul(Para(A), iVar(x)))`` (DiagTermWithSum)
    - Second term → ``Mat_Mul(Diag(iVar(x)), Para(A, dim=2))``
      (RowScale)

    Both Solverz expressions flow through the existing
    ``mutable_mat_analyzer`` → ``diag_term`` / ``row_scale_term``
    decomposition verified by the earlier probe. ``inner_J`` gets
    emitted as a scatter-add kernel; ``J_`` wrapper pre-computes
    ``A @ x`` via ``csc_matvec`` (if ``A`` is sparse) once per
    Newton step and passes it to ``inner_J``. Newton converges to
    the analytic answer.

    Picked a rank-1 ``A`` so every row of ``A @ x`` equals the
    scalar ``sum(x)``, making the analytic solution easy to state:
    choose ``b`` such that ``F[x_target] == 0`` by construction.
    """
    nb = 3
    A_dense = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    x_target = np.array([1.0, 2.0, 3.0])
    # F[i] = x_target[i] * sum(x_target) - b[i]; pick b to make F=0 exactly.
    b_value = x_target * (A_dense @ x_target)

    m = Model()
    # Start off the target so Newton has real work to do, but close
    # enough that the bilinear system still converges in a few steps.
    m.x = Var('x', x_target + np.array([0.3, -0.2, 0.1]))
    m.A = Param('A', A_dense, dim=2, sparse=False)
    m.b = Param('b', b_value)

    i = Idx('i', nb)
    j = Idx('j', nb)
    body = Sum(m.A[i, j] * m.x[i] * m.x[j], j) - m.b[i]
    m.eqn = LoopEqn('bilinear', outer_index=i, body=body, model=m)

    spf, y0 = m.create_instance()
    mdl = made_numerical(spf, y0, sparse=True)
    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y['x'], x_target, atol=1e-8)


def test_loop_eqn_bilinear_newton_j2_sparse():
    """Phase J2 Newton solve with a SPARSE 2-D Param ``A``. Exercises
    the same DiagTerm + RowScale classification but drives the
    downstream ``mutable_mat_analyzer`` into emitting its
    ``SolCF.csc_matvec`` fast-path precompute for ``Diag(A @ x)``
    (runs once per Newton call in the ``J_`` wrapper, not in
    ``inner_J``).
    """
    from scipy.sparse import csc_array as _csc

    nb = 3
    A_dense = np.array([
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [1.0, 0.0, 4.0],
    ])
    x_target = np.array([1.5, 2.5, 0.5])
    b_value = x_target * (A_dense @ x_target)

    m = Model()
    m.x = Var('x', x_target + np.array([0.1, -0.2, 0.05]))
    m.A = Param('A', _csc(A_dense), dim=2, sparse=True)
    m.b = Param('b', b_value)

    i = Idx('i', nb)
    j = Idx('j', nb)
    body = Sum(m.A[i, j] * m.x[i] * m.x[j], j) - m.b[i]
    m.eqn = LoopEqn('bilinear', outer_index=i, body=body, model=m)

    spf, y0 = m.create_instance()
    mdl = made_numerical(spf, y0, sparse=True)
    sol = nr_method(mdl, y0)
    np.testing.assert_allclose(sol.y['x'], x_target, atol=1e-8)


def test_canonicalize_kronecker_bilinear_factor_out_and_collapse():
    """Bilinear body produces a Sum with TWO KroneckerDelta terms:

        Sum(δ(k, i) * G[i, j] * Vm[j]
            + δ(k, j) * G[i, j] * Vm[i], (j, 0, n-1))

    - The ``δ(k, i)`` term: ``i`` is outer, NOT the sum dummy — the
      delta is factored out, leaving the Sum as a template.
    - The ``δ(k, j)`` term: ``j`` is the dummy — collapse via
      substitution, drop the Sum.

    Expected canonical form:

        δ(k, i) * Sum(G[i, j] * Vm[j], (j, 0, n-1)) + G[i, k] * Vm[i]

    This is the key pattern for bilinear ``Var * Var`` bodies (polar
    PF), and the ability to keep the first Sum as a template is
    exactly why we avoid ``.doit()``.
    """
    from Solverz.equation.loop_jac import canonicalize_kronecker
    import sympy as sp
    from sympy.functions.special.tensor_functions import KroneckerDelta

    i = sp.Idx('i', 5)
    j = sp.Idx('j', 5)
    k = sp.Idx('_sz_loop_dk')
    G = sp.IndexedBase('G')
    Vm = sp.IndexedBase('Vm')

    raw = sp.Sum(
        KroneckerDelta(k, i) * G[i, j] * Vm[j]
        + KroneckerDelta(k, j) * G[i, j] * Vm[i],
        (j, 0, 4),
    )
    canonical = canonicalize_kronecker(raw, i, k)

    # The canonicalizer's output is exactly:
    #   δ(k,i) * Sum(G[i,j] * Vm[j], (j,0,4)) + G[i,k] * Vm[i]
    expected = (KroneckerDelta(k, i)
                * sp.Sum(G[i, j] * Vm[j], (j, 0, 4))
                + G[i, k] * Vm[i])
    # Use sp.simplify to normalise ordering (commutative factors).
    assert sp.simplify(canonical - expected) == 0

    # Critical property: the inner ``Sum(G[i,j] * Vm[j], j)`` MUST
    # still be present as a template — it is NOT unrolled into 5
    # explicit terms. This is the whole point of not calling
    # ``.doit()``. Verify by checking the canonical expression still
    # contains a ``Sum`` node.
    assert canonical.has(sp.Sum)


def test_loop_eqn_sparse_walker_rejects_multi_sparse_sum():
    """Phase 1 constraint: a single ``Sum`` may contain at most ONE
    sparse 2-D ``Param`` walker. If a user tries to put ``G`` and
    ``B`` in the same Sum, we raise ``NotImplementedError`` with a
    message pointing at the "Case B / shared skeleton" deferral.
    """
    from Solverz import LoopEqn

    nb = 3
    G_dense = np.eye(nb)
    B_dense = 0.5 * np.eye(nb)

    m = Model()
    m.ux = Var('ux', np.ones(nb))
    m.uy = Var('uy', np.zeros(nb))
    m.G = Param('G', csc_array(G_dense), dim=2, sparse=True)
    m.B = Param('B', csc_array(B_dense), dim=2, sparse=True)
    m.ix_pin = Param('ix_pin', np.zeros(nb))

    i, j = sp.Idx('i'), sp.Idx('j')
    ux_sym = sp.IndexedBase('ux')
    uy_sym = sp.IndexedBase('uy')
    G_sym = sp.IndexedBase('G')
    B_sym = sp.IndexedBase('B')
    ix_pin_sym = sp.IndexedBase('ix_pin')

    # Both G and B in the same Sum body — should fail.
    bad_body = ix_pin_sym[i] - sp.Sum(
        G_sym[i, j] * ux_sym[j] - B_sym[i, j] * uy_sym[j],
        (j, 0, nb - 1),
    )

    import pytest
    with pytest.raises(NotImplementedError, match="Case B"):
        m.bad_eqn = LoopEqn(
            'bad_eqn', outer_index=i, n_outer=nb,
            body=bad_body,
            var_map={'ux': m.ux, 'uy': m.uy, 'G': m.G, 'B': m.B,
                     'ix_pin': m.ix_pin},
        )
