# SolMuseum refactor plan — flatten per-pipe state Vars

**Status**: deferred — wait until IES benchmark runs through the Phase 1
LoopEqn port on the currently-unblocked patterns, then re-evaluate.

**Scope**: `SolMuseum/dae/heat_network.py` (primary) and
`SolMuseum/dae/gas_network.py` (same pattern). `SolMuseum/ae/eps_network.py`
is *not* affected (no per-pipe state Vars).

---

## Why this refactor exists

LoopEqn's F-printer can cover every for-loop equation pattern in the
three network modules **except** those involving per-pipe 1-D `Var`s:

- `heat_network.py:44-62` declares one `Var(f'Tsp_{j}', ...)` and
  `Var(f'Trp_{j}', ...)` per pipe `j`, each of length `M[j] + 1`. `M[j]`
  varies per pipe (`floor(L[j] / dx)`), so the Vars have different
  lengths.
- `gas_network.py` does the same for each pipe's pressure `p_j` and
  flow `q_j` spatial states.

sympy's `IndexedBase` model maps one symbolic name to one array. When
the LoopEqn translator hits a body like
`Sum(H(m[p]) * |m[p]| * Tsp_p[M[p]], (p, 0, n_pipe - 1))` it has no way
to emit `Tsp_{p}[...]` where the symbolic name itself varies with the
runtime dummy `p`. That's a **language-level** gap, not a translator
bug — there's nothing to parameterise over.

The consequence is concrete: three patterns can't be LoopEqn-ed until
the per-pipe Vars get fused into single flat Vars:

1. **Heat supply / return temperature mixing** (`heat_network.py:93-147`)
   — body references `Tsp_p[M[p]]` (outlet) and `Tsp_p[0]` (inlet) for
   every edge of every node.
2. **Heat mass-continuity pipe-inlet temp BCs** (`heat_network.py:72-81`)
   — two small Eqns per pipe wiring `Tsp_p[0] = Ts[from_node(p)]` /
   `Trp_p[0] = Tr[to_node(p)]`.
3. **Gas node mass continuity** (`gas_network.py:43-68`) — body
   references `q_p[M[p]]` and `q_p[0]` at every in/out edge.

---

## The proposed refactor — fused flat Vars + offset Params

Replace each family of per-pipe Vars with a **single flat `Var`**
whose entries are the concatenation of all per-pipe spatial profiles,
and add precomputed **offset / outlet / inlet** integer Params so the
LoopEqn body can hop into the right segment with a single indirection.

### Heat network changes

**Current (`heat_network.py:44-62`)**:
```python
L = self.df.L
dx = dx
M = np.floor(L / dx).astype(int)
for j in range(self.df.n_pipe):
    # ... compute Tsp0, Trp0 of length M[j] + 1 ...
    m.__dict__['Tsp_' + str(j)] = Var('Tsp_' + str(j), value=Tsp0)
    m.__dict__['Trp_' + str(j)] = Var('Trp_' + str(j), value=Trp0)
```

**Proposed**:
```python
L  = self.df.L
M  = np.floor(L / dx).astype(int)
seg_lens = M + 1                                  # len[p] = segments in pipe p
offsets  = np.concatenate([[0], np.cumsum(seg_lens)])  # len = n_pipe + 1
# offsets[p]             → first (inlet)  segment of pipe p in the flat Var
# offsets[p+1] - 1       → last  (outlet) segment of pipe p in the flat Var
total_len = int(offsets[-1])

# Precompute initial profiles for every pipe, then concatenate.
Tsp0_all = np.empty(total_len)
Trp0_all = np.empty(total_len)
for j in range(self.df.n_pipe):
    # compute Tsp0_j, Trp0_j (same as today, length M[j] + 1 each)
    Tsp0_all[offsets[j]:offsets[j+1]] = Tsp0_j
    Trp0_all[offsets[j]:offsets[j+1]] = Trp0_j

m.Tsp_all = Var('Tsp_all', Tsp0_all)
m.Trp_all = Var('Trp_all', Trp0_all)

# Int Params carrying the per-pipe entry points into the flat Vars.
m.pipe_inlet_idx  = Param('pipe_inlet_idx',  offsets[:-1].astype(int), dim=1)
m.pipe_outlet_idx = Param('pipe_outlet_idx', (offsets[1:] - 1).astype(int), dim=1)
# Optional: per-segment base offset (for PDE interior), if the semi-
# discretization needs it.
m.pipe_seg_base   = Param('pipe_seg_base',   offsets[:-1].astype(int), dim=1)
```

After the refactor, every `Tsp_j[0]` becomes
`m.Tsp_all[m.pipe_inlet_idx[j]]` and every `Tsp_j[M[j]]` becomes
`m.Tsp_all[m.pipe_outlet_idx[j]]`. For pipe-interior references inside
the PDE semi-discretization (each pipe has `M[j] + 1` segments), the
addressing is `m.Tsp_all[m.pipe_seg_base[j] + k]` where `k` is the
segment index within the pipe.

### Gas network changes

Same pattern, just rename `Tsp` / `Trp` to `q` / `p` and the fused Vars
become `m.q_all` / `m.p_all`. The `offsets` / `pipe_inlet_idx` /
`pipe_outlet_idx` Params are identical.

---

## What becomes LoopEqn-able after the refactor

### Heat supply temperature mixing (`heat_network.py:93-120`)

One LoopEqn per node type (source, intermediate, load, slack) via an
int-Param subset map, or one big LoopEqn over all nodes with an
`is_source[node]` mask. Sparse incidence Params `V_in[node, pipe]` and
`V_out[node, pipe]` replace the networkx edge iteration (same recipe
as the Phase 0 mass-continuity test). Each Sum contains exactly one
sparse walker (split into two parallel Sums for `V_in` / `V_out` to
satisfy the current "one walker per Sum" rule).

Pseudocode (intermediate node flavour, showing the key terms):

```python
from Solverz import Idx, Sum, LoopEqn, Abs, heaviside

p = Idx('p', n_pipe)
i = Idx('i', n_node)

body_Ts = m.Ts[i] * (
    m.is_source_mask[i] * Abs(m.min[i])
    + Sum(m.V_in[i, p]  * heaviside(m.m[p]) * Abs(m.m[p]), p)
    + Sum(m.V_out[i, p] * (1 - heaviside(m.m[p])) * Abs(m.m[p]), p)
) - (
    m.is_source_mask[i] * m.Tsource[i] * Abs(m.min[i])
    + Sum(m.V_in[i, p]  * heaviside(m.m[p]) * Abs(m.m[p])
          * m.Tsp_all[m.pipe_outlet_idx[p]], p)
    + Sum(m.V_out[i, p] * (1 - heaviside(m.m[p])) * Abs(m.m[p])
          * m.Tsp_all[m.pipe_inlet_idx[p]], p)
)
m.Ts_mixing = LoopEqn('Ts_mixing', outer_index=i, body=body_Ts, model=m)
```

This reduces ~`n_node` scalar Eqns per side (supply + return) to **2**
LoopEqns. That's the dominant single-term compile-time saving in the
heat submodel.

### Heat pipe-inlet temp BCs (`heat_network.py:72-81`)

Two LoopEqns, one per side, each of length `n_pipe`:

```python
i_p = Idx('i_p', n_pipe)

m.Tsp_inlet_BC = LoopEqn(
    'Tsp_inlet_BC', outer_index=i_p,
    body=m.Tsp_all[m.pipe_inlet_idx[i_p]]
         - m.Ts[m.pipe_from_node[i_p]],
    model=m,
)
m.Trp_inlet_BC = LoopEqn(
    'Trp_inlet_BC', outer_index=i_p,
    body=m.Trp_all[m.pipe_inlet_idx[i_p]]
         - m.Tr[m.pipe_to_node[i_p]],
    model=m,
)
```

where `m.pipe_from_node` and `m.pipe_to_node` are precomputed int
Params storing, for each pipe, the index of its "from" / "to" node in
the node list.

### Heat mass flow continuity (already works)

No change needed. `heat_network.py:65-82` is already reachable today
via `V_in` / `V_out` sparse incidence Params plus `m.m[p]` walking —
covered by Phase 0 `test_loop_eqn_mass_continuity`.

### Heat pipe interior PDE (temperature drop / semi-discretization)

Currently each pipe emits `M[j] + 1` scalar Eqns per timestep (the KT2
or upwind stencil applied to `Tsp_j`). After flattening, this becomes a
single LoopEqn with `n_outer = total_len` (total segments across all
pipes) and a body that references `m.Tsp_all[m.pipe_seg_base[i] + k]`
for the stencil neighbours. Some care is needed at pipe boundaries (the
stencil wraps within a pipe, not across the whole flat Var) — either a
per-pipe mask or a bounded inner Sum keeps boundaries clean.

---

## Risks

1. **Jacobian sparsity regression**. The per-pipe Vars currently give
   Solverz's mutable-matrix analyzer a natural block structure (one
   block per pipe). Fused Vars collapse this into one large column
   group. Need to verify the `mutable_mat_analyzer` still recognises
   the block pattern (it should — the analyzer is structure-agnostic)
   and that `inner_J` stays sparse rather than densifying.
   **Verification**: compare the COO row/col arrays and the generated
   `inner_J` source before/after the refactor; shape must match.

2. **Newton warm-start alignment**. The flat Var's initial value is
   `concatenate([Tsp0_0, Tsp0_1, ...])`. If the per-pipe initial
   profiles were previously loaded from different sources (init
   functions, trigger callbacks, steady-state solver output), the
   flatten must preserve that ordering exactly. Any mismatch ruins the
   first Newton step.
   **Verification**: run the pre-refactor `IES` benchmark, dump
   `y_after_first_Newton_step` and `sol.y` at the end; repeat
   post-refactor; assert element-wise match to `rtol=1e-10`.

3. **Boundary condition DOF counting**. The pipe-inlet-temp BCs
   currently emit `2 * n_pipe` scalar Eqns (one supply-side, one
   return-side per pipe). After the refactor they become two
   `n_pipe`-length LoopEqns. Total equation count must stay the same
   and the variable address layout inside the solver must account for
   the `Tsp_all` / `Trp_all` having length `total_len` rather than
   `n_pipe` separate chunks.
   **Verification**: `assert eqs.eqn_size == eqs.vsize` before and
   after; dump and diff `eqs.var_address` and the COO row/col arrays.

4. **Downstream reader assumptions**. Anything that reads
   `sol.y['Tsp_3']` post-solve (plotting scripts, result-file writers,
   the cookbook IES notebook) now gets nothing — those Vars don't
   exist. Need a compatibility shim that exposes per-pipe slices via
   `sol.y['Tsp_3']` computed from `sol.y['Tsp_all'][offsets[3]:offsets[4]]`,
   or update every reader to use the flat form.
   **Verification**: grep the SolMuseum tree and the cookbook for
   `Tsp_` / `Trp_` / `q_` / `p_` naked references; fix each one.

5. **Existing test suite**. `SolMuseum/tests/` has tests asserting
   specific Var names and shapes. They need to be updated or
   paralleled by a flat-Var equivalent during the transition. Should
   be cheap but must not be skipped.

---

## Rollout plan

**Step 1 — dry run on heat_network only, on a fresh SolMuseum branch.**
Don't touch `gas_network.py` yet. The heat side is larger and has the
IES benchmark as a real workload. Land the flat-Var plumbing + the
LoopEqn ports + compatibility shims for downstream readers.

**Step 2 — verify F/J equivalence pre-refactor vs. post-refactor.**
Run the IES benchmark both ways on a tiny test case and diff:
- Initial `F(y0)` residual (to `atol=1e-12`)
- Initial `J(y0)` as a COO triple (row, col, data to `rtol=1e-10`)
- First Newton step `y1 - y0` (to `atol=1e-10`)
- Final solution after full integration (to `rtol=1e-4`, `atol=1e-5` —
  the current test_ies gate)

If any of the first three fail: bug in the flattening. Bisect by
temporarily disabling individual LoopEqn ports.

**Step 3 — measure compile-time delta.**
Run the cold-cache (fresh `NUMBA_CACHE_DIR=tmpdir`) module build on the
IES DAE both ways. Record:
- `inner_F<N>` count (should drop ~50-70%)
- LLVM compile time (the reason this whole effort exists)
- First Newton iteration wall-clock time (should be unchanged or
  slightly faster due to CSR walking)

Report the numbers in a comment on issue #128 so the refactor's value
is documented.

**Step 4 — repeat for gas_network.** Same refactor pattern, simpler
model.

**Step 5 — update documentation.** Add a note to the Cookbook IES
example explaining the flat-Var convention for heat / gas pipe state.

---

## Out of scope

- **Adding a 2-D `Var` primitive to Solverz core.** That's the
  alternative path to this refactor. It's bigger, touches the code
  printer's address bookkeeping, state vector flattening, Jacobian
  address computation, etc. Maybe worth it eventually, but not for
  LoopEqn's immediate needs — flat Vars give the same numerical
  behaviour with less Solverz core disruption.
- **Changing `Eqn` / `LoopEqn` to accept ragged per-pipe Var lists.**
  Would require LoopEqn to dispatch on runtime dummy into a pool of
  symbolic names, which the current sympy-IndexedBase substrate does
  not allow (it's a language-level mismatch, not an implementation
  detail).
- **Refactoring EPS.** `eps_network.py` has no per-pipe Vars and
  already fits LoopEqn without any changes beyond Phase 0 / Phase 1
  primitives.

---

## Gating: why this is deferred

The F-printer additions landed so far (Phase 0 + the Class C function
pass-through + recursive index rewriting) cover every pattern that
*doesn't* touch per-pipe Vars, in every three network modules. The
realistic Phase 1 benchmark target (`eps_network.dyn=True` via
LoopEqn) is unblocked and can drive the IES compile-time comparison
alone. Once we have data on how big the EPS-side win is — i.e. does
the `inner_F<N>` sub-function count collapse enough to matter — we'll
know whether the heat / gas side flattening is worth the model
refactor cost.

**Decision trigger**: after IES passes with
`eps_network.mdl(dyn=True)` ported to LoopEqn, record the LLVM
compile-time delta. If the heat / gas sub-functions still dominate,
execute this plan. If the saving is already good enough with EPS
alone, defer indefinitely.
