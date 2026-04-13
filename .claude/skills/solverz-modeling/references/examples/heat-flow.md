# Example: District Heating Hydraulic Flow (AE with Mutable Jacobian)

**What this teaches**: Static hydraulic balance for a district-heating pipe network. The model has two equation blocks: a **linear** mass continuity equation (constant Jacobian) and a **nonlinear** loop pressure drop equation that produces a *mutable-matrix* Jacobian — the analyzer recognises it as a row-scale term and emits a scatter-add `@njit` kernel instead of falling back to scipy fancy indexing.

**Why this is canonical**: It's the smallest realistic example that exercises the **mutable Jacobian fast path** (Layer 2 of the matrix calculus engine). The loop pressure term `Mat_Mul(L, K * m * Abs(m))` has a Jacobian block that depends on `m` itself, which is exactly what the mutable-matrix analyzer was built to handle.

**Solver**: `nr_method`. Converges in ~3-5 iterations from a reasonable initial guess (computed by SolUtil's `DhsFlow`).

## Setup

`SolUtil.DhsFlow` runs an alternating IPOPT-hydraulic + Solverz-thermal steady-state solve and exposes the network topology (`hc` dict with `n_node`, `n_pipe`, `G` networkx DiGraph, `K` per-pipe friction coefficients, `pinloop` cycle matrix, `slack_node`).

## Code

```python
import numpy as np
from scipy.sparse import csc_array

from Solverz import Var, Eqn, Model, Param, Mat_Mul, Abs, made_numerical, nr_method
from SolUtil import DhsFlow

# === 1. Load and run steady state ===
df = DhsFlow("case_heat.xlsx")
df.run()
hc = df.hc            # dict: n_node, n_pipe, G, K, pinloop, slack_node, ...

m_init = df.m         # initial pipe flows from steady state
m_inj  = df.minset    # node injections (positive = source, negative = load)

# === 2. Build the Mat_Mul model ===
model = Model()
model.m = Var('m', m_init)

# --- Build the signed node-pipe incidence V (sparse), drop the slack row ---
n_node = hc['n_node']
n_pipe = hc['n_pipe']
V_dense = np.zeros((n_node, n_pipe))
for edge in hc['G'].edges(data=True):
    fnode, tnode, data = edge
    pipe = data['idx']
    V_dense[tnode, pipe] = +1     # flows INTO tnode
    V_dense[fnode, pipe] = -1     # flows OUT of fnode

slack_nodes = sorted(hc['slack_node'].tolist())
skip_node = slack_nodes[0] if slack_nodes else 0
non_slack_rows = [n for n in range(n_node) if n != skip_node]
V_ns = csc_array(V_dense[non_slack_rows, :])

model.V_ns      = Param('V_ns',      V_ns,                  dim=2, sparse=True)
model.m_inj_ns  = Param('m_inj_ns',  m_inj[non_slack_rows])
model.K         = Param('K',         hc['K'])

# --- Equation 1: mass continuity at non-slack nodes ---
# Linear in m → constant Jacobian, fast path
model.mass_balance = Eqn(
    'mass_balance',
    Mat_Mul(model.V_ns, model.m) - model.m_inj_ns)

# --- Equation 2: loop pressure drop ---
# Nonlinear in m → mutable Jacobian (the row-scale fast path)
pinloop = np.atleast_2d(np.asarray(hc['pinloop']))
if pinloop.shape[1] != n_pipe and pinloop.shape[0] == n_pipe:
    pinloop = pinloop.T
nontrivial_rows = [i for i in range(pinloop.shape[0])
                   if np.any(pinloop[i] != 0)]
if nontrivial_rows:
    L_sparse = csc_array(pinloop[nontrivial_rows].astype(np.float64))
    model.L = Param('L', L_sparse, dim=2, sparse=True)
    model.loop_pressure = Eqn(
        'loop_pressure',
        Mat_Mul(model.L, model.K * model.m * Abs(model.m)))

# === 3. Compile + solve ===
spf, y0 = model.create_instance()
mdl = made_numerical(spf, y0, sparse=True)
sol = nr_method(mdl, y0)

print(f"Converged in {sol.stats.nstep} iterations")
print(f"Pipe flows m: {sol.y['m']}")
```

## Notes

- **Why drop one row of `V`**: A pure mass-continuity matrix is rank-deficient (rows sum to zero — total in = total out). Dropping any one row breaks the redundancy. Convention: drop the slack node's row.
- **Why `pinloop` is sparse**: only the pipes belonging to a given fundamental cycle have non-zero coefficients (typically ±1). Cookbook stores it as a `(n_loops × n_pipe)` array; we wrap it in `csc_array` to hit the fast path.
- **The mutable-matrix Jacobian magic**: `Mat_Mul(L, K * m * Abs(m))` symbolically differentiates to (roughly) `L @ Diag(2 * K * Abs(m))`. The matrix calculus engine recognises this as a `Mat_Mul(Matrix, Diag)` shape — a **column-scale term** — and emits a scatter-add `@njit` kernel that updates the Jacobian's nonzero entries in place at every solver step. No scipy fancy indexing per call.
- **What if the analyser can't decompose your block**: the fallback path uses `scipy.sparse` evaluation + fancy indexing. You'll see a `UserWarning` from `analyze_mutable_mat_expr`. Common cause: a term that mixes row-scale + col-scale + diag in one expression. Workaround: split into two `Eqn`s.
- **Element-wise alternative**: writing this with explicit Python loops over nodes and pipes (instead of `Mat_Mul`) gives the same numerical result but produces ~10× more scalar `Eqn`s and ~3-5× longer render time. See `Solverz-Cookbook/docs/source/ae/heat_flow/src/heat_flow_mdl.py` for the side-by-side comparison.

## See also

- Cookbook chapter: `ae/heat_flow/heat_flow.md` (rendered: <https://cookbook.solverz.org/latest/ae/heat_flow/heat_flow.html>)
- For full DHS dynamics (transient temperature in pipes), see `examples/m3b9-dynamics.md` for the DAE solver pattern, then `SolMuseum.dae.heat_network` for the prebuilt block.
- Matrix calculus mutable-Jacobian internals: <https://docs.solverz.org/matrix_calculus.html#two-paths-scatter-add-fast-and-fancy-indexing-fallback>
