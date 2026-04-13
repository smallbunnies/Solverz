# Example: Power Flow with Mat_Mul (Canonical AE)

**What this teaches**: The modern matrix-vector formulation of static power flow on the IEEE case30 network. Uses `Mat_Mul` with sparse `dim=2` `Param`s for the conductance / susceptance submatrices, in **rectangular coordinates** ($e + jf$). This is the form that drives the matrix calculus engine's fast path.

**Why this is canonical**: It's the worked example that motivated the entire matrix-vector codegen (`Mat_Mul`, `csc_matvec`, mutable Jacobian) in Solverz 0.7+. Compare to the for-loop form (`pf_mdl.py` in the same chapter) — same problem, ~6× shorter source, much faster code-gen time, identical numerical answer.

**Solver**: `nr_method` (Newton-Raphson) — converges in ~4 iterations from flat start on case30.

## Setup

`case30.xlsx` is the standard Matpower IEEE 30-bus case. Loaded via SolUtil's `PowerFlow` class for steady-state operating point + bus classification (slack / PV / PQ).

## Code

```python
import os
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_array

from Solverz import Var, Eqn, Model, Param, Mat_Mul, made_numerical, nr_method

# === 1. Load case data (use SolUtil for any real case) ===
sys_data = loadmat("test_pf_jac/pf.mat")
PQ = loadmat("test_pf_jac/pq.mat")

V = sys_data["V"].reshape((-1,))
nb = V.shape[0]
Ybus = sys_data["Ybus"].tocsc()
G_full = Ybus.real
B_full = Ybus.imag

ref = (sys_data["ref"] - 1).reshape((-1,)).tolist()
pv  = (sys_data["pv"]  - 1).reshape((-1,)).tolist()
pq  = (sys_data["pq"]  - 1).reshape((-1,)).tolist()
non_ref = pv + pq
mbase = 100

# Rectangular: V_i = e_i + j f_i
e0 = V.real
f0 = V.imag
Pg = PQ["Pg"].reshape(-1) / mbase
Qg = PQ["Qg"].reshape(-1) / mbase
Pd = PQ["Pd"].reshape(-1) / mbase
Qd = PQ["Qd"].reshape(-1) / mbase
Pinj = Pg - Pd
Qinj = Qg - Qd

# Submatrices for non-reference buses
n_nr = len(non_ref); n_pq = len(pq); n_pv = len(pv)
G_nr = csc_array(G_full[np.ix_(non_ref, non_ref)])
B_nr = csc_array(B_full[np.ix_(non_ref, non_ref)])

e_ref = e0[ref[0]]; f_ref = f0[ref[0]]
G_ref_col = G_full[non_ref, ref[0]].toarray().ravel()
B_ref_col = B_full[non_ref, ref[0]].toarray().ravel()
p_ref = G_ref_col * e_ref - B_ref_col * f_ref
q_ref = B_ref_col * e_ref + G_ref_col * f_ref

pq_in_nr = [non_ref.index(i) for i in pq]
G_pq = csc_array(G_full[np.ix_(pq, non_ref)])
B_pq = csc_array(B_full[np.ix_(pq, non_ref)])
G_pq_ref_col = G_full[pq, ref[0]].toarray().ravel()
B_pq_ref_col = B_full[pq, ref[0]].toarray().ravel()
p_ref_pq = G_pq_ref_col * e_ref - B_pq_ref_col * f_ref
q_ref_pq = B_pq_ref_col * e_ref + G_pq_ref_col * f_ref

# === 2. Build the symbolic Solverz model with Mat_Mul ===
m = Model()

# Unknowns: e, f at non-reference buses (flat start)
m.e = Var("e", np.ones(n_nr))
m.f = Var("f", np.zeros(n_nr))

# Sparse matrix parameters — the fast-path requires dim=2, sparse=True, bare Param
m.G_nr = Param("G_nr", G_nr, dim=2, sparse=True)
m.B_nr = Param("B_nr", B_nr, dim=2, sparse=True)
m.G_pq = Param("G_pq", G_pq, dim=2, sparse=True)
m.B_pq = Param("B_pq", B_pq, dim=2, sparse=True)

# Vector parameters (offset terms from the slack column)
m.p_ref    = Param("p_ref",    p_ref)
m.q_ref    = Param("q_ref",    q_ref)
m.p_ref_pq = Param("p_ref_pq", p_ref_pq)
m.q_ref_pq = Param("q_ref_pq", q_ref_pq)

# Net injection (Pg - Pd, Qg - Qd) at the non-ref buses
m.Pinj = Param("Pinj", Pinj[non_ref])
m.Qinj = Param("Qinj", Qinj[pq])

# Active power balance, all non-ref buses
m.P_eqn = Eqn("P_balance",
              m.e * (Mat_Mul(m.G_nr, m.e) - Mat_Mul(m.B_nr, m.f) + m.p_ref)
            + m.f * (Mat_Mul(m.B_nr, m.e) + Mat_Mul(m.G_nr, m.f) + m.q_ref)
            - m.Pinj)

# Reactive power balance, PQ buses only
e_pq = m.e[pq_in_nr[0]:pq_in_nr[-1] + 1]
f_pq = m.f[pq_in_nr[0]:pq_in_nr[-1] + 1]
m.Q_eqn = Eqn("Q_balance",
              f_pq * (Mat_Mul(m.G_pq, m.e) - Mat_Mul(m.B_pq, m.f) + m.p_ref_pq)
            - e_pq * (Mat_Mul(m.B_pq, m.e) + Mat_Mul(m.G_pq, m.f) + m.q_ref_pq)
            - m.Qinj)

# Voltage magnitude constraint, PV buses
pv_in_nr = [non_ref.index(i) for i in pv]
Vm_pv_sq = np.abs(V[pv]) ** 2
m.Vm_sq = Param("Vm_sq", Vm_pv_sq)
e_pv = m.e[pv_in_nr[0]:pv_in_nr[-1] + 1]
f_pv = m.f[pv_in_nr[0]:pv_in_nr[-1] + 1]
m.V_eqn = Eqn("V_pv", e_pv ** 2 + f_pv ** 2 - m.Vm_sq)

# === 3. Compile + solve ===
spf, y0 = m.create_instance()
mdl = made_numerical(spf, y0, sparse=True)
sol = nr_method(mdl, y0)

# === 4. Reconstruct full voltage vector ===
e_sol = np.zeros(nb); f_sol = np.zeros(nb)
e_sol[ref] = e_ref; f_sol[ref] = f_ref
e_sol[non_ref] = sol.y["e"]
f_sol[non_ref] = sol.y["f"]
V_sol = e_sol + 1j * f_sol

print(f"Converged in {sol.stats.nstep} iterations")
print(f"|V| = {np.abs(V_sol[non_ref])}")
print(f"Va  = {np.angle(V_sol[non_ref])}")
```

## Notes

- **Why rectangular coordinates and not polar**: The polar form $V_i V_j (G \cos(\theta_i - \theta_j) + B \sin(\theta_i - \theta_j))$ doesn't factor cleanly into `Mat_Mul`. Rectangular gives bilinear terms that the matrix calculus engine handles natively.
- **The 4 sparse `Param`s (`G_nr`, `B_nr`, `G_pq`, `B_pq`) all hit the fast path**: each `Mat_Mul(m.G_nr, m.e)` etc. compiles to a `SolCF.csc_matvec` call inside `inner_F`. Confirm by `output_code=True` and reading the generated source — there should be **no** `scipy.sparse` calls in the inner function for these placeholders.
- **Mutable-matrix Jacobian**: the diagonal scaling by `m.e` / `m.f` produces blocks of the form `Diag(e) @ G_nr` (mutable: depends on `e`). Solverz analyzes these as `Mat_Mul(Diag(...), Matrix)` row-scale terms and emits a scatter-add `@njit` kernel — no scipy fancy indexing per call.
- **Switching to `module_printer`**: for production, swap `made_numerical(...)` for `module_printer(...).render()` and import `mdl` from the rendered module. First call is slower (Numba compile); steady-state hot-F per call is sub-microsecond on case30.
- **For larger cases**: the for-loop form (`pf_mdl.py`) generates many scalar `Eqn`s and is **faster to render** for case2000+ at the cost of slower runtime. The `Mat_Mul` form is the right choice for everything ≤ case300; above that, profile both. The cookbook chapter has the decision matrix and the case30 benchmark numbers (~50 µs hot-F, ~55 µs hot-J).

## See also

- Cookbook chapter (full discussion + benchmarks): <https://cookbook.solverz.org/latest/ae/pf/pf.html>
- For-loop form (alternative): `Solverz-Cookbook/docs/source/ae/pf/src/pf_mdl.py`
- Matrix calculus reference: <https://docs.solverz.org/matrix_calculus.html>
- For ill-conditioned cases use `sicnm` instead — see Cookbook `ae/ill_pf/ill_pf.md`.
