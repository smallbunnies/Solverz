(loopeqn)=

# LoopEqn and LoopOde

{class}`~Solverz.equation.eqn.LoopEqn` (and its ODE sibling
{class}`~Solverz.equation.eqn.LoopOde`) express a family of
structurally-identical scalar equations as a **single** parameterised
template. The code printer emits one `@njit`-compiled sub-function
containing an explicit `for` loop, instead of emitting one sub-function
per scalar equation. On large networks (tens to hundreds of buses,
nodes, or pipes) this collapses the Numba LLVM compilation time from
minutes to seconds.

## When to use it

Use `LoopEqn` whenever you're about to write

```python
for i in range(n_bus):
    m.__dict__[f'balance_{i}'] = Eqn(f'balance_{i}', ...)
```

and the per-`i` body is a scalar expression parameterised by `i`. The
three canonical SolMuseum patterns that motivated `LoopEqn` are:

* `eps_network` current / power balance per bus.
* `heat_network` mass-continuity, mixing, and pressure balance per node.
* `gas_network` mass-continuity and pressure boundary per node.

On the Barry Island IES benchmark (1378-DOF DAE, `module_printer(jit=True)`,
M4 Mac, cold Numba cache), replacing the scalar per-node loops with
`LoopEqn`s reduced cold-cache compile time from **303.58 s** (all
scalar `Eqn`s) to **148.65 s** (`−51 %`), with F-eval wall 7.7× faster
and J-eval wall 4× faster. On the larger 8996-DOF "Big" IES system the
`loopeqn=True` render produces a **3.6× faster end-to-end Rodas run**
than the `loopeqn=False` scalar expansion — see the `#134` benchmark
tracker for per-run measurements.

## Quick start — native syntax

The recommended style passes the containing `Model` so `LoopEqn` can
resolve each Solverz `IdxVar` / `IdxPara` reference by name:

```python
import numpy as np
import sympy as sp
from Solverz import Idx, LoopEqn, Model, Param, Sum, Var

nb = 3
G_dense = ...   # (nb, nb) admittance real part
B_dense = ...
ix_pin = ...
iy_pin = ...

m = Model()
m.ux = Var('ux', np.ones(nb))
m.uy = Var('uy', np.zeros(nb))
m.G = Param('G', G_dense, dim=2, sparse=True)
m.B = Param('B', B_dense, dim=2, sparse=True)
m.ix_pin = Param('ix_pin', ix_pin)
m.iy_pin = Param('iy_pin', iy_pin)

i = Idx('i', nb)       # bounded — LoopEqn auto-infers n_outer = nb
j = Idx('j', nb)

m.ix_balance = LoopEqn(
    'ix_balance',
    outer_index=i,
    body=(m.ix_pin[i]
          - Sum(m.G[i, j] * m.ux[j], j)
          + Sum(m.B[i, j] * m.uy[j], j)),
    model=m,                # resolves ``m.G``, ``m.ux`` etc.
)
m.iy_balance = LoopEqn(
    'iy_balance',
    outer_index=i,
    body=(m.iy_pin[i]
          - Sum(m.G[i, j] * m.uy[j], j)
          - Sum(m.B[i, j] * m.ux[j], j)),
    model=m,
)
```

Two blocks — one for the real-current balance, one for the imaginary —
emit two `inner_F` functions instead of `2 * nb` scalar helpers.

{class}`~Solverz.equation.eqn.Idx` is a thin re-export of
`sympy.Idx` that accepts a length directly:
`Idx('i', n)` is equivalent to `sympy.Idx('i', (0, n - 1))`. Sum is
similarly re-exported so you can write `Sum(body, j)` instead of
`sympy.Sum(body, (j, 0, n - 1))` when `j` is a bounded `Idx`.

## Legacy `var_map` syntax

For cases where you want to build the body from bare
`sympy.IndexedBase` objects rather than Solverz symbols, pass a
`var_map` dict keyed by IndexedBase name:

```python
ux_sym = sp.IndexedBase('ux')
G_sym = sp.IndexedBase('G')

body = sp.Sum(G_sym[i, j] * ux_sym[j], (j, 0, nb - 1))

m.ix_balance = LoopEqn(
    'ix_balance',
    outer_index=i,
    body=body,
    n_outer=nb,
    var_map={'ux': m.ux, 'G': m.G},
)
```

The two styles produce byte-identical internal state. `model=` is
preferred for readability and for avoiding the "forgot to add to
var_map" trap when extending a body.

## Sparse walker — automatic CSR matrix iteration

When a `Sum` body contains a sparse 2-D `Param` accessed as
`M[outer, dummy]`, `LoopEqn` pre-computes `M.tocsr()` once at
construction and the printer emits CSR row walking (`O(nnz_per_row)`)
instead of the dense `for j in range(n_cols)` loop. No user action is
required — declare the Param with `sparse=True`:

```python
m.G = Param('G', csc_array(G_dense), dim=2, sparse=True)
# Body unchanged; the printer now emits a CSR walker for the G Sum.
```

Both the inline path (captured as module globals via `exec`) and the
JIT module path (baked into `param_and_setting.pkl` and loaded at
module init) use the same CSR arrays, so the dense layout never has to
live in memory at runtime.

## Inline path is disallowed

As of `0.8.3`, `made_numerical` raises `NotImplementedError` when any
`LoopEqn` is present in the equation system. The inline path runs
LoopEqn's explicit Python `for`-loops at interpreter speed — 3–7×
slower than the lambdify-vectorised legacy scalar-`Eqn` path — so
emitting a LoopEqn model there defeats the purpose. Render through
`module_printer(jit=True)` instead:

```python
from Solverz import module_printer

module_printer(spf, y0, name='my_model', directory='.', jit=True).render()
```

The same guard covers
{meth}`Solverz.equation.equations.Equations.FormPartialJac`.

## LoopOde

`LoopOde` is `LoopEqn`'s ODE counterpart — same template body, same
`@njit` for-loop codegen, but stored under the ODE side of a DAE
system. Use it when the per-index body is a right-hand side of
$\dot{y}_i = f_i(y, p)$ rather than a residual. All substrate and
sparsity machinery is shared with `LoopEqn`.

See the {class}`Solverz.equation.eqn.LoopOde` docstring for details.

## Supported body shapes

The F-side printer walks the LoopEqn body as a pure sympy expression
tree. The following shapes are understood and emit a compiled
`@njit` `inner_F` under `module_printer(jit=True)`:

| Body element | Syntax | Notes |
|---|---|---|
| Direct Var / Param reference | `m.x[i]`, `m.G[i, j]` | `i` / `j` are bounded `Idx` |
| Indirect gather | `m.x[m.map[i]]`, `m.G[m.ns[i], j]` | `map` / `ns` are 1-D `int` Params |
| Bare scalar Param | `m.Cp`, `m.Cp * something` | scalar Param used as coefficient |
| Nested `Sum` over `Idx` | `Sum(body, j)` with `j = Idx('j', n)` | printer emits a `for j in range(n)` |
| `Sum` with bounded Idx | `Sum(body, (j, 0, n-1))` | equivalent sympy form |
| Sparse 2-D Param walker | `Sum(m.G[i, j] * m.x[j], j)` with `G` sparse | emits CSR row-walk automatically |
| Arithmetic | `+`, `-`, `*`, `/`, `**` | element-wise in the loop body |
| Unary functions | `sin`, `cos`, `exp`, `log`, `sqrt`, `Abs`, `Sign`, `heaviside`, `atan2` | dispatched to `np.*` / `SolCF.*` |
| Piecewise / conditionals | not currently in the F-side grammar | raise via user-defined `Function` |

### Jacobian translator coverage (Phase J2)

When a LoopEqn block is differentiated, the canonical expression is
routed through the Phase J2 classifier which turns recognised shapes
into standard Solverz `Diag` / `Mat_Mul` / `Para` expressions.
Anything the classifier can lower to a constant or
Diag/Row-scale/Col-scale/Biscale term lands on the compiled
`_mut_block_` scatter-add fast path. Unrecognised shapes fall to the
Phase J3 Python double-for-loop kernel (`_sz_loop_jac_kernel_`) —
correct, but slower.

The J2 patterns currently recognised (#133):

| Pattern | Shape (after canonicalization) | Emission |
|---|---|---|
| Bare KD | `KD(outer, diff)` | `_LoopJacEye(n)` — constant identity |
| Indirect KD | `KD(diff, map[outer])` | `_LoopJacSelectMat(map)` — constant selection |
| Bare 2-D Param | `Param[outer, diff]` | `Para(name, dim=2)` — constant block |
| Row-scale | `Param[outer, diff] * Var[outer]` | `Diag(Var) @ Para` |
| Col-scale | `Param[outer, diff] * Var[diff]` | `Para @ Diag(Var)` |
| DiagTerm (direct) | `KD(outer, diff) * Sum(Param[outer, j] * Var[j], j)` | `Diag(Mat_Mul(Para, iVar))` |
| DiagTerm (identity indirect) | `KD(diff, map[outer]) * Sum(...)` with identity `map` | same as direct |
| **Pattern 1 DiagSelect** | `KD(diff, map[outer]) * Π scalar_factors(outer)` | `Diag(scalar_vec) @ SelectMat(map)` (or just `Diag(scalar_vec)` for direct KD) |
| **Pattern 2 bilinear mixing** | `Π outer_scalars(outer) * Sum(...)` where Sum matches Pattern 4 | `Diag(outer_vec) @ m_inner` where `m_inner` is Pattern 4's output |
| **Pattern 4 Sum-KD (identity map)** | `Sum(f(outer, dummy) * KD(diff, map[dummy]), dummy)` with identity `map` | bare `Para` / `SelectMat_row @ Para` (constant) or `Mat_Mul` with `Diag(scalar_vec)` (mutable) |
| **Pattern 4 Sum-KD (non-identity injective map)** | same body, map is an injection | `Mat_Mul(Para, SelectMat_col)` (single-matrix body) |

### Not supported (Phase J3 fallback)

The following shapes still hit the Phase J3 LoopEqnDiff kernel
(correct but ~3–7× slower than J2 for small `nnz`). Rewrite the
body to avoid them when the inner loop is on the hot path:

* **Multi-arg sympy `Function` without simplification** — e.g. a
  user-defined `sp.Function('f')(x[i], y[i])` that doesn't reduce
  on `sp.diff`. Prefer elementary operations (`atan2` is fine
  because `sp.diff(atan2(x, y), x)` simplifies).
* **Bilinear outer–dummy products inside `Sum`** — e.g.
  `Sum(Vm[i] * Vm[j] * cos(Va[i] - Va[j]), j)` (polar power flow).
  The Phase J2 classifier cannot express the two-axis product in
  its `Diag` / `Mat_Mul` vocabulary. The diagonal derivative
  collapses cleanly, but the bilinear off-diagonal needs J3.
* **Non-identity-map Sum-KD with Var-dependent scalar factors** —
  e.g. `Sum(Sign(ms[orig[p]]) * KD(k, map[p]) * V[i, p], p)` with
  `map` a non-identity injection. The identity-map variant is
  supported (Pattern 4 mutable); the non-identity case needs an
  inverse-map gather that the current scatter-add primitives
  don't express.
* **Piecewise / conditional bodies** — `canonicalize_kronecker`
  unwraps the first non-zero branch heuristically, which can
  produce the wrong result for non-trivial conditions.
* **Stencil PDE bodies** (`SolPde`-style multi-arg callables) —
  inherently J3. The sparsity analyzer correctly reports the
  per-cell contribution pattern, so the J3 kernel is already O(nnz)
  per Newton step; this is the design endpoint for that shape.

## Body-level limitations (unchanged from prototype)

* **Per-pipe 1-D Vars with different lengths** (e.g. SolMuseum's
  pre-refactor `Tsp_0`, `Tsp_1`, …) cannot be indexed by a LoopEqn
  dummy, because `sympy.IndexedBase` can select array elements but
  not symbolic *names*. The SolMuseum-side solution is a "flatten"
  refactor that fuses per-pipe states into one flat `Tsp_all` Var
  with explicit `(pipe_offset, segment)` addressing; this unlocks
  heat / gas network LoopEqn ports (see the `heat_network` and
  `gas_network` modules for the pattern).

* **Nested `Sum` with reset** is not supported by the current
  translator — a body like
  `Sum(Sum(body, (inner, …)), (outer, …))` where the inner sum
  resets each outer iteration cannot be expressed as a flat
  linear-prelude. Split into two bodies or flatten the sum algebra.

## Further reading

* {class}`Solverz.equation.eqn.LoopEqn` — full API.
* {class}`Solverz.equation.eqn.LoopOde` — ODE variant.
* Tracking issues:
  [`smallbunnies/Solverz#128`](https://github.com/smallbunnies/Solverz/issues/128)
  (prototype), [`#133`](https://github.com/smallbunnies/Solverz/issues/133)
  (J-translator extensions), [`#134`](https://github.com/smallbunnies/Solverz/issues/134)
  (benchmark tracker).
