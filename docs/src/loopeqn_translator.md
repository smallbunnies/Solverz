(loopeqn_translator)=

# Appendix — how the `LoopEqn` Jacobian translator works

This page is for readers who want to extend the Jacobian translator
or diagnose why a particular body dropped to the slow path. If
you're only *using* `LoopEqn`, the main guide at {ref}`loopeqn` is
enough — come back here once you hit a term shape that the printer
can't classify.

## Glossary

| Term | Meaning |
|---|---|
| **Phase J1** | Constant Jacobian blocks: `∂F/∂x` has no free Var symbols, so the block value is baked into `_data_` at module-build time and the runtime `J` just slices it. |
| **Phase J2** | Mutable Jacobian blocks classified into `Diag(var)` / `Mat_Mul(Diag(var), Matrix)` / `Mat_Mul(Matrix, Diag(var))` / `Diag(u) @ Matrix @ Diag(v)` (biscale) terms. Each contributes via a compiled scatter-add loop over the matrix's nnz — no `scipy.sparse` construction at call time. |
| **Phase J3** | The fallback: a per-entry Python-loop `LoopEqnDiff` kernel that evaluates the canonical derivative expression at each nnz position. Correct for any shape; slower than J2 because the loop body isn't vectorised. |
| **KD** | Shorthand for `sympy.KroneckerDelta`. `KD(i, j)` is 1 when `i == j`, zero otherwise — the canonicalised form of the `∂x[i]/∂x[k]` derivative sympy produces. |
| **`_LoopJacEye(n)`** | Solverz primitive for the `n × n` identity matrix. Emitted when the canonical form reduces to `KD(outer, diff)`. |
| **`_LoopJacSelectMat(col_map, n_outer, n_diff)`** | Solverz primitive for a sparse selection matrix whose row `i` has a single 1 at column `col_map[i]`. Emitted for `KD(diff, map[outer])`. |
| **CSR walker** | The per-row sparse iteration pattern that the F-side printer uses for `Sum(M[i, j] * x[j], j)` when `M` is declared `sparse=True`. |
| **Biscale** | Phase J2 term shape `Diag(u) @ Matrix @ Diag(v)` where `Matrix` is a constant sparse matrix. Each nnz `(r, c, d)` scatters `u[r] * v[c] * d` to output position `(r, c)`. |

## Supported body shapes (F side)

What the F-side printer (`_translate_loop_body_njit`) will accept
inside a `LoopEqn.body` without raising. If the body stays within
this grammar, `module_printer(jit=True).render()` succeeds.

| Body element | Syntax | Notes |
|---|---|---|
| Direct Var / Param reference | `m.x[i]`, `m.G[i, j]` | `i` / `j` are bounded `Idx` |
| Indirect gather | `m.x[m.map[i]]`, `m.G[m.ns[i], j]` | `map` / `ns` are 1-D int Params; equivalent to `m.x[i]` when `i` came from a non-identity `Set.idx(...)` |
| Bare scalar Param | `m.Cp`, `m.Cp * something` | scalar Param used as coefficient |
| Nested `Sum` over `Idx` | `Sum(body, j)` with `j = Idx('j', n)` | printer emits `for j in range(n)` |
| Sparse 2-D Param walker | `Sum(m.G[i, j] * m.x[j], j)` with `G` declared `sparse=True` | CSR row walk emitted automatically |
| Arithmetic | `+`, `-`, `*`, `/`, `**` | element-wise in the loop body |
| Unary functions | `sin`, `cos`, `exp`, `log`, `sqrt`, `Abs`, `Sign`, `heaviside`, `atan2` | dispatched to `np.*` / `SolCF.*` |

## Phase J2 translator coverage

The canonical derivative of a `LoopEqn.body` is computed via
`sympy.diff`, run through `canonicalize_kronecker` (which pulls `KD`
factors out of `Sum`s where safe and collapses `Sum(f(j) · KD(j, k), j)
→ f(k)` collapses), and then classified term-by-term. Terms that
match one of the shapes below land on the compiled fast path; the
rest fall to Phase J3.

| Shape (after canonicalisation) | Emission | Where (file) |
|---|---|---|
| `KD(outer, diff)` | `_LoopJacEye(n)` — constant identity | `loop_jac.py::_translate_kronecker_delta` |
| `KD(diff, map[outer])` | `_LoopJacSelectMat(map)` — constant selection | same |
| `Indexed(Param, outer, diff)` | `Para(name, dim=2)` — constant 2-D block | `loop_jac.py::_translate_indexed_param` |
| `Indexed(Param, outer, diff) * Indexed(Var, outer)` | `Mat_Mul(Diag(Var), Para)` — row-scale | `_classify_and_translate_term` |
| `Indexed(Param, outer, diff) * Indexed(Var, diff)` | `Mat_Mul(Para, Diag(Var))` — col-scale | same |
| `KD(outer, diff) * Sum(Param[outer, j] * Var[j], j)` | `Diag(Mat_Mul(Para, iVar))` — DiagTerm | `_classify_and_translate_term` |
| `KD(diff, map[outer]) * Sum(...)` with **identity** `map` | same as the direct-KD DiagTerm above | `_classify_and_translate_term` + `_indirect_kd_is_identity` |
| `KD(diff, map[outer]) * Π scalar_factors(outer)` | `Diag(scalar_vec) @ SelectMat(map)` (direct: just `Diag(scalar_vec)`) | `_try_kd_outer_scalars` |
| `Π outer_scalars(outer) * Sum(...)` with a Sum-KD inside | `Diag(outer_vec) @ m_inner` | `_try_outer_scalar_times_sum_kd` |
| `Sum(f(outer, dummy) * KD(diff, map[dummy]), dummy)` identity `map` | bare `Para` / `Mat_Mul(SelectMat_row, Para)` / `Mat_Mul(matrix_expr, Diag(scalar_vec))` | `_try_sum_kd_collapse` |
| same, non-identity injective `map`, single matrix factor | `Mat_Mul(Para, SelectMat_col)` | same |

## Mutable-matrix analyser decompositions

Once a term is classified as Phase J2, it becomes one entry in a
[`MutableMatBlockMapping`](https://github.com/smallbunnies/Solverz/blob/main/Solverz/code_printer/python/module/mutable_mat_analyzer.py)
structure. The analyser recognises four term kinds:

- `diag_terms` — `Diag(inner)`, scatters `inner[i]` to output
  position `(i, i)`.
- `row_scale_terms` — `Mat_Mul(Diag(u), Matrix)`, scatters
  `u[r] · M.data[k]` to `(r, c)` at each matrix nnz.
- `col_scale_terms` — `Mat_Mul(Matrix, Diag(v))`, symmetric.
- `biscale_terms` — `Mat_Mul(Diag(u), Matrix, Diag(v))` in any of
  the N-arg flat, 2-arg nested, or wrapped-`-1` forms, scatters
  `u[r] · v[c] · M.data[k]`.

`Matrix` in any of the above may be a `Para`, a `_LoopJacSelectMat`,
or a `Mat_Mul` chain of either — `_sparse_matrix_nnz` evaluates the
chain to a single COO sparse matrix at analyser time. `_extract_sign_and_core`
strips any numeric coefficient into a stored `sign` field (allowing
`-2 * Diag(x)` to be handled without falling back).

## Not supported (Phase J3 fallback)

The following shapes still hit the Phase J3 `LoopEqnDiff` kernel
(correct but ~3–7× slower than J2 for small `nnz`). Rewrite the
body to avoid them when the inner loop is on the hot path:

- **Multi-arg sympy `Function` without simplification** — e.g. a
  user-defined `sp.Function('f')(x[i], y[i])` whose `sp.diff` keeps
  the multi-arg form. Prefer elementary operations (the built-in
  `atan2` is fine because `sp.diff(atan2(x, y), x)` simplifies to
  `y / (x² + y²)`).
- **Bilinear outer–dummy products inside `Sum`** — e.g.
  `Sum(Vm[i] * Vm[j] * cos(Va[i] - Va[j]), j)` (polar power flow
  in the full form). None of the J2 patterns express two free
  axes under the same `Sum` in their `Diag` / `Mat_Mul` grammar.
  The diagonal derivative part simplifies cleanly, but the
  bilinear off-diagonal falls to J3.
- **Non-identity-map Sum-KD with Var-dependent scalar factors** —
  e.g. `Sum(Sign(ms[orig[p]]) * KD(k, map[p]) * V[i, p], p)` with
  `map` a non-identity injection. The identity-map variant is
  supported (Pattern 4 mutable); the non-identity case needs an
  inverse-map gather the current scatter-add primitives don't
  express.
- **Piecewise / conditional bodies** — `canonicalize_kronecker`
  unwraps the first non-zero branch heuristically, which produces
  the wrong result for non-trivial conditions. Avoid `sp.Piecewise`
  in LoopEqn bodies.
- **Stencil PDE bodies** (`SolPde`-style multi-arg callables) —
  inherently J3. The sparsity analyser still reports the per-cell
  contribution pattern correctly, so the J3 kernel iterates only
  `O(nnz)` entries per Newton step.

## Body-level limitations

- **Per-pipe 1-D Vars with different lengths** (e.g. pre-refactor
  `Tsp_0`, `Tsp_1`, …) cannot be indexed by a `LoopEqn` dummy:
  `sympy.IndexedBase` can select array *elements* but not symbolic
  *names*. The SolMuseum solution is the "flatten" refactor that
  fuses per-pipe states into a single flat `Tsp_all` Var with
  explicit `(pipe_offset, segment)` addressing.
- **Nested `Sum` with reset** is not supported. A body like
  `Sum(Sum(body, (inner, …)), (outer, …))` where the inner sum
  resets each outer iteration cannot be expressed as a flat
  linear-prelude. Split into two bodies or flatten the sum algebra.

## Debugging a fall-to-J3

Set `SZ_DEBUG_PATTERN4MUT=1` or `SZ_DEBUG_ANALYZER=1` in the
environment when rendering to print every term the analyser / Pattern
4 translator sees and accepts or rejects. The output points at the
exact canonical expression that wasn't classifiable, so you can
decide whether to (a) rewrite the body to avoid the shape,
(b) extend the translator with a new pattern, or (c) accept the
Phase J3 fallback.

## Further reading

- Source of the translator: `Solverz/equation/loop_jac.py`.
- Source of the mutable-matrix analyser: `Solverz/code_printer/python/module/mutable_mat_analyzer.py`.
- Tracking issue for J2 extensions: [#133](https://github.com/smallbunnies/Solverz/issues/133).
