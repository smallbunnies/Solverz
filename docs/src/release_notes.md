(release_notes)=

# Release Notes

## 0.8.3

Safety and correctness hardening for the LoopEqn prototype landing
through its Phase 1 refactors.

### New

- **`Model.add` name-collision detection** (#130). Raises
  `ValueError` when merging a sub-model whose attribute would
  overwrite an existing `Param` / `Var` / `Eqn` with a non-equal value.
  Previously the clash was silent — a real IES integration bug
  (gas- vs heat-network `pipe_from_node` parameters silently
  overwriting each other, producing `|F| ≈ 1.5e6` residuals) motivates
  the guard. Identical values (shared object or value-equal Params like
  a common `Cp`) still merge without error.

- **LoopEqn + inline / partial-Jacobian path guard** (#132, C2 guard
  from #128). `made_numerical` and `Equations.FormPartialJac` now
  raise `NotImplementedError` when the equation system contains any
  `LoopEqn` instance. LoopEqn kernels emit Python `for`-loops that are
  3–7× slower than the lambdify-vectorised legacy `Eqn` path without
  Numba JIT; the guard redirects users to
  `Solverz.module_printer(..., jit=True)`, which is LoopEqn's design
  target. Tests that previously exercised the inline path with
  LoopEqn are migrated to `module_printer(jit=False)`.

### Performance

- **Phase J2 translator extension: Sum-KD → sliced / row-gathered /
  column-permuted Param** (#133 Pattern 4). LoopEqn Jacobian blocks
  of the shape `Sum(f(outer, dummy) * KroneckerDelta(diff, map[dummy]),
  (dummy, lo, hi))` now land on the constant-matrix fast path in
  three configurations:
  * Identity map, bare outer → bare `Para(V, dim=2)`.
  * Identity map, indirect outer (`V[row_map[outer], dummy]`) or
    bare outer with `n_outer < V.shape[0]` → `Mat_Mul(SelectMat_row,
    Para(V))`.
  * Non-identity injective column map → `Mat_Mul(Para(V),
    SelectMat_col)`.

  All three cases keep `is_constant_matrix_deri = True`, so Value0
  is evaluated once at FormJac and baked into `_data_` at module-
  build time — zero per-Newton-iteration J cost versus the previous
  Phase J3 Python double-for-loop kernel.

- **Pattern 3 identity-map indirect KD → DiagTerm** (#133). A
  top-level `KD(diff, map[outer]) * Sum(...)` with identity `map`
  (and `n_outer == n_diff`) now folds into the existing direct-KD
  `Diag(Mat_Mul(Para, iVar))` branch — previously raised
  `NotImplementedError` and dropped to Phase J3.

- **Still-deferred cases for #133**:
  * Pattern 4 with mixed Var-dependent factors (e.g.
    `Sign(ms[orig]) * Tsp_all[inlet] * V[row_map, :]` after KD
    collapse) — needs a vector translator for outer/diff-gathered
    scalars plus a mutable-matrix analyzer extension that accepts
    `_LoopJacSelectMat` in `_classify_matmul`.
  * Pattern 1 DiagSelectTerm and Pattern 2 bilinear mixing — same
    infrastructure dependencies. Tracked for a follow-up release.

### Internal

- The inline printer no longer registers `LoopEqn.NUM_EQN` or
  `LoopEqnDiff._kernel_func_name` callables — that code path is now
  unreachable behind the guard and has been pruned.
- Value0 shape is verified against `(n_outer, n_diff)` before Pattern 4
  emits its translated Param expression, preventing a latent
  `IndexError` when a sparse-pattern row falls outside the LoopEqn's
  equation-block range.

## 0.8.2

**Documentation-only release.** No source-code, test, or behaviour
changes. The PyPI wheel for 0.8.2 is bit-identical to 0.8.1 modulo
the in-tree documentation.

The release rewrites the {ref}`Matrix-Vector Calculus
<matrix_calculus>` chapter to address eight reviewer comments about
misleading language, stale code references, missing terminology, and
content duplication with the
[Solverz Cookbook power-flow chapter](https://cookbook.solverz.org/latest/ae/pf/pf.html).

### Documentation

- **New {ref}`Glossary <matrix-calculus-glossary>` section** at the
  top of the matrix calculus chapter, defining
  {term}`SpMV`, {term}`SpMM`, {term}`CSC / CSR <CSC / CSR>`,
  {term}`@njit`, {term}`scatter-add`, {term}`fancy indexing`,
  {term}`fast path`, {term}`fallback path`, {term}`lambdify`,
  {term}`hot F / hot J <hot F / hot J>`, {term}`cold compile`,
  {term}`LICM`, {term}`inline mode`, and {term}`module printer mode`.
  Every body usage of these terms now hyperlinks to the glossary
  entry.

- **Supported Operations table clarified.** The third column was
  renamed from "Derivative" to "Jacobian block (∂/∂x for vector x)"
  and a paragraph was added above the table explaining that Solverz
  first computes the elementwise vector derivative (e.g. `cos(x)`
  for `sin(x)`) and only inserts the `diag(...)` matrix wrapper at
  Jacobian assembly time. The previous wording risked making
  vector-equation users think `sin(x)` produces a matrix directly.

- **Stale `Mat_Mul` + scipy.sparse claim removed.** The first note
  block under `## Supported Operations` previously said "Mat_Mul
  uses scipy.sparse directly", which was true for 0.8.0 but false
  for 0.8.1 (where the fast path moves the matvec into
  `SolCF.csc_matvec` inside `inner_F`). The note now describes
  the fast / fallback split correctly and points users at the
  Layer 1 discussion.

- **Newton-step language replaced with solver-neutral phrasing.**
  `matrix_calculus.md` is the API reference for Solverz's matrix
  calculus engine, which is shared across AE / FDAE / DAE / ODE.
  Wherever the previous text said "every Newton step assembles
  Jacobian block data", it now says "every `J_(y, p)` call" or
  "every solver step", with one explicit paragraph noting that
  the cost model applies to all `J_` consumers, not just
  algebraic-equation Newton solvers. The remaining "Newton"
  references are intentional and concrete (e.g. naming
  Newton-Raphson as one specific solver among several).

- **Layer 1 narrative refreshed for the second-pass review fixes.**
  References to the obsolete `_is_csc_matvec_fast_path` helper
  were updated to the current `_classify_matmul_placeholders`
  (with inner shape predicate `_shape_is_fast`). A new paragraph
  describes the **dependency-aware demotion** introduced in the
  second-pass review fix: a fast-path candidate consumed by a
  fallback placeholder's matrix or operand expression is demoted
  to fallback so the wrapper never emits a reference to a
  not-yet-materialised placeholder.

- **Fallback path subsections rewritten.** The earlier "Why not
  fancy indexing?" subsection title and prose suggested that
  fancy indexing was an option Solverz "still supports". The new
  title is "Two paths: scatter-add (fast) and fancy indexing
  (fallback)", and the body makes explicit that **both paths
  always co-exist** in every generated module — the runtime picks
  per Jacobian block based on what the symbolic classifier
  recognised at code-gen time. There is no on/off switch.

- **Benchmark environment block added** to the Performance
  subsection: hardware (Apple M4), OS (macOS 26.4), Python
  (3.11.13), library versions (numpy 2.3.3, scipy 1.16.0,
  numba 0.65.0, sympy 1.13.3, Solverz 0.8.1+), and methodology
  (10 warm-up + 5000–20000 timed iterations, median of three
  repeats). Without this metadata the absolute numbers in the
  doc were unreproducible.

- **"When to use Mat_Mul" section slimmed down**, deferring the
  full case30 decision matrix and the known hot-F regression case
  to the [Solverz Cookbook power-flow chapter](https://cookbook.solverz.org/latest/ae/pf/pf.html#performance-comparison-mat_mul-vs-for-loop).
  The Solverz-dev doc is the API reference and now keeps just the
  3-bullet API guidance plus the "matrix shapes that fall out of
  the fast path" reference list. The Cookbook is the right place
  for the case-driven narrative; this avoids contradictions when
  one of the two docs drifts.

The companion **Solverz-Cookbook v0.8.2** release does the same
three things in `pf.md` (terminology cleanup, benchmark environment
block, refined "Newton step" wording on the one line where it
overgeneralised). The Cookbook's heat flow chapter is unchanged.

## 0.8.1

Hotfix release addressing correctness findings in the 0.8.0 `Mat_Mul`
mutable-matrix Jacobian code generation. Users of `Mat_Mul` should
upgrade — 0.8.0 has a latent miscompilation when two independent
`Diag(...)` terms land on the same `(i,i)` positions (see Bug Fixes).
The release also brings a large runtime-performance improvement to
the `Mat_Mul` hot-F path (see Performance below).

### Performance

- **`Mat_Mul` hot F: 4.4× speedup on small networks.** Before 0.8.1
  every `Mat_Mul(A, v)` precompute executed as a `scipy.sparse` SpMV
  in the Python `F_` wrapper (`_sz_mm_N = A @ v`). On small systems
  this was dispatch-bound: each SpMV crossed Python → scipy → C →
  back and cost ≈ 1.5 µs *per call* of pure overhead, regardless of
  how little arithmetic the matrix actually had. Eight SpMVs per
  power-flow `F_()` call added up to ≈ 12 µs of scipy dispatch on
  case30, compared to well under 1 µs of actual matvec work.

  The 0.8.1 code generator now recognises plain sparse `dim=2`
  `Para` matrix operands and emits the matvec **inside** `inner_F`
  using the existing `SolCF.csc_matvec` Numba helper
  (`Solverz/num_api/custom_function.py`). The CSC decomposition
  (`<name>_data` / `<name>_indices` / `<name>_indptr` / `<name>_shape0`)
  that `Solverz/model/basic.py` already emits for every sparse
  `dim=2` `Param` is the direct input for the helper — no new
  `setting` entries, no new helper function.

  Case30 power-flow benchmark (per `F_()` call, Apple M4):

  | Formulation | 0.8.1 baseline | **0.8.1 fast path** | Speedup |
  |---|---:|---:|---:|
  | `Mat_Mul` (rectangular) | 14.1 µs | **3.23 µs** | **4.4×** |
  | For-loop (polar) reference | 1.40 µs | 1.11 µs | — |

  The `Mat_Mul` / polar hot-F ratio drops from **10.1× to 2.9×**.
  `J` call, cold compile, module render, and every other phase
  are unchanged. The remaining 2.9× gap is the structural cost of
  8 `SolCF.csc_matvec` calls + 3 sub-function calls + dispatcher in
  Mat_Mul vs 53 inlined scalar kernels in polar; closing it further
  would require SpMV fusion or switching to CSR format, neither of
  which is in this release. See
  {ref}`When to use (and not use) Mat_Mul <when-to-use-mat_mul>`
  in the matrix calculus guide for a full decision matrix.

- **Fallback path** — `Mat_Mul` placeholders whose matrix operand is
  not a plain sparse `Para` (negated matrices `Mat_Mul(-A, x)`,
  nested matrix expressions, dense `dim=2` params) keep the old
  scipy SpMV path in the wrapper. They are functionally correct
  but do not benefit from the fast path; users who hit performance
  regressions should check whether they can rewrite `-Mat_Mul(A,x)`
  instead of `Mat_Mul(-A,x)`, or declare matrices with
  `sparse=True`.

- **`print_F` dead-load cleanup.** With the fast path in place, the
  `F_` wrapper no longer loads `A = p_["A"]` for sparse `dim=2`
  matrices that are used *only* as fast-path `Mat_Mul` operands —
  previously that line was dead but still emitted. The filter
  inspects each placeholder's `matrix_arg` and only keeps the
  matrix load if at least one fallback `Mat_Mul` needs it.

### Bug Fixes

- **Multiple `Diag` terms now accumulate correctly.** Equations whose
  Jacobians contain two or more independent `Diag` terms sharing
  output positions — e.g. `x*(A@x) + x*(B@x)` producing
  `diag(A@x) + diag(B@x) + diag(x)@A + diag(x)@B` — previously had
  one of the two diagonal contributions silently overwritten in the
  module-printer path. The scatter-add kernel now uses `+=` for every
  term, including diag. Inline mode was already correct. Regression
  test: `test_multi_diag_accumulation`.

- **`Model` no longer crashes on dense `dim=2` parameters.**
  `model.create_instance()` unconditionally tried to decompose every
  `dim=2` parameter into sparse CSC flat arrays (`.data`, `.indices`,
  `.indptr`), which for a dense `ndarray` fed a `memoryview` into
  `Array` and raised `TypeError: Unsupported array type <class
  'memoryview'>`. Decomposition is now restricted to
  `sparse and dim == 2`; dense matrices pass through untouched.

- **Selective `@njit` gating respects sparse parameter content.** The
  `inner_F` / `inner_J` helpers are now generated without `@njit` when
  the generated parameter list contains a sparse `dim=2` param, a
  triggerable param, or a `TimeSeriesParam` — objects Numba cannot
  lower. Pure element-wise models are unaffected.

- **`FormJac` and `JacBlock` agree on the mutable-matrix predicate.**
  The "is this block a mutable-matrix block?" decision is now made
  in a single place using the same criterion on both sides —
  matrix-valued derivative that is not a plain `Para` / `-Para`.
  Previously `FormJac` additionally required the expression to
  contain both `Mat_Mul` and `Diag`, while `JacBlock.is_mutable_matrix`
  did not. The divergence would have let a derivative like
  `Diag(x)` skip the `SpDiag` perturbation step and produce a
  shrunken `CooRow` / `CooCol` at a flat start — the downstream
  scatter-add kernel would then write to fewer output positions
  than the runtime expected. No known model hit this in practice
  but the fix closes the corner case.

### API Changes

- **Time-varying sparse `dim=2` `Param`s are now rejected at
  construction.** Declaring a `Param(..., dim=2, sparse=True,
  triggerable=True, ...)` or a `TimeSeriesParam(..., dim=2,
  sparse=True)` raises `NotImplementedError` at the point of
  construction, regardless of whether the parameter is ever
  referenced in an equation.

  Every Solverz code path that consumes a sparse `dim=2` `Param`
  caches its CSC decomposition (`<name>_data`, `<name>_indices`,
  `<name>_indptr`, `<name>_shape0`) at model-build time: the
  legacy `MatVecMul` pipeline, the new 0.8.1 `Mat_Mul`
  `SolCF.csc_matvec` fast path, and the mutable-matrix Jacobian
  scatter-add kernel all read the frozen flats. A runtime
  `trigger_fun` firing or a `get_v_t(t)` update simply gets lost,
  and the Newton iteration either diverges or silently converges
  to the wrong solution.

  Earlier 0.8.1 drafts narrowed this check to "sparse dim=2
  time-varying Para used as the matrix operand of a `Mat_Mul`",
  catching the combination only at `FormJac` time. That was a
  loophole: legacy `MatVecMul` usage, or any other code path that
  consumes the CSC flats, slipped through. The 0.8.1 policy
  closes the loophole by rejecting the shape at the exact line
  where it is declared — the error message now points at the
  user's source, not a deep internal.

  The check lives in `ParamBase.__init__` and
  `TimeSeriesParam.__init__`, with a backstop in
  `Equations._check_no_timevar_sparse_matrices` (runs on every
  `FormJac` call) for the edge case where a tainted `Param` was
  built via `__new__` + attribute assignment, bypassing the
  `__init__` guard.

  **Allowed alternatives**: triggerable / time-series *vectors*
  or *scalars* sitting next to a `Mat_Mul`, element-wise
  formulations where per-row coefficients are 1-D time-varying
  parameters, and dense `dim=2` parameters (`sparse=False`, which
  takes the `MutableMatJacDataModule` fallback path that
  re-evaluates the full block expression on every `J_()` call and
  tolerates runtime updates). See the
  {ref}`Restrictions section <restrictions>` of the Matrix
  Calculus guide for full workarounds.

- **Reserved symbol prefixes `_sz_mm_` and `_sz_mb_`.** Any user
  symbol (`Var`, `Param`, `iVar`, `Para`, ...) whose name starts with
  either prefix is rejected at construction time with a `ValueError`.
  These prefixes are used by the code generator for Mat_Mul precompute
  helpers and mutable-matrix Jacobian block helpers. The check is
  bypassed internally via `internal_use=True`.

- **Dense `dim=2` params in `Mat_Mul` emit a one-shot `UserWarning`.**
  A parameter declared with `dim=2, sparse=False` and used inside a
  `Mat_Mul` works correctly via the fallback path but forfeits the
  scatter-add fast path. `FormJac` now warns once per offending
  parameter to flag the performance cost. See the new
  {ref}`Restrictions and reserved names <restrictions>` section of
  the matrix calculus guide for migration guidance.

### Documentation

- Extended {ref}`Matrix-Vector Calculus <matrix_calculus>`:
  - New {ref}`Restrictions and reserved names <restrictions>` section
    documenting the three API boundaries introduced in 0.8.1.
  - Code examples updated to show the `_sz_mm_` / `_sz_mb_` helper
    names actually emitted by the code printer.
  - Explicit note about the `+=` accumulation rule for diagonal
    scatter-add terms.
  - New explicit list of the cases that push a mutable-matrix
    block onto the {ref}`fallback path <fallback-path>` — useful
    when a model is slower than expected and you want to know
    whether a block is hitting the fast or slow path.
  - The immutability warning now spells out that the restriction
    only strictly applies to matrices used in the vectorised
    `Mat_Mul` fast path (the fallback path re-evaluates the full
    expression each call and would reflect mutations), but still
    recommends treating all sparse matrix params as immutable.

### Internal cleanup

- Removed the dead `include_sparse_in_list` parameter from
  `print_F` / `print_inner_F` / `print_J` / `print_inner_J` and from
  `_has_sparse_in_param_list`. With the Mat_Mul precompute
  architecture every caller hard-coded `False` and the parameter was
  vestigial.
- Dropped the unused `_var_base_name` / `_var_access_expr` helpers
  from `mutable_mat_analyzer`; they were leftovers from an earlier
  draft of `print_inner_J`.

### Full Changelog

[0.8.0...0.8.1](https://github.com/smallbunnies/Solverz/compare/0.8.0...0.8.1)

## 0.8.0

### Highlights

**Complete Matrix-Vector Calculus** — Solverz now fully supports symbolic
differentiation of mixed matrix-vector equations. Write equations like
`e*(G@e - B@f) + f*(B@e + G@f) - P` and get analytical Jacobians automatically.

**Unified `Mat_Mul` Interface** — `Mat_Mul(A, x)` replaces the legacy `MatVecMul`
as the standard matrix-vector product. It uses `scipy.sparse` directly (faster than
the old `csc_matvec`) and supports full matrix calculus.

### New Features

- **Matrix calculus operators**: `exp`, `sin`, `cos`, `ln`, power (`**`), `transpose`,
  and `Diag` now work inside matrix-vector expressions with automatic differentiation.

  ```python
  from Solverz import Mat_Mul, Var, Param, Model, Eqn
  from Solverz.sym_algebra.functions import exp

  m = Model()
  m.A = Param('A', [[1, 0], [0, 2]], dim=2, sparse=True)
  m.x = Var('x', [1, 1])
  m.f = Eqn('f', exp(Mat_Mul(m.A, m.x)))  # Jacobian: diag(exp(A@x)) @ A
  ```

- **Mutable matrix Jacobian**: Variable-dependent matrix derivatives
  (e.g., `diag(e)@G + diag(f)@B` from power flow equations) are now evaluated
  dynamically at each Newton step. The sparsity pattern is determined at
  initialization and remains fixed; only the data values are updated.

- **Selective Numba `@njit`**: In module mode, Numba compilation is applied
  selectively — equations using `Mat_Mul` run with `scipy.sparse` (fast C-level
  sparse operations), while pure element-wise equations retain `@njit` acceleration.

- **`atan2` symbolic function**: Added `atan2(y, x)` for computing the two-argument
  arctangent in symbolic equations.

- **Plugin-based module discovery**: Third-party numerical modules (e.g., SolMuseum)
  are now discovered via `entry_points(group='solverz.num_api')` instead of
  hard-coded imports. Packages register via `pyproject.toml`
  `[project.entry-points."solverz.num_api"]`. Closes [#118](https://github.com/smallbunnies/Solverz/issues/118).

- **Improved solution stats**: Solvers now record more detailed statistics and
  profiling information in the solution object.

### Bug Fixes

- Stabilized solution slicing and incidence matrix helpers.

### Deprecations

- **`MatVecMul` is deprecated** — use `Mat_Mul(A, x)` instead. `MatVecMul` will
  emit a `DeprecationWarning` when used. It will be removed in a future release.

  ```python
  # Before (deprecated):
  from Solverz import MatVecMul
  m.f = Eqn('f', MatVecMul(m.A, m.x) - m.b)

  # After (recommended):
  from Solverz import Mat_Mul
  m.f = Eqn('f', Mat_Mul(m.A, m.x) - m.b)
  ```

### Documentation

- New: {ref}`Matrix-Vector Calculus <matrix_calculus>` — functionality, mathematical
  background, and application examples (power flow, heat network, nonlinear equations).
- New: {ref}`Extending Matrix Calculus <extend_matrix_calculus>` — developer guide
  for adding new operations to the matrix calculus module.
- Updated: {ref}`Getting Started <gettingstarted>` — matrix equation examples now use
  `Mat_Mul`.

### Full Changelog

[0.7.2...0.8.0](https://github.com/smallbunnies/Solverz/compare/0.7.2...0.8.0)
