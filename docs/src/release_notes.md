(release_notes)=

# Release Notes

## 0.8.1

Hotfix release addressing correctness findings in the 0.8.0 `Mat_Mul`
mutable-matrix Jacobian code generation. Users of `Mat_Mul` should
upgrade — 0.8.0 has a latent miscompilation when two independent
`Diag(...)` terms land on the same `(i,i)` positions (see Bug Fixes).

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

- **`Mat_Mul` + triggerable 2-D sparse matrix now errors out.** Using
  a `triggerable=True` (or `TimeSeriesParam`) 2-D sparse parameter as
  the **matrix operand** of a `Mat_Mul` raises `NotImplementedError`
  at `FormJac` / `create_instance` time. The Layer-2 scatter-add
  Jacobian kernel bakes the matrix's `.data` array into its compiled
  code at model-build time; a trigger firing at runtime would
  silently be ignored and the Jacobian would freeze on the initial
  values while the residual kept evolving. Users who need a
  triggerable matrix must split the residual and write the affected
  equation in explicit elementwise form (one scalar `Eqn` per row).

  The check is deliberately narrow: a triggerable *vector* or
  *scalar* sitting next to a `Mat_Mul`, or used *inside* the vector
  operand of a `Mat_Mul`, is unaffected and allowed — the `F_` / `J_`
  wrappers read the current parameter value from `p_` on every
  Newton step, so no bake-in can happen. `TimeSeriesParam` is always
  1-D and is therefore always allowed to coexist with `Mat_Mul`.

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
