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

### API Changes

- **`Mat_Mul` + triggerable / time-series params now errors out.**
  Using a `triggerable=True` parameter or a `TimeSeriesParam` in the
  same equation as `Mat_Mul` raises `NotImplementedError` at
  `FormJac` / `create_instance` time. Previously the combination
  silently froze the triggered values inside the generated Jacobian
  kernel and produced wrong Newton steps once the trigger fired.
  Users who hit this restriction should rewrite the matrix assembly
  elementwise.

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
