(release_notes)=

# Release Notes

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
