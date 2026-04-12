# Solverz Development Guidelines

## Project Overview

Solverz is a Python-based simulation modelling language that provides symbolic modelling interfaces and numerical solvers for algebraic equations (AE), ordinary differential equations (ODE), and differential-algebraic equations (DAE).

## Canonical documentation URLs

Use these URLs whenever linking to Solverz docs from code, tests, comments, commit messages, PR descriptions, or release notes. Do **not** link to the underlying ReadTheDocs hostnames.

- **Solverz reference docs**: <https://docs.solverz.org/>
- **Solverz Cookbook (worked examples)**: <https://cookbook.solverz.org/latest/>

## Development Conventions

- Run tests with: `pytest tests/ Solverz/`
- Use conventional commit format: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`
- Prefer composition over inheritance
- Code style follows existing patterns in the codebase

## Architecture Notes

- **Symbolic layer** (`Solverz/sym_algebra/`): SymPy-based symbolic expressions, functions, and matrix calculus
- **Equation layer** (`Solverz/equation/`): equation definitions, Jacobian assembly, parameter management
- **Code printer** (`Solverz/code_printer/`): generates Python/Numba code for F and J functions
  - `inline/`: direct lambdify (no Numba)
  - `module/`: file-based module generation with optional `@njit`
- **Solvers** (`Solverz/solvers/`): Newton-Raphson, ODE/DAE integrators

### Matrix-Vector Equations

Use `Mat_Mul(A, x)` for matrix-vector products in equations. The matrix parameter should be declared with `dim=2, sparse=True`. The matrix calculus module automatically computes symbolic Jacobians. See `docs/src/matrix_calculus.md` for details.

`MatVecMul` is a legacy interface — new code should use `Mat_Mul`.

## Release Process

- Add release notes to `docs/src/release_notes.md` every time a new version is to be released
- Document new features, breaking changes, deprecations, and bug fixes
