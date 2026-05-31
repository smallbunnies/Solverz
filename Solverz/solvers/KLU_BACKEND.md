# Solverz KLU backend: design philosophy

This document records why and how the SuiteSparse KLU sparse-LU backend (`klu_backend.py`) is wired into Solverz, and the reasoning behind each decision. It is the rationale companion to the code; the code is the source of truth for the interface.

## Summary

KLU is the default sparse linear-solver backend when SuiteSparse `libklu` is installed, with scipy SuperLU as the automatic fallback. It is bound through a pure-Python `ctypes` layer to an already-installed `libklu`, so Solverz remains a pure-Python package with no compiled extension and no per-platform wheels. When `libklu` is absent, its ABI does not match, or the matrix is complex or dense, the backend transparently uses SuperLU. On the sparse network-DAE Jacobians that Solverz is built for, the production wrapper factorizes and solves roughly 2.1 to 2.8 times faster than SuperLU, with the linear-solve result identical to SuperLU to machine precision under identical step control.

## Why KLU specifically

KLU is the SuiteSparse sparse-LU solver designed by Tim Davis for circuit-simulation matrices: it applies a block-triangular-form (BTF) permutation, an approximate-minimum-degree (AMD) ordering on each diagonal block, and a left-looking LU with partial pivoting. Power-system and other network-DAE Jacobians are structurally the same class of matrix, which is why Andes, Milano's Dome, and PSAT all default to KLU. An earlier trial of UMFPACK in Solverz was slow and not accurate enough on one Apple-silicon build, but UMFPACK is a multifrontal solver tuned for matrices with dense substructure, a different algorithm for a different matrix class; that result does not predict KLU's behavior, and the benchmark below confirms KLU is the right fit.

## Binding strategy: a pure-Python ctypes backend

Three properties drove the choice of a pure-Python ctypes binding over the alternatives.

1. Packaging. A ctypes binding compiles nothing: it `dlopen`s `libklu` at runtime. Solverz therefore ships a single platform-agnostic wheel, and KLU activates only when the library is found. A compiled C or Cython extension would force either a C toolchain at install time or pre-built binary wheels for every platform and operating system, which is the maintenance burden this design avoids.
2. License. KLU is `LGPL-2.1+`, which a permissively licensed package can use through dynamic linking while keeping its own license. The `kvxopt` package, the usual route to KLU from Python, is `GPL-3.0`, so it could only ever be an optional extra. Binding KLU directly inherits only the LGPL obligation.
3. Zero-copy. KLU's C entry points take the matrix as `Ap`, `Ai`, `Ax`, that is column pointers, row indices, and values. These are bit-identical to `scipy.csc_array`'s `indptr`, `indices`, and `data`. The ctypes binding hands those three NumPy buffers straight to KLU with no copy, provided the index width matches, which is int32 for matrices with fewer than 2^31 nonzeros.

The `klu_common` control struct is mirrored as a 24-field `ctypes.Structure`; at import the backend calls `klu_defaults` and checks that the returned values match KLU's documented defaults, disabling itself if the ABI does not match.

## Backend selection

The backend is read from a `contextvars.ContextVar`, so it can be set once for a whole script, scoped to a block, or overridden per solve.

- `Opt(linsolver='klu' | 'superlu')` overrides a single solve. The default `linsolver=None` defers to the global selection.
- `set_linsolver('superlu')` sets the default for the rest of the script.
- `with linsolver('superlu'):` scopes the choice to a block.
- `SOLVERZ_LINSOLVER=superlu` sets the import-time default via the environment.

The default is `'klu'`. Resolution always degrades `'klu'` to `'superlu'` when `libklu` is unavailable, so the default is safe on any machine. A `ContextVar` rather than a plain global keeps this thread- and async-safe and correctly scoped for nested solves, for example a DAE solve whose inner Newton iteration also factorizes.

## Symbolic-cache design

KLU separates the symbolic factorization, the BTF and AMD ordering that depends only on the sparsity pattern, from the numeric factorization that depends on the values. A Solverz model's Jacobian pattern is invariant for the model's lifetime: the code generator bakes fixed coordinate arrays into the generated `J`, so only the values change with `t`, `y`, and `p`. Diagonal row-scaling, fault hooks that add to an existing entry, and time-series parameters all change values, not structure. The iteration matrix the solvers factorize, `c*M - J` for a differential-algebraic system or `J` for an algebraic Newton step, therefore also has an invariant pattern. The symbolic ordering is computed once with `klu_analyze` and reused for every subsequent factorization with `klu_factor`; this reuse is the difference between roughly 0.6 to 0.8 times SuperLU when the ordering is recomputed each step and roughly 2 to 2.8 times SuperLU when it is cached.

A `KLUSymbolic` owns the `klu_symbolic*` and a cheap fingerprint of shape, nonzero count, and column pointers. A `KLUCache` holds one reusable symbolic. Solvers obtain a cache in one of two ways: a local `KLUCache()` created before the stepping loop, or `model_cache(dae)`, which lazily attaches a cache to the model object so a single `cache=model_cache(dae)` at the factorization site reuses the ordering across steps without threading a cache object through the loop. Newton-Raphson uses the same mechanism through `solve(A, b, cache=...)`, reusing the ordering across iterations. On a fingerprint mismatch the ordering is recomputed, so reuse is correct by construction even if a pattern ever changes.

There is no ahead-of-time enumeration of patterns and no dispatch by solver type. The cache starts empty and the ordering is discovered lazily on the first factorization of a run. In practice this collapses to one pattern per model, because every differential-algebraic solver factorizes the same `M`-union pattern regardless of the per-step scalar, and only an algebraic Newton solve uses `J` alone.

## Ecosystem coverage

Every linear-solve site routes through the backend-aware `lu_decomposition` or `solve`, so the whole solver suite honors the selection and degrades to SuperLU when needed: the Rosenbrock and BDF and NDF and Radau and trapezoidal DAE solvers, the ode15s and ode45 paths, the Newton-Raphson and continuous-Newton algebraic solvers, the DAE initial-condition projection, and the semi-implicit continuous Newton method. The complex iteration matrix built by the Radau-style solvers falls back to SuperLU automatically. The one place that remains on SuperLU by design is the partial-decomposition path of the semi-implicit continuous Newton method, which reads the explicit `L`, `U`, and permutation factors that KLU does not expose in the same form.

## Accuracy and parity

Under identical step control, that is a fixed step size so both backends follow the same control flow, the KLU linear-solve result matches SuperLU to machine precision; a 100-step fixed-step run on a stiff network DAE agrees to order 1e-15 in the final state. The wrapper output is bit-identical to raw KLU, so it introduces no accuracy loss of its own.

KLU's default pivot tolerance prefers the diagonal, which gives a slightly larger linear-solve residual than SuperLU's partial pivoting, on the order of 1e-7 against 1e-9 on large cases. This is well within solver tolerances and is corrected by the Newton iteration in implicit solvers. On an adaptive run, KLU and SuperLU can take slightly different accepted-step sequences, after which the trajectories separate at the tolerance level; this is the expected behavior of an adaptive solver under any small perturbation, including a change of tolerance or of machine, and is not a defect of the backend. For applications that require the per-solve residual to match SuperLU's, `klu_decomposition(A, tol=1.0)` selects partial pivoting.

## Benchmark

Factorization of the iteration matrix `W = c*M - J` from three sparse network-DAE Jacobians, scipy SuperLU against KLU in four regimes. The production ctypes wrapper meets or slightly beats the no-conversion ceiling, confirming the Python layer adds no measurable overhead.

| n | nnz | SuperLU factor+solve | KLU via a copy-in binding | KLU zero-copy ceiling | Solverz ctypes wrapper |
| --- | --- | --- | --- | --- | --- |
| 10,459 | 97,273 | 3.29 ms | 1.18x | 2.55x | 2.62x |
| 72,196 | 674,793 | 22.94 ms | 0.93x | 2.08x | 2.12x |
| 158,957 | 1,432,859 | 72.68 ms | 1.28x | 2.47x | 2.79x |

A binding that must copy the matrix into a separate sparse type, rather than passing the scipy CSC buffers directly, loses about half of KLU's advantage to the per-step conversion; the direct ctypes binding avoids that copy. End to end, with the symbolic ordering cached across steps, KLU reduces dynamic-simulation wall time by a factor of 1.4 to 1.6 on stiff network DAEs; the end-to-end factor is smaller than the factorization factor because each step also evaluates the Jacobian and the residual and runs error control, work that is unaffected by the linear solver.

## Scope, limits, and future work

1. Real matrices only. Complex iteration matrices fall back to SuperLU. A `klu_z_*` complex binding is a natural extension.
2. Int32 indices. Matrices with more than 2^31 nonzeros would need KLU's `klu_l_*` long-integer API; the backend raises a clear error in that case rather than silently truncating.
3. `klu_refactor`, which reuses the pivot ordering as well as the symbolic ordering, is not yet used; it could reduce the numeric step further on matrices whose values change little between steps.
4. The largest remaining opportunity is to assemble the iteration matrix once into a persistent structure and scatter new values into its data array each step, which would remove the per-step sparse addition and make buffer identity trivial.

## Usage

Install SuiteSparse so that `libklu` is on the loader path: `brew install suite-sparse` on macOS, `conda install -c conda-forge suitesparse`, or the distribution package on Linux. KLU is then used by default. Set the environment variable `SOLVERZ_LIBKLU` to a full path if automatic discovery does not find the library. To force SuperLU, pass `Opt(linsolver='superlu')`, call `set_linsolver('superlu')`, use `with linsolver('superlu'):`, or set `SOLVERZ_LINSOLVER=superlu`.
