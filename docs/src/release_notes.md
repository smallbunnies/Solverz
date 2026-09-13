(release_notes)=

# Release Notes

## 0.11.2

### Changed

- **The residual of a numerical model is evaluated in place through an `out` keyword.** `F(t, y, p, out=buf)` writes the residual into `buf` and returns it, the form of SciML's `f!(du, u, p, t)` in NumPy's `out=` spelling, so a solver that owns its work arrays allocates nothing per evaluation. Without `out` a fresh array is returned. Both `made_numerical` and `module_printer` print this form, and `nAE`, `nFDAE` and `nDAE` give any other residual, such as a user's lambda or a module rendered by an older Solverz, the keyword by copying, so every solver can use the one calling convention. See [#187](https://github.com/smallbunnies/Solverz/issues/187).

- **A `LoopEqn` whose canonical Jacobian is structurally unsound now fails to build instead of warning.** `check_canonical_invariants` reports two states of a canonical `LoopEqn` Jacobian that always mean a defect upstream. One is two free symbols that share a printed name, which the analyzer, since it compares indices by label, reads as one index. The other is a `Sum` dummy that also occurs outside every `Sum`, which makes the generated Jacobian wrong. The check warned, so that a model in either state still ran and could be compared with a finite-difference Jacobian, and Newton or Rodas then ran on a Jacobian that could be wrong. Both states were reachable while one model's symbols could reach another's, and the fixes of [#168](https://github.com/smallbunnies/Solverz/issues/168) and [#175](https://github.com/smallbunnies/Solverz/issues/175) in 0.11.1 closed that path. Nothing in Solverz, SolMuseum or the Cookbook reaches either state, and their test suites pass with the warning turned into an error. The check now raises `UnsoundLoopJacobianError`, a `RuntimeError`, from `create_instance`, and the message names the equation and the variable. `canonical_problems` returns the same descriptions without raising. See [#177](https://github.com/smallbunnies/Solverz/issues/177).

### Performance

- **A sparse `LoopEqn` Jacobian kernel evaluates a row sum once per row and reads each entry of a sparse Param at a precomputed position.** A Jacobian block that `LoopEqn` cannot express through `Mat_Mul` and `Diag` is evaluated by a generated kernel that loops over the nonzeros of the block. When the derivative holds a `Sum`, such as the row sum that the product rule puts on the diagonal of `x_i * Sum_j A[i, j] sin(x_j)`, the kernel ran the loop of the `Sum` at every nonzero of the row and kept its value only at the one where the `KroneckerDelta` in front of it holds, so a row of r stored entries was walked r times. Every entry `A[i, k]` outside a `Sum` was also found by searching row i. The loop of the `Sum` now runs only where its `KroneckerDelta` holds, and the position of each such entry in the CSR data of its Param is computed together with the pattern of the block, with the search kept for an entry whose position depends on a `Sum` dummy. Where the kernel computes a value it performs the same operations in the same order, and on the reproduction of the issue and on the SolUtil power flow of case118, case9241pegase and case_ACTIVSg70k the Jacobian is bit for bit that of 0.11.1, at the start point and at a random point. On the reproduction the Jacobian is 1.9 to 23 times faster for 3 to 81 entries per row, and its cost relative to the residual stays between 2.5 and 4.1 where it grew from 4.9 to 93. On the power flow it is 2.7 to 5.0 times faster, and it takes 1.75 ms against 8.8 ms on case9241pegase and 12.0 ms against 38.4 ms on case_ACTIVSg70k. See [#179](https://github.com/smallbunnies/Solverz/issues/179).
- **`render()` no longer evaluates the Jacobian kernels of a `LoopEqn` in plain Python.** `module_printer(...).render()` calls `Equations.FormJac`, which needs the sparsity pattern of every Jacobian block. A `LoopEqn` block that reaches the generated sparse kernel already holds that pattern, which `compute_loop_jac_sparsity` derived from the canonical expression, and the generated `J_` writes the data of the block at every call. `FormJac` nevertheless ran the kernel twice through its plain-Python source, once in `Fy` and once more at a perturbed point. On a large model these two evaluations took most of the render, because the kernel also repeats its row sums at every nonzero, see [#179](https://github.com/smallbunnies/Solverz/issues/179). `FormJac` now builds the block from the stored pattern with placeholder ones and runs no kernel. `Fy`, `gy` and `fy` take `eval_loop_kernels=False` for this, and by default they still evaluate every derivative. The rendered `num_func.py` and `dependency.py` are byte-identical to those of 0.11.1 for the SolUtil power flow of case118, case9241pegase and case_SyntheticUSA and for the Cookbook IES, legacy and `LoopEqn`. On case118 and case9241pegase `F` and `J` are bitwise identical at the start point and at a perturbed point. The one difference in the rendered module is the initial Jacobian data array. Its entries for these blocks held the kernel values at the perturbed point and now hold ones, and `J_` overwrites them before it reads them. Rendered back to back with 0.11.1, the power flow of case_SyntheticUSA, 82 000 buses, takes 1.8 to 2.5 s against 31 to 39 s, and that of case_ACTIVSg25k 0.47 to 0.50 s against 10 to 13 s. See [#180](https://github.com/smallbunnies/Solverz/issues/180).
- **`create_instance` spent most of its time on the sparsity pattern of the `LoopEqn` Jacobian blocks.** `compute_loop_jac_sparsity` collected the structural nonzeros of a block as Python tuples in a set, walked the rows of a sparse Param in a double Python loop, and sorted the set with a Python key, so every nonzero passed through several Python objects. It now keeps each fragment of the pattern as a pair of `int64` arrays, walks the rows with `np.repeat` and `np.cumsum`, and takes the union once, by sorting the linear index `col * n_outer + row` and dropping repeated neighbours. It does not call `np.unique`, which in NumPy 2.3 hashes the values before it sorts them and took 80 percent of the analysis on the four blocks of the 70 000-bus power flow. The pattern is the same, element for element, on all 176 blocks built by the Solverz, SolMuseum and Cookbook test suites and by the SolUtil power flows of 21 MATPOWER cases, from 9 to 82 000 buses. On the model of [#179](https://github.com/smallbunnies/Solverz/issues/179) with 50 000 unknowns and 9 entries per row, `create_instance` is about 3 times faster and the analysis inside it 20 to 27 times faster. On case9241pegase, case_ACTIVSg25k and case_SyntheticUSA the analysis of the four power-flow blocks is 11 to 14 times faster, timed in one process with both implementations back to back. See [#181](https://github.com/smallbunnies/Solverz/issues/181).
- **The SuperLU backend computes its column ordering once per sparsity pattern.** SuperLU orders the columns of a matrix with COLAMD before it factorizes it. The ordering depends only on the sparsity pattern, which the iteration matrix of a model keeps from one Newton iteration or time step to the next. For the same reason the KLU backend keeps its symbolic analysis on the model, through the cache of [#156](https://github.com/smallbunnies/Solverz/issues/156), while SuperLU computed COLAMD again at every factorization. That cache, `KLUCache`, now also holds the SuperLU ordering. The first factorization of a pattern computes the ordering as before, and every later factorization of the same pattern factorizes the matrix with its columns already in that order and SuperLU's own ordering switched off. `nr_method`, `Rodas`, `Radau`, `sicnm` with full decomposition, `backward_euler`, `implicit_trapezoid` and `fdae_solver` pass the cache, so they reuse the ordering whenever SuperLU runs, which is with `linsolver='superlu'`, without `libklu` as on the Windows CI runners, and when KLU falls back. `Rodas` used to create the cache only for KLU. On the Newton Jacobian of the SolUtil power flow of MATPOWER case9241pegase, SuperLU factorizes in 13.4 ms instead of 19.9 ms, and on 15 iteration matrices of the Cookbook IES, the Cookbook gas network and three SolPSDyn systems a factorization takes 16 to 44 percent less time. Partial pivoting still runs at every factorization, and the columns are eliminated in the order COLAMD chose. SciPy switches SuperLU to its symmetric mode when the ordering is off, and in that mode SuperLU breaks an exact tie between the largest entries of a column on another row than under COLAMD. The solution was the same bit for bit on 8 of these 16 matrices, and on the other 8 it differed by at most 5e-17 of its largest entry, with the fill within 0.3 percent. Newton on the power flow of case118 and case9241pegase takes the same iterations and returns the same solution bit for bit. A change of the pattern, in `indptr` or in `indices`, is detected, and the ordering is then computed again. See [#182](https://github.com/smallbunnies/Solverz/issues/182).
- **A `LoopEqn` evaluates each sine and cosine of a Var once per entry, and by default expands the sine and cosine of a difference of two entries.** A generated kernel evaluated a function of a Var entry wherever its body read it. Inside a `Sum` over the stored entries of a sparse row, `sin(x[j])` was therefore evaluated once per stored entry, and the SolUtil power flow evaluated the cosine and the sine of `Va[i] - Va[j]` once per stored entry of every row. A transcendental function of one entry of a Var is now read from a vector that `inner_F` or `inner_J` computes once per call and passes to every kernel that reads it. The vector holds the same numbers, so this alone leaves F and J unchanged bit for bit. The generated code also evaluates `cos(X[p] - X[q])` and `sin(X[p] - X[q])` of a Var `X` as `cos X[p] cos X[q] + sin X[p] sin X[q]` and `sin X[p] cos X[q] - cos X[p] sin X[q]`, so the kernels of the power flow evaluate no trigonometric function at all. This changes F and J at the level of rounding. On the 21 MATPOWER cases of the power-flow benchmark every residual row moves by at most 1.54 machine epsilons of the sum of the magnitudes of its terms, every Newton solve takes the same number of iterations as with 0.11.1, and the solutions agree within 1.8e-11. `LoopEqn(..., expand_trig=False)` keeps the difference, and the symbolic body and its derivatives are not rewritten. Together with the row sums of [#179](https://github.com/smallbunnies/Solverz/issues/179), on case9241pegase F takes 219 µs against 515 µs, J 0.93 ms against 8.4 ms and a warm Newton solve 36 ms against 83 ms. On case_ACTIVSg70k they take 1.9 ms against 3.8 ms, 8.3 ms against 39 ms and 402 ms against 596 ms. See [#183](https://github.com/smallbunnies/Solverz/issues/183).
- **`render()` no longer evaluates a `Mat_Mul` derivative as a dense matrix.** `Equations.FormJac` evaluated every derivative through its lambdified `NUM_EQN`, where `Diag` prints as `np.diagflat`. A derivative such as `Diag(x) @ A`, which the matrix calculus produces for a Var times a matrix-vector product, therefore built a dense n-by-n matrix and multiplied it with `A` into another. For a derivative that depends on a Var, `FormJac` used that value only to learn that the block is a matrix, and then evaluated the block again with the sparse `SpDiag`. `Fy`, `gy` and `fy` now take `eval_mutable_diag=False` from `FormJac` and skip such a derivative, and `FormJac` learns the same from the `SpDiag` evaluation that it needs anyway. A constant derivative keeps its evaluation. The rendered modules and parameters are identical for the rectangular power flow of case118 and case9241pegase and for the Cookbook IES without `LoopEqn`. On the reproduction of the issue with 8000 unknowns the render takes 0.02 s and 0.21 GB against 0.56 s and 1.68 GB, and building the rectangular power flow of case9241pegase takes 4.9 s against 10.8 s. The rectangular power flow of case_ACTIVSg70k and case_SyntheticUSA, whose render needed a dense matrix of 39 GB and 54 GB and was killed on a laptop with 24 GB, now renders in 1.0 s and 1.2 s with a peak memory of 1.1 GB. See [#185](https://github.com/smallbunnies/Solverz/issues/185).

### Fixed

- **A rendered module returned one shared residual array from every `F_` call.** `print_module_code` emitted a module-level buffer that `F_` wrote into and returned, so two residuals were never valid at the same time and every solver that differences two residuals read a zero derivative. `Rodas` takes `dF/dt` as `(F(t + ddt) - F(t)) / ddt`, which was exactly zero, so on a model with a time-varying input it lost its order while still converging: on a 28-unknown data-center model its accepted-step count scaled as `rtol**(-1/2)` instead of `rtol**(-1/5)`, 14359 steps against 57 at `rtol = 1e-8`. `Radau`, `ode15s`, `adams_bdf` and the `rodas3` interpolant held two residuals in the same way. `made_numerical` was not affected. The buffer was also sized by the number of variables rather than the number of equations. `F_` now returns a fresh array, or writes into the caller's `out`, and the rendered and inline models take the same steps and give the same trajectory bit for bit. A module rendered by an older Solverz keeps the defect until it is rendered again. See [#187](https://github.com/smallbunnies/Solverz/issues/187).
- **The KLU symbolic cache could reuse the analysis of another pattern and solve another matrix.** `klu_decomposition` reuses a cached symbolic analysis when the new matrix has the shape, the number of stored entries and the `indptr` of the matrix it was built for. Above `MATCHING_MIN_N`, 1000 unknowns, the cached analysis also holds the row permutation of the matching and the gather that applies it to the data, both computed from the row indices. A matrix with the same `indptr` and other row indices therefore had its values gathered into the old pattern, and KLU factorized and solved another matrix without an error; on a reproduction with 1200 unknowns the residual was 2.6e-3. Rodas and Radau can meet such a matrix, since SciPy drops every entry of `M - h * gamma * J` that is exactly zero, so the pattern of the iteration matrix can change between two factorizations while `nnz` and `indptr` stay the same. The row indices are now part of the fingerprint, and comparing them takes 0.09 ms for 926 416 entries against 56 ms for the factorization. See [#184](https://github.com/smallbunnies/Solverz/issues/184).

## 0.11.1

### Fixed

- **A `LoopEqn` walker was evaluated more slowly than in 0.10.2 whenever its arrays fit under Numba's freeze limit.** To let Numba cache the kernels of large models, 0.11.0 passed every CSR array of a `LoopEqn` walker, and the row and column arrays of every loop Jacobian kernel, to the compiled functions as arguments. Only an array that Numba will not freeze needs that. Numba compiles a global array into a function as a constant when the array is contiguous and holds at most 1 000 000 bytes, and such a function caches normally; a larger global is embedded by its address instead, and only that disables the cache. For every smaller array the arguments bought nothing. Each call unboxed them, and the compiled code lost the constant it had been optimized against, so the residual evaluation of the SolUtil power flow of MATPOWER case118, where the walkers are nearly all of the work, took 6.33 µs against 5.73 µs on 0.10.2. The renderer now decides array by array. An array that Numba freezes stays a module-level global that the compiled functions read directly, as in 0.10.2, and a larger or non-contiguous one is passed as an argument, as in 0.11.0, so every function stays cacheable. On case118 the residual evaluation takes 5.69 µs, the Jacobian evaluation 50.1 µs against 54.6 µs, and `inner_F` and `inner_J` are back to 21 arguments from 27 and 35. On the Cookbook IES `inner_F` and `inner_J` are back to 213 arguments from 219 and 227 and no compiled function receives an array, but its residual and Jacobian evaluations take the same time before and after, within 1 percent, because its two walkers are a small part of either. Freezing has a price that grows with the array, since the frozen bytes are compiled into every function that reads them and stored again in the cache entry of every caller. On case_ACTIVSg25k, whose arrays are the largest under the limit, the first import takes 3.69 s against 2.89 s and the Numba cache holds 49 MB against 0.94 MB, in exchange for a residual evaluation 4.6 percent and a Jacobian evaluation 10 percent faster at every call; 0.10.2 had the same profile. On case_ACTIVSg70k, whose walker data exceeds the limit, only the 560 kB row pointers become globals, the residual evaluation is 3 percent and the Jacobian evaluation 7 percent faster, and the first import takes 0.3 s longer. The limit is a local variable inside Numba, so Solverz repeats it as `_NUMBA_FREEZE_LIMIT`, and a test that compiles one global array at the limit and one just above it fails if a Numba release moves it. See [#170](https://github.com/smallbunnies/Solverz/issues/170).
- **A model could rewrite the dimension or the value of another model's symbol, or receive that symbol from SymPy's caches.** `Para`, `iVar`, `iAliasVar` and `idx` are SymPy `Symbol`s, which SymPy caches and compares by name alone, but a Solverz symbol also carries `dim`, which says whether it is a vector or a matrix, and the value it was built with, if any. A second construction of one name therefore returned the first object and overwrote its `dim` and value, so building a model with a 2-D `A` turned the `Para('A')` of an earlier 1-D expression into a matrix. Once SymPy had evicted that object from its cache, a second construction received a new object that compared equal to the first, and SymPy's other caches, which are keyed on that equality, returned expressions that still held the first object. `create_instance` then read the old `dim` and failed with `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0`, and an equation system built without a `Model` registered the value of the other symbol. Whether either happened depended on the state of SymPy's caches and therefore on the order in which tests ran. These symbols are now built without SymPy's name cache, their equality and hash include `dim` and the value, and a copied or pickled symbol keeps both. The derivative of the legacy `MatVecMul` rebuilt its matrix from the name alone and relied on the cache for `dim`; it now states `dim=2`. See [#175](https://github.com/smallbunnies/Solverz/issues/175).
- **A model built after another in the same process could receive the indices of the first.** An indexed symbol such as `m.Va[i_q]` is a SymPy `Symbol` whose name prints a `Set` index by its label alone, and SymPy caches symbols by name. Two power-flow builds in one process therefore shared one `Va[i_q]` object, and the second build overwrote the index that the expressions of the first already held. Once SymPy had evicted that object from its cache, the second build received a new one that compared equal to it, and SymPy's other caches, which are keyed on that equality, returned the first build's `sin(Va[i_q] - Va[j])` with the first build's `Set` index inside. The Jacobian of one `LoopEqn` then carried two `SetIdx` of one label, which `check_canonical_invariants` reported on the cookbook IES test. The values stayed correct while both sets had the same name, because the generated code binds a `Set` by its name. A second model whose set had another name failed to build with `Parameter PQ uninitialized`, and whether it failed depended on the state of SymPy's caches. `IdxSymBasic` is now built without SymPy's name cache, and its equality and hash include the index itself, with the token of a `Set` index and the bounds of an `Idx`, so indexed symbols on different indices never meet in a SymPy cache. The base symbol takes part in that equality too, so two indexed symbols also differ when their bases differ in `dim` or value, as for the plain symbols of [#175](https://github.com/smallbunnies/Solverz/issues/175). See [#168](https://github.com/smallbunnies/Solverz/issues/168).

## 0.11.0

### Measured against 0.10.2

Upgrading moves the wall-clock time of a model in two directions. The KLU row matching removes most of the factorization time of a large power-flow Jacobian, while the calling convention of issue 162 makes the residual evaluation of a `LoopEqn` walker slightly slower, and which of the two dominates depends on the model. Both releases were installed side by side and measured back to back on one machine, an Apple M4 with Python 3.11.13, NumPy 2.3.5, SciPy 1.16.3 and Numba 0.65.0, so that Solverz was the only difference. The comparison therefore covers every change listed below, not only the row matching that the first bullet measures on its own.

The power flow is the SolUtil `LoopEqn` formulation of 21 MATPOWER cases, solved by `nr_method` with KLU from the voltages stored in the case file. Each time is the median of five solves.

| case | buses | 0.10.2 | 0.11.0 | gain |
| --- | ---: | ---: | ---: | ---: |
| case_SyntheticUSA | 82 000 | 11.5 s | 826 ms | 13.9 |
| case_ACTIVSg70k | 70 000 | 10.8 s | 723 ms | 15.0 |
| case_ACTIVSg25k | 25 000 | 535 ms | 158 ms | 3.39 |
| case13659pegase | 13 659 | 141 ms | 123 ms | 1.14 |
| case_ACTIVSg10k | 10 000 | 95.0 ms | 57.8 ms | 1.64 |
| case9241pegase | 9 241 | 117 ms | 111 ms | 1.05 |
| case6515rte | 6 515 | 36.0 ms | 31.4 ms | 1.15 |
| case3375wp | 3 374 | 14.0 ms | 11.3 ms | 1.25 |
| case2869pegase | 2 869 | 26.5 ms | 28.9 ms | 0.92 |
| case_ACTIVSg2000 | 2 000 | 16.3 ms | 11.3 ms | 1.45 |
| case1888rte | 1 888 | 6.31 ms | 6.73 ms | 0.94 |
| case1354pegase | 1 354 | 7.58 ms | 8.85 ms | 0.86 |
| case_ACTIVSg500 | 500 | 2.12 ms | 2.01 ms | 1.05 |
| case300 | 300 | 2.14 ms | 2.03 ms | 1.06 |
| case_ACTIVSg200 | 200 | 0.837 ms | 0.832 ms | 1.01 |
| case118 | 118 | 0.640 ms | 0.614 ms | 1.04 |
| case57 | 57 | 0.495 ms | 0.400 ms | 1.24 |
| case39 | 39 | does not build | 0.218 ms | |
| case30 | 30 | 0.422 ms | 0.330 ms | 1.28 |
| case14 | 14 | 0.259 ms | 0.201 ms | 1.29 |
| case9 | 9 | 0.378 ms | 0.297 ms | 1.27 |

The gain is largest where the factorization dominates. On case_SyntheticUSA one Newton iteration factorizes the Jacobian in 63.6 ms against 1.87 s, a factor of 29, while the residual evaluation takes 5.13 ms against 5.51 ms and the Jacobian evaluation 53.7 ms against 59.0 ms. Three cases between 1 000 and 3 000 buses solve more slowly on 0.11.0; their gains are 0.86 for case1354pegase, 0.92 for case2869pegase and 0.94 for case1888rte. case39 does not build on 0.10.2 because of [#151](https://github.com/smallbunnies/Solverz/issues/151). For each of the twenty cases that both releases build, the Jacobian carries the same number of nonzeros on both, so the sparsity analyzer still reserves the same pattern after the `Set` and `LoopEqn` fixes below.

These gains are specific to power flow. The row matching pays off in proportion to how empty the structural diagonal of the Jacobian is, and a `LoopEqn` power-flow Jacobian, whose rows follow the PV and PQ sets while its columns follow the buses, is the extreme case.

The Cookbook IES, a differential-algebraic model of coupled electricity, heat and gas networks integrated by `Rodas`, is the one other model measured on both releases. With the scalar `Eqn` formulation, `loopeqn=False`, the Jacobian evaluation takes 2969 µs on 0.11.0 against 3569 µs on 0.10.2, 17 percent less, which is the gather of [#160](https://github.com/smallbunnies/Solverz/issues/160), while the residual evaluation is unchanged at 813 µs against 808 µs. Both figures come from the two generated modules loaded in one process and timed in ten alternating rounds, and each module timed alone in its own environment gives the same residual, 806 µs against 805 µs. The first Numba compilation of the module took 329 s twice on 0.11.0 and 342.5 s on 0.10.2, and two runs of one build differed by as much as 50 s. On this model the release is therefore neutral in the residual and faster in the Jacobian. Figures posted earlier to [#134](https://github.com/smallbunnies/Solverz/issues/134) and [#170](https://github.com/smallbunnies/Solverz/issues/170) showed the residual 18 percent slower on 0.11.0; they came from one process per environment and do not reproduce.

### New

- **KLU factorizes with a maximum-product row matching in front of its ordering, and its block triangular form is off.** KLU keeps the structural diagonal as its pivot sequence and used to obtain that diagonal from the maximum transversal of the block triangular form, which is blind to the magnitudes. For a Jacobian whose structural diagonal is empty, as every `LoopEqn` model produces because rows follow the equation blocks and columns follow the variables, the AMD ordering of that arbitrary matching carried up to several times the fill of a magnitude-aware one. On the SolUtil power flow of MATPOWER case_ACTIVSg70k the factorization took 1.3 s against 0.19 s for SuperLU; with the rows permuted so that every column has its largest entry, or one close to it, on the diagonal, it takes 52 ms, and every other matrix tried (transient-stability and gas-network iteration matrices, power-flow Jacobians from 9 to 82 000 buses) factorizes as fast or faster than before. The permutation is the maximum-product matching of the MC64 criterion, computed by the new `Solverz.solvers.matching` module with successive shortest augmenting paths in Numba under a work budget, stored with the cached `KLUSymbolic`, and applied to the right-hand side inside `klu_decomposition.solve`, so `nr_method`, `Rodas` and every other caller keep their interface. SciPy's `min_weight_full_bipartite_matching` was not used because its Hopcroft-Karp feasibility pass does not terminate in reasonable time on some chain-like patterns such as the cookbook IES iteration matrix. Systems below 1 000 unknowns keep the previous path unchanged, because the fixed cost of the matching call exceeds their whole factorization; `set_klu_matching(True, min_n=...)` or `SOLVERZ_KLU_MATCHING_MIN_N` moves that threshold, and `set_klu_matching(False)` or `SOLVERZ_KLU_MATCHING=0` restores the previous behaviour everywhere. Measured with the SolUtil LoopEqn power flow on one machine, a Newton solve from the case-file start with KLU takes 8.5 s before and 1.1 s after the change on case_ACTIVSg70k, 9.9 s and 1.2 s on case_SyntheticUSA (82 000 buses), and 476 ms and 241 ms on case_ACTIVSg25k, twice as fast as SuperLU in the same run, while cases between 2 000 and 14 000 buses are within run-to-run noise of the previous times.
- **`PE` and `AdamsBDF` are built-in DAE solvers.** `PE` provides fixed-step partitioned-explicit integration with Euler and modified-Euler differential updates followed by an algebraic Newton solve. `AdamsBDF` provides a variable-step mixed Adams-BDF method that applies Adams treatment to differential variables and BDF treatment to algebraic variables. Both solvers are exported from the public `Solverz` namespace and are documented in the DAE solver API reference.

### Fixed

- **The `tests_in_museum` and `tests_in_cookbook` gates could report on a Solverz that is not the one being released.** Both jobs installed the commit under test by git URL first and other packages after it. Any of those may declare a Solverz version range, and when the range excludes the commit, `uv` silently uninstalls it and puts the newest admitted PyPI release in its place, so every test below reports on that release instead. It is silent because the version a git-URL install reports comes from `setuptools_scm`, which derives it from the tags reachable in that clone: six consecutive runs on a fork whose newest tag is 0.8.2 passed this way, because the commit that derives 0.10.3.dev22 in this repository derives 0.8.2.dev72 there, SolMuseum's `Solverz>=0.10.0,<0.11` excluded it, and the job went green on Solverz 0.10.2 while the real code failed. Both jobs now install the commit under test last, and assert from its `direct_url.json` that the installed Solverz is that commit, so a future replacement fails the job instead of hiding in it.
- **A `Sum` in a `LoopEqn` body silently produced a wrong Jacobian when its dummy reached the analyzer as two objects.** Roughly fifty structural tests in `loop_jac` compare indices by label through `_name_of`; seven compared raw SymPy objects instead. The two policies agreed only for as long as `Set.idx` returned a plain `sympy.Idx`, which SymPy interns, so two indices of one label were one object. `SetIdx` carries its set's token as a third argument and therefore compares unequal to a plain `Idx` of the same label and bounds, by design, so the policies could disagree. Where they did, `_canonicalize_sum` failed to match the `KroneckerDelta` against the `Sum` dummy, lifted it out of the `Sum` it belonged to and left it naming a bound variable, and the `subs` that performs the collapse missed the dummy for the same reason. The generated Jacobian was then finite and wrong rather than merely over-reserved, and the Newton solve diverged without a single NaN; on the polar power flow of SolUtil the assembled Jacobian differed from a central-difference Jacobian by the full scale of its own largest entry, 29.92 against 29.93. The seven sites now compare by label like the rest of the module, and the collapse substitutes every occurrence of the dummy whatever its identity. Comparing by label is safe because the module only ever compares indices within one equation body, where two indices of one label are the same index; whether an index belongs to a given `Set` is a different question and is still answered from the token. A new `check_canonical_invariants` reports the two states that always mean a defect and used to pass silently, namely two free symbols sharing a printed name and a `Sum` dummy that also occurs outside every `Sum`. It warns rather than raises, naming the offending objects with their types and arguments. See [#161](https://github.com/smallbunnies/Solverz/issues/161).
- **A `Set`-backed outer index composed with a second index map forced a dense Jacobian block.** `Set.idx('g')` desugars every access to a gather, so a body written as `m.b[m.back[g]]` reaches the Jacobian pipeline as `b[back[S[g]]]` — two levels of indexing. Both the Phase J1/J2 classifier and the structural sparsity analyzer required the inner index of a map to be a bare `Idx`, so neither recognised the composition and the block fell back to the full `n_outer x n_diff` reservation. The values were always correct, since an over-reservation is a superset of the true pattern, but the cost was paid on every factorization: on a model with 342 coupled rows and 4342 unknowns the assembled Jacobian carried 121 306 nonzeros where 4 684 is exact. A new `resolve_outer_index_values` helper materialises an index expression of any nesting depth into its concrete column vector at build time, so `map[outer]`, `back[S[outer]]`, and `outer + c` all resolve through one rule. The pattern is now the exact permutation, and the same expression also classifies to a constant `_LoopJacSelectMat` instead of falling to the per-entry kernel. Time-varying, fractional, and out-of-range maps are refused rather than guessed, so those cases keep the conservative dense path.
- **A `Set` whose values form the leading range `0..k-1` of a larger index space lost its gather on sparse 2-D Params, and `module_printer(...).render()` failed with `IndexError` inside `FormJac`.** Such a set is flagged `is_identity`, and the body rewriter decides whether an identity-valued set still needs a gather from the length of the array it indexes. For a sparse 2-D Param that length was read through `np.asarray(csc_array).shape[0]`, which raises and was swallowed, so `Gbus[i_q, j]` was walked with the bare position `i_q` while `Vm[PQ[i_q]]` in the same body was gathered. The values were still right, since an identity-valued set maps `i` to `i`, but the structural sparsity analyzer then took every stored row of the Param for a `k`-row block and the generated kernel indexed the set out of bounds. MATPOWER `case39`, whose PQ buses are exactly buses 0 to 28 of 39, is the smallest standard case that triggers it. `_target_first_axis_len` now reads `.shape[0]` from any object that carries a shape, so a leading-range set is gathered like every other subset, and the direct-outer branch of `compute_loop_jac_sparsity` keeps only the entries with `row < n_outer` and `col < n_diff`, which also covers a plain `Idx('i', k)` outer range over a Param with more rows. See [#151](https://github.com/smallbunnies/Solverz/issues/151).
- **A `LoopEqn` whose outer range was shorter than the `Var` it indexes failed to render.** The identity block of `LoopEqn('c', outer_index=Idx('i', 15), body=m.a[i] - 1.0)` over a 16-entry `a` is `15x16`, but `_LoopJacEye` printed a square `np.eye(n_outer)` unconditionally, so `JacBlock` raised `ValueError: Incompatible matrix derivative size (15, 15) and vector variable size (16,)`. `_LoopJacEye` now takes both sizes and prints `np.eye(n_outer, n_diff)` when they differ.
- **`nr_method` recomputed the KLU symbolic analysis and the row matching on every call.** The Newton solver created a fresh `KLUCache` per call, so the ordering lived only for the Newton loop of that one call and every repeated solve of the same `nAE`, namely a power flow restarted from a new operating point or with new injections, a continuation, and every time step of `implicit_trapezoid`, `backward_euler` and `fdae_solver`, which build a throwaway `nAE` per step, paid the analysis again although the Jacobian pattern is fixed. On the SolUtil power flow of MATPOWER case9241pegase a repeated solve took 17.7 ms, of which 12.9 ms were the analysis and the matching; on a synthetic tree-like network of 140 000 unknowns 65 ms against 28 ms. The holder now lives on the `nAE` through `laesolver.model_cache`, as it already did for `Radau` and `sicnm`, and the three per-step solvers hand their throwaway `nAE` the holder of the model they integrate, so one ordering serves the whole integration. A pattern change between calls is still detected by `KLUSymbolic.matches` and re-analysed. See [#156](https://github.com/smallbunnies/Solverz/issues/156).
- **`nr_method` ignored `Opt(linsolver=...)`.** The Newton solver called `solve` without a backend argument, so the option documented for every solver had no effect there and the backend could only be chosen with `set_linsolver` or the `linsolver` context manager, while `Rodas` honoured it. `nr_method` now passes `opt.linsolver` to `solve`; `None` follows the global selection as before. See [#158](https://github.com/smallbunnies/Solverz/issues/158).
- **`sp_decomposition` copied the SuperLU factors at every factorization.** The wrapper that `lu_decomposition` returns for the SuperLU backend read `L`, `U` and `nnz` of the SuperLU object in its constructor; each of `L` and `U` builds a scipy sparse matrix from the factor, a copy of the factor's size, and `Rodas` and `Radau` paid both copies at every step under `linsolver='superlu'` although nothing read them. On the case9241pegase power-flow Jacobian the factorization took 27.2 ms against 25.1 ms for `splu` alone, and on the cookbook IES iteration matrix 6.10 ms against 4.84 ms. The three attributes are now read on first access and kept, so `sicnm`, which reads the factors, pays once and every other caller pays nothing. See [#159](https://github.com/smallbunnies/Solverz/issues/159).
- **A plain `Idx` with the name and size of a set index was treated as set-backed.** `Set.idx(name)` returned a plain `sympy.Idx` and recorded the set in a process-wide registry keyed by the index name and bounds, so a later `Idx(name, size)` of the same name and size, in the same model or in any other model of the process, was gathered through that set's Param and received a parameter its model did not have; `create_instance` then failed with `Parameter S uninitialized`, or, had the model owned a Param of that name, the residual would have been computed at the wrong positions. `Set.idx` now returns a `SetIdx`, a `sympy.Idx` that carries the token of its set as a third argument, so the link survives SymPy rebuilds, two set indices of the same name and size from different sets compare unequal, and a plain `Idx` is never mistaken for a set index. See [#161](https://github.com/smallbunnies/Solverz/issues/161).
- **The generated `J_` rebuilt the CSC structure from COO on every call.** The module printer ended `J_` with `sps.coo_array((data, (row, col)), shape).tocsc()`, whose sort and duplicate merge repeat at every call although the pattern of a generated Jacobian is fixed and only the values change. The pattern is now analysed once when the module is imported, by a `SolCF.CooToCsc` object named `_sz_coo2csc` in the generated `num_func.py`, and `J_` gathers the fresh values into the CSC arrays, summing duplicate entries as scipy does. On the SolUtil power-flow Jacobian of MATPOWER case9241pegase, 138 608 entries, the assembly drops from 517 µs to 65 µs per call, about 5 percent of the Jacobian evaluation. Modules rendered before this change keep their own `J_`, and the inline `made_numerical` path is unchanged. See [#160](https://github.com/smallbunnies/Solverz/issues/160).
- **Numba could not cache the LoopEqn kernels of large models.** The CSR arrays of a sparse walker, and the row and column arrays of a loop Jacobian kernel, were read by the compiled functions as module-level globals. Numba freezes a global array into the compiled code only up to 1 MB; above that it embeds the array's address, treats the function as using dynamic globals and refuses to cache it, so every process that imported such a module recompiled `inner_F`, the walker kernels and `inner_J`. The Python-level `F_` and `J_` wrappers now hand the arrays down as arguments, through `inner_F` and `inner_J` to the kernels and the point-lookup helpers, so no compiled function reads a global array. On a LoopEqn over 20 000 rows with a 170 000-entry walker, a fresh process spent 0.90 s compiling at import before and 0.09 s after, and Numba no longer reports `Cannot cache compiled function`. Modules rendered before this change are unaffected. See [#162](https://github.com/smallbunnies/Solverz/issues/162).

## 0.10.2

### Changed

- **`Rodas` decides the row-equilibration reduction once per integration.** The row-scale step needs the row maxima of the iteration matrix, which change every step, but whether that reduction yields a sparse column to densify or an already-dense array is fixed for the whole solve. `Rodas` now judges this once on the first reduction instead of type-checking each step, so the dense path never calls `.toarray()` and the sparse path skips the repeated `hasattr` probe.

## 0.10.1

### Fixed

- **`Rodas` crashed on a dense iteration matrix.** The 0.10.0 Rodas row-equilibration step computed `1 / max|row|` and called `.toarray()` on the row reduction, which only exists for a sparse iteration matrix. A model compiled with `made_numerical(..., sparse=False)` produces a dense `ndarray`, so `Rodas` raised `AttributeError: 'numpy.ndarray' object has no attribute 'toarray'`. The row reduction is now coerced to a flat array for sparse, dense-`ndarray`, and `np.matrix` iteration matrices alike, so the dense `Rodas` path runs and matches the sparse result.
- **Generated modules with provenance docstrings failed to import on Windows.** The 0.10.0 provenance docstrings used a Unicode em-dash as the source separator, and the module printer wrote the generated `.py` files with `open(..., "w")`, which uses the OS locale encoding. On Windows (cp1252) the em-dash was written as byte `0x97`, so importing the generated module raised `SyntaxError: 'utf-8' codec can't decode byte 0x97`. This only affected stamped models, so it surfaced once SolMuseum / SolPSDyn began calling `stamp_source`. The printer now writes all generated `.py` files as UTF-8 explicitly, and `format_source` uses a plain ASCII separator, so generated modules import on every platform.

## 0.10.0

Self-describing generated modules: every `module_printer(...).render()` artifact now carries provenance docstrings, and a new `stamp_source` helper lets reusable components tag the equations they contribute.

### New

- **Generated modules are now self-describing through provenance docstrings.** The module-level docstring in the generated `__init__.py` records the generating Solverz version, a UTC generation timestamp, a `Provenance:` block grouping each contributing package and version with its components, and an `Equations:` table mapping every `inner_F` function to its equation name, declaration-order row range, and source. Each F equation function and each J derivative kernel in `num_func.py` is annotated with a one-line docstring naming its equation, rendered as the equation name for F residuals and as `d(eqn)/d(var)` for J kernels, including the loop-jacobian kernels of `LoopEqn`. When an equation was stamped, the docstring also carries its source as `package version / component`.

  ```python
  from Solverz import stamp_source

  stamp_source(model, component='gt', package='SolMuseum', version='0.2.0')
  ```

  Reusable building blocks call `stamp_source(model, component=..., package=..., version=...)` once per fragment to tag every equation they contribute. The default `overwrite=False` leaves equations that already carry a more specific source untouched, so a composite block does not clobber the stamps of the sub-fragments it aggregates. Solverz core never imports the downstream packages; the version travels as data on the equation objects.

  Plain models that are never stamped degrade gracefully: F and J docstrings carry the equation name only, and the module docstring groups their equations under `(user-defined)`. `num_func.py` carries no timestamp and is byte-stable across renders of the same model, so only the `__init__.py` header timestamp varies between two renders.

### Fixed

- **`Rodas` produced `NaN` on tightly-coupled, ill-conditioned DAEs.** The Rosenbrock iteration matrix `M - dt * gamma * J` was factored without scaling. On systems whose blocks span very different magnitudes, such as an IEGS model coupling a gas subsystem in Pa with an EPS subsystem in per-unit through an algebraic link, the matrix was ill-conditioned enough that the LU factorization returned values around `1e100` and the integration filled with `NaN`. The iteration matrix is now row-equilibrated by `1 / max|row|` before the LU, matching the BDF `perf_lu` path, and each stage right-hand side is scaled by the same factor so every stage solution is unchanged. This restores a well-conditioned solve on the coupled DAEs while leaving well-scaled problems numerically identical to within row-scaling round-off.
- **Single-element `LoopEqn` over a length-1 `Var` raised `TypeError` during Jacobian generation.** `module_printer(...).render()` aborted in `FormJac` with `TypeError: Matrix derivative of scalar variables not supported!` whenever a `LoopEqn` covered exactly one row whose loop variable was length-1, even though `create_instance()` reported a well-formed square system. The body's identity Jacobian is the `1x1` `_LoopJacEye(1)`, a matrix derivative, taken with respect to a Var that the Jacobian builder classified as scalar purely from its length-1 size. `JacBlock` rejected that scalar/matrix combination, while the same `LoopEqn` over two or more rows rendered without error. A matrix-valued derivative now reclassifies a length-1 Var as a 1-element vector, so the block flows through the same vector/matrix path as the `n >= 2` case. This unblocks any vectorized device that emits its wiring or balance equations through `LoopEqn` and is instantiated with a single member. The fix is general: a non-`LoopEqn` `1x1` matrix derivative of a length-1 Var, such as `Mat_Mul(A, x)` with a `(1, 1)` `A`, now renders as well.

## 0.9.0

Built-in Radau IIA(5) DAE solver, SuiteSparse KLU as the default sparse
linear-solver backend, and a silent-correctness fix for LoopEqn
Jacobians that multiply a Var by a `TimeSeriesParam`.

### New

- **`Radau` — 3-stage 5th-order Radau IIA fully-implicit Runge-Kutta
  DAE integrator.** Ported from SciML's `RadauIIA5`
  (`OrdinaryDiffEqFIRK`) and adapted to Solverz's `nDAE` API.
  Stiffly accurate (`y_{n+1} = y_n + Z_3`), with simplified Newton
  iteration in the W-coordinate system that decouples each step into
  one real and one complex sparse LU factorisation. Adaptive step
  size uses the Hairer / SciML predictive (Gustafsson) controller.
  Exposed as `Solverz.Radau`.

  ```python
  from Solverz import Radau, Opt
  sol = Radau(ndae, [0.0, 20.0], y0, Opt(rtol=1e-6, atol=1e-9))
  ```

  Suitable for stiff or oscillatory DAEs that need higher order than
  `Rodas`. Dense output and terminal/non-terminal event detection
  both run on the Radau IIA(5) collocation polynomial, so event
  times are resolved to machine precision regardless of how large
  the controller lets `h` grow on smooth trajectories.

### Changed

- **SuiteSparse KLU is now the default sparse-LU backend.** Newton /
  Rosenbrock / BDF solves factor sparse Jacobians through a pure-Python
  ctypes binding to SuiteSparse KLU, with scipy SuperLU as the automatic
  fallback when `libklu` is missing or the matrix is complex or dense.
  KLU is roughly a factor of two faster than SuperLU on power-system /
  IEGS network Jacobians and needs no build step. Override per solve
  with `Opt(linsolver='superlu')`, for a script with
  `set_linsolver('superlu')`, for a block with `with linsolver('superlu'):`,
  or globally with `SOLVERZ_LINSOLVER=superlu`. KLU requires SuiteSparse
  installed (`brew install suite-sparse` or
  `conda install -c conda-forge suitesparse`). To reproduce older
  SuperLU-only results exactly, force SuperLU, since KLU can change
  adaptive step sequences at the tolerance level.

### Fixed

- **LoopEqn Jacobian silently baked `TimeSeriesParam` contributions.**
  The matrix-derivative constant classifier in the LoopEqn Jacobian
  pipeline (`is_constant_matrix_deri`) did not inspect whether a `Para`
  free symbol was a `TimeSeriesParam`. Any LoopEqn body multiplying a
  `Var` by a `TimeSeriesParam` factor (e.g. a short-circuit
  `G_shunt[i] * ux[i]` term) produced a Jacobian whose `F_` correctly
  evaluated `get_v_t(t)` but whose `J_` froze at the build-time
  `TimeSeriesParam` value and never updated, corrupting every
  short-circuit / leak / valve-trip / step-change simulation built that
  way. The classifier is now `TimeSeriesParam`-aware, and
  `loop_jac_to_solverz_expr` falls through to a per-cell dynamic kernel
  that re-evaluates the `TimeSeriesParam` every `J` call.
- **`Saturation` / `In` now preserve input shape under `@njit`.** The
  Numba-compiled forms could return a scalar for a length-1 vector
  input; they now return an array matching the input shape.

## 0.8.6

Mat_Mul fallback diagnostic warnings (#125). When a `Mat_Mul`
placeholder or a mutable Jacobian block is forced onto the slower
scipy.sparse fallback path, the module printer now emits a
`UserWarning` that tells the user *what* expression broke the fast
path, *why*, and *how to rewrite* it.

### New

- **Layer 1 `Mat_Mul` placeholder fallback warnings.** The module
  printer's `print_F` now emits one `UserWarning` per fallback
  placeholder, with specialised messages for negation, scalar
  multiplication, sum-of-matrices, multi-argument `Mat_Mul` folding,
  sparse `dim=2` `Param` references in the operand, and demotion
  cascades (root-cause traceback).

- **Layer 2 mutable Jacobian block fallback warnings.** When a
  Jacobian block term doesn't match the `Diag` / row-scale /
  col-scale / biscale fast-path shapes, the module printer emits a
  `UserWarning` naming the equation, variable, and offending term.

- **`matrix_calculus.md` diagnostic warnings section** listing every
  warning category, what triggers it, and the canonical fix.

### Changed

- Warnings only fire in module mode (not inline), because inline
  mode has no fast/fallback split. The existing
  `_warn_dense_matmul_params` warning (from `FormJac` time) still
  fires in both modes.

### Fixed

- **`Mat_Mul` printer associativity.** When the matrix argument was
  a non-atomic expression such as `A + B` or `2 * A`, `Mat_Mul`'s
  numpy / sympystr / octave printers emitted `A + B@x` instead of
  `(A + B)@x`, so `Mat_Mul(A+B, x)` lambdified to a 2-D ndarray and
  failed `Array(..., dim=1)` validation in `create_instance()`.
  Each printer now wraps any non-`Symbol`/`Function` operand in
  parentheses.

- **Layer 2 classifier no longer conflates structurally distinct
  fallback shapes.** The previous classifier short-circuited on any
  `Diag`-bearing term and unconditionally emitted `"multiple Diag
  nodes"`, misdiagnosing single-`Diag` and element-wise `Mul` cases.
  The new classifier dispatches on the post-sign-extraction core and
  produces distinct messages for biscale, single-`Diag`, no-`Diag`
  `Mat_Mul`, element-wise `Mul`, bare `Para`, and other shapes.

- **Layer 1 `Add`+non-`Para` suggestion** previously contained literal
  Python source (`' + '.join(f'Mat_Mul({arg}, <operand>)' for arg in
  ...)`) instead of a usable distributed-form rewrite.

### How to silence

```python
import warnings
warnings.simplefilter('ignore', UserWarning)
```

## 0.8.5

LoopEqn prototype close-out: new `Set` primitive for subset
iteration (#129), Phase J2 translator Patterns 1 / 2 / 4 landings
(#133), safety guards (#130, #132), and a full LoopEqn
documentation rewrite.

### New

- **Pyomo-style `Set` primitive** (#129). `Solverz.Set`
  (internal class `IndexSet`) replaces the three-object
  subset-iteration pattern — auxiliary `Param(..., dtype=int)` +
  bounded `Idx` + every `m.Var[m.subset_param[i]]` indirection —
  with a one-line declaration:

  ```python
  m.PVPQ = Set('PVPQ', pv_pq_arr)
  m.Bus  = Set('Bus', nb)
  i_p = m.PVPQ.idx('i_p')
  j   = m.Bus.idx('j')
  body_P = m.Vm[i_p] * Sum(m.Vm[j] * m.Gbus[i_p, j] * ..., j) + ...
  m.P_eqn = LoopEqn('P_eqn', outer_index=i_p, body=body_P, model=m)
  ```

  The body rewriter inserts the indirect gather
  (`m.Vm[PVPQ[i_p]]`) only when the target array is strictly
  larger than the set — subset-aligned storage whose length equals
  the set is indexed bare. Same indirect-outer path the sparsity
  analyzer already handles, no engine changes required. Partial-
  unknown Vars (Pyomo's `Var(m.PQ)` pattern) are tracked
  separately in #136.

- **`Model.add` name-collision detection** (#130). Raises
  `ValueError` when merging a sub-model whose attribute would
  overwrite an existing `Param` / `Var` / `Eqn` with a non-equal
  value. Previously the clash was silent — a real IES integration
  bug (gas- vs heat-network `pipe_from_node` parameters silently
  overwriting each other, producing `|F| ≈ 1.5e6` residuals)
  motivated the guard. Identical values (shared object or
  value-equal Params like a common `Cp`) still merge without
  error.

- **LoopEqn + inline / partial-Jacobian path guard** (#132, C2
  guard from #128). `made_numerical` and
  `Equations.FormPartialJac` now raise `NotImplementedError` when
  the equation system contains any `LoopEqn` instance. LoopEqn
  kernels emit Python `for`-loops that are 3–7× slower than the
  lambdify-vectorised legacy `Eqn` path without Numba JIT; the
  guard redirects users to `Solverz.module_printer(..., jit=True)`,
  which is LoopEqn's design target.

### Performance

- **Phase J2 translator Patterns 1 / 2 / 4** (#133 completion).
  LoopEqn Jacobian blocks of the following shapes now land on the
  constant-matrix fast path with zero per-Newton-iteration J cost:
  * Pattern 1 DiagSelectTerm — `Diag(u) @ SelectMat @ Para`.
  * Pattern 2 bilinear mixing — `Diag(u) @ Para @ Diag(v)`
    (new **biscale** term, flat / nested / `-1`-wrapped forms).
  * Pattern 4 Sum-KD — identity and non-identity column maps,
    bare and indirect outer. All three cases keep
    `is_constant_matrix_deri = True` so Value0 is baked into
    `_data_` at module-build time.

- **Pattern 3 identity-map indirect KD → DiagTerm** (#133). A
  top-level `KD(diff, map[outer]) * Sum(...)` with identity
  `map` (and `n_outer == n_diff`) folds into the existing
  direct-KD `Diag(Mat_Mul(Para, iVar))` branch.

- **Mutable-matrix analyzer extensions** — `_LoopJacSelectMat`
  accepted as matrix operand alongside `Para`, including
  `Mat_Mul(SelectMat, Para)` composites; variadic `Mat_Mul`
  chains accepted in `_classify_matmul`;
  `_extract_sign_and_core` strips any numeric scalar coefficient.

- **Measured impact (Big IES, 8 996-DOF, Rodas3 Mode II)** —
  wall 13.76 s vs 52.25 s for legacy (**3.80× faster**), F-eval
  709.8 µs vs 4 348.5 µs, J-eval 4 634 µs vs 15 832 µs. Step
  counts identical (757 accepted) confirming trajectory match.

### Documentation

- `docs/src/loopeqn.md` — full rewrite as a beginner walkthrough.
  10 sections from motivation to troubleshooting; two Cookbook
  worked examples (polar power-flow, integrated energy system)
  instead of synthetic snippets; `Set`-based subset iteration; the
  `module_printer(jit=True)` requirement; `LoopOde`; performance
  numbers.
- `docs/src/loopeqn_translator.md` (new) — developer-facing
  appendix that defines every translator jargon term (KD, Mat_Mul,
  Phase J1/J2/J3, CSR walker, SelectMat, biscale) at first use
  and houses the Phase J2 coverage table, the supported-body-shape
  table, and the not-supported list.
- `docs/src/conf.py` — enable
  `myst_enable_extensions = ['dollarmath', 'amsmath', 'colon_fence']`
  so `$$...$$` and `:::{math}` blocks render typeset in `.md`
  pages.
- API reference gains `IndexSet` autoclass entry.

### Internal

- `SymbolExtractor` leaf-symbol fix — `Para` / `iVar` /
  `iAliasVar` used as indices now register in `SymInIndex` by
  name, so lambdified bodies that gather via a Param (e.g.
  `iVar('Ts')[Para('fht_nl')]`) no longer raise `NameError` at
  runtime.
- Inline printer no longer registers `LoopEqn.NUM_EQN` or
  `LoopEqnDiff._kernel_func_name` callables — that code path is
  unreachable behind the guard and has been pruned.
- Value0 shape is verified against `(n_outer, n_diff)` before
  Pattern 4 emits its translated Param expression, preventing a
  latent `IndexError` when a sparse-pattern row falls outside the
  LoopEqn's equation-block range.

## 0.8.4

**Never released.** Content merged into 0.8.5.

## 0.8.3

**Tooling-only release.** No Python source-code, test, or runtime
behaviour changes. The PyPI wheel for 0.8.3 is bit-identical to 0.8.2
modulo the new in-tree skill files (which are not bundled into the
wheel).

### Tooling

- **New [Claude Code](https://claude.com/claude-code) skill** at
  `.claude/skills/solverz-modeling/` that teaches Claude how to use
  Solverz for symbolic modeling and numerical simulation. Bundles:

  - `SKILL.md` — 4-step workflow (equation type → build → compile →
    solve), `Var` / `Param` / `Eqn` / `Ode` idioms, inline vs
    `module_printer` decision, every built-in solver with when-to-use
    notes, `Mat_Mul` fast vs fallback path with the full rewrite
    table, common pitfalls table, and a quick-reference card.
  - `references/ecosystem.md` — chapter map of the Solverz Cookbook,
    every reusable block in [SolMuseum](https://github.com/rzyu45/SolMuseum)
    (`gt`, `pv`, `st`, `eb`, `eps_network`, `heat_network`,
    `gas_network`, `pde.heat`, `pde.gas`), and every helper in
    [SolUtil](https://github.com/rzyu45/SolUtil) (`PowerFlow`,
    `DhsFlow`, `GasFlow`, `DhsFaultFlow`).
  - `references/examples/` — six canonical end-to-end runnable
    examples covering AE / DAE / FDAE / mutable Jacobian / events /
    `AliasVar` / `Mat_Mul` / `model.add()` composition:

    - `bouncing-ball.md` — minimal DAE with event handling
    - `power-flow.md` — canonical AE with `Mat_Mul` (case30,
      rectangular coordinates)
    - `heat-flow.md` — AE with mutable-matrix Jacobian
    - `m3b9-dynamics.md` — DAE with `TimeSeriesParam` fault scenario
    - `gas-characteristics.md` — FDAE with `AliasVar` (method of
      characteristics)
    - `integrated-energy-system.md` — multi-domain DAE composition
      using all three SolUtil flow solvers + the major SolMuseum
      DAE / AE blocks via `model.add(...)`

  - `README.md` — install command (symlink) and sync rule for
    contributors.

  Install on a contributor's machine:

  ```sh
  ln -sfn "$(pwd)/.claude/skills/solverz-modeling" \
          ~/.claude/skills/solverz-modeling
  ```

  After the symlink is in place, `git pull` on a Solverz checkout
  updates the skill content automatically — no re-install step.
  Inside a Solverz checkout itself the skill auto-loads even without
  the global symlink, because Claude Code reads
  `<cwd>/.claude/skills/`.

The skill is **not** bundled into the PyPI wheel — it lives only in
the source tree. Pip-installed users who want the skill should clone
the repo separately.

The contribution guide in `.claude/skills/solverz-modeling/README.md`
documents the sync rule for Solverz contributors: when a PR changes
Solverz's public API, the same PR should update the relevant skill
files. Reviewers will check both.

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
