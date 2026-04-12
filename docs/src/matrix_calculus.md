(matrix_calculus)=

# Matrix-Vector Calculus

Solverz provides automatic symbolic differentiation for mixed matrix-vector expressions.
When an equation contains matrix-vector products (`Mat_Mul`), Solverz automatically computes
analytical Jacobian blocks using tensor calculus with Einstein index notation.

This module is based on the approach described in:

> Laue, Mitterreiter, Giesen. *A Simple and Efficient Tensor Calculus.*
> Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

See also: [matrixcalculus.org](https://www.matrixcalculus.org/)

## Supported Operations

| Operation | Symbolic Form | Example | Derivative |
|-----------|---------------|---------|------------|
| Matrix-vector multiply | `Mat_Mul(A, x)` | $Ax$ | $A$ |
| Element-wise multiply | `x * y` | $x \odot y$ | $\operatorname{diag}(y)$ |
| Addition | `x + y` | $x + y$ | $I$ |
| Absolute value | `Abs(x)` | $\|x\|$ | $\operatorname{diag}(\operatorname{sign}(x))$ |
| Exponential | `exp(x)` | $e^x$ | $\operatorname{diag}(e^x)$ |
| Sine / Cosine | `sin(x)` / `cos(x)` | $\sin(x)$ | $\operatorname{diag}(\cos(x))$ |
| Logarithm | `ln(x)` | $\ln(x)$ | $\operatorname{diag}(x^{-1})$ |
| Power | `x**n` | $x^n$ | $\operatorname{diag}(n x^{n-1})$ |
| Transpose | `transpose(A)` | $A^T$ | $A^T$ |
| Diagonal | `Diag(x)` | $\operatorname{diag}(x)$ | $\operatorname{diag}(\cdot)$ |

```{note}
`Mat_Mul` is the recommended interface for matrix-vector products. The legacy `MatVecMul` function
(which decomposes sparse matrices into CSC components for Numba) is deprecated. `Mat_Mul` uses
`scipy.sparse` directly, which is both faster and supports full matrix calculus.
```

```{warning} **Sparse matrices MUST be immutable after modelling**

All sparse matrix parameters declared with `Param(..., dim=2, sparse=True)` are
assumed to be **immutable** (both sparsity pattern *and* numerical values) once
modelling is complete and `model.create_instance()` has been called.

Solverz relies on this assumption for maximum runtime efficiency:

- At code-generation time, the sparsity pattern of every mutable-matrix Jacobian
  block is analysed *once*, the output data-array layout is fixed, and a pre-
  computed index mapping is baked into a Numba-compiled scatter-add loop.
- At runtime, each Newton step assembles Jacobian block data by directly
  indexing into `Matrix.data` using the pre-computed row/column mappings —
  **no scipy.sparse matrix is constructed per iteration**.
- For sparse-matrix parameters used in `Mat_Mul(A, x)`, the matrix-vector
  product `A @ x` is precomputed in the `F_` wrapper and passed as a dense
  vector to the `@njit`-compiled equation functions.

If your application truly needs to update sparse-matrix values between solves
(e.g., sweeping a parameter), rebuild the model with `model.create_instance()`
each time. Mutating `p_["A"].data` after creation will silently produce wrong
Jacobians because the stored index mappings and precomputed factors no longer
match.

Dense parameters (`Param(..., dim=2, sparse=False)`) are fine to update via
`mdl.p["name"] = new_value`; the restriction only applies to **sparse**
dim=2 parameters.
```

All element-wise functions (exp, sin, cos, ln, Abs) can be composed with matrix operations:

```python
from Solverz import Var, Param, Eqn, Model
from Solverz.sym_algebra.functions import Mat_Mul, exp, sin

m = Model()
m.x = Var('x', [1, 2, 3])
m.A = Param('A', [[1, 0, 0], [0, 2, 0], [0, 0, 3]], dim=2)
m.f = Eqn('f', exp(Mat_Mul(m.A, m.x)))  # exp(A@x)
```

Solverz will automatically compute $\frac{\partial}{\partial x} e^{Ax} = \operatorname{diag}(e^{Ax}) \cdot A$.

## How It Works

The matrix calculus engine is a four-stage pipeline: each expression is
lifted into Einstein index notation, rebuilt as a computation DAG,
differentiated forward through the DAG, and finally collapsed back to
ordinary matrix/vector operations. We walk through each stage with a
running example from power flow.

### 1. Einstein Index Notation

Every tensor is labelled with one index per axis:

- **Scalar**: no index (e.g., $\alpha$). Zeroth-order tensor, no axes.
- **Vector** $x \in \mathbb{R}^n$: a single index $x_i$, where $i$ ranges
  over $\{1, \dots, n\}$. The index *names* the axis; the letter used is
  arbitrary, only its pattern of repetition matters.
- **Matrix** $A \in \mathbb{R}^{m \times n}$: two indices $A_{ij}$. The
  first index $i$ labels rows, the second $j$ labels columns.

A **repeated index inside a product** implies *summation*
(Einstein summation convention, or *contraction*):

```{math}
A_{ij}\, x_j \;\equiv\; \sum_{j=1}^{n} A_{ij}\, x_j \;=\; (Ax)_i.
```

Here $j$ is repeated (contracted — it disappears in the result), while
$i$ is a **free index** that labels the resulting vector's axis.
Conventionally we write the non-contracted indices on the left of the
expression, so the indices of the left-hand side $(Ax)_i$ match the
remaining free indices $i$ on the right.

The benefit of this notation is that the *shape* of an expression is
fully encoded in its index signature. Element-wise multiplication uses
repeated indices *without* contraction:

```{math}
(x \odot y)_i \;=\; x_i\, y_i,
```

and the outer product uses two different free indices:

```{math}
(x\, y^\top)_{ij} \;=\; x_i\, y_j.
```

### 2. Computation DAG

Solverz parses each equation's symbolic expression into a **directed
acyclic graph (DAG)**:

- **Leaf nodes** — variables ($x$, $y$) and parameters ($A$, $B$, $G$).
  Each leaf carries its index signature ($x_i$, $A_{ij}$, etc.).
- **Internal nodes** — operations: addition $+$, element-wise product
  $\odot$, matrix product $@$, element-wise unary function $f(\cdot)$.
- **Root node** — the output expression whose Jacobian we want.

For the power flow active-power sub-expression $e \odot (Ge - Bf)$:

```
        *[i]          ← element-wise multiply (free index i)
       /    \
     e[i]   -[i]      ← subtraction (free index i)
           /    \
       @[i]    @[i]   ← matrix-vector products, contraction on j
       / \     / \
     G[ij] e[j] B[ij] f[j]
```

Each node remembers its free index set. The matrix product nodes
`@[i]` each *contract* index $j$ between $G_{ij}$ and $e_j$, so the
result has free index $i$. The subtraction and element-wise product
keep $i$ as their only free index, so the whole expression has index
signature $[i]$ — a vector.

### 3. Forward-Mode Automatic Differentiation

To obtain $\partial f / \partial x$ where $f$ is the root and $x$ is
some leaf variable, derivatives propagate from **leaves upward to the
root** (pushforward, a.k.a. forward-mode AD). Every internal node
computes its own Jacobian contribution and combines it with the
Jacobians already accumulated at its children.

Throughout this section, $A_{s_1}$ and $B_{s_2}$ denote the free
indices of the two operands (before differentiation), $s_3$ denotes the
free index of the result, and $s_4$ is the **derivative-direction
index** — the axis along which we differentiate, equal to the free
index of the variable $x$ we differentiate with respect to. Tilded
tensors $\tilde A, \tilde B$ carry one *extra* axis
(labelled $s_4$) compared to $A, B$: the derivative of a rank-$k$
tensor w.r.t. a rank-1 variable is a rank-$(k{+}1)$ tensor.

1. **Leaf variables.** Differentiating $x_j$ w.r.t. $x_{s_4}$ gives the
   Kronecker delta:

   ```{math}
   \tilde{x}_{j\,s_4} \;=\; \frac{\partial x_j}{\partial x_{s_4}} \;=\; \delta_{j s_4}.
   ```

   All other variables (and all parameters) have derivative $0$ w.r.t.
   $x$.

2. **Multiplication** (product rule in index notation):

```{math}
\tilde{C}_{s_3\,s_4} \;=\; B_{s_2}\cdot\tilde{A}_{s_1\,s_4}
                      \;+\; A_{s_1}\cdot\tilde{B}_{s_2\,s_4}
```

   where $A_{s_1} \cdot B_{s_2} = C_{s_3}$ is the original product (the
   specific index pattern $s_1, s_2 \to s_3$ depends on the product
   type — element-wise, matrix-vector, or matrix-matrix). The two
   summands encode "differentiate the first factor, keep the second"
   plus "keep the first factor, differentiate the second".

3. **Element-wise unary function** $f$ (e.g. $\exp$, $\sin$, $\ln$):
   since $f$ acts point-wise, the chain rule gives

```{math}
\frac{\partial f(A_{s_1})}{\partial x_{s_4}} \;=\; f'(A_{s_1})\odot\tilde{A}_{s_1\,s_4}.
```

   The derivative $f'$ is also computed element-wise and multiplied
   (Hadamard) with the child's Jacobian.

4. **Addition**. Linearity:

```{math}
\tilde{(A+B)}_{s_3\,s_4} \;=\; \tilde{A}_{s_1\,s_4} + \tilde{B}_{s_2\,s_4}.
```

### 4. TMul2Mul: Index Resolution

After the forward pass, derivatives live as tensor expressions labelled
with free indices. The **TMul2Mul** stage reads the index pattern of
each product and emits the corresponding concrete matrix / vector
operation. Each row of the table has the form $(s_1, s_2, s_3)$: the
free indices of the left operand, right operand, and the result.

| Index pattern $(s_1, s_2, s_3)$ | Matrix operation | Meaning |
|---|---|---|
| $(i,\; i,\; i)$                 | $x \odot y$                            | element-wise multiply — both operands share axis $i$, no contraction |
| $(i,\; j,\; ij)$                | $x\,y^\top$                            | outer product — two disjoint free indices survive into the result |
| $(ij,\; j,\; i)$                | $A\,y$                                 | matrix-vector: index $j$ is shared (contracted), $i$ survives |
| $(ij,\; jk,\; ik)$              | $A\,B$                                 | matrix-matrix: inner $j$ is contracted, $(i,k)$ survive |
| $(i,\; ij,\; ij)$               | $\operatorname{diag}(x)\,A$            | left diagonal scaling — scale each row of $A$ by the matching entry of $x$ |
| $(ij,\; j,\; ij)$               | $A\,\operatorname{diag}(y)$            | right diagonal scaling — scale each column of $A$ by the matching entry of $y$ |
| $(i,\; ii,\; ii)$               | $\operatorname{diag}(x \odot y)$       | diagonal-from-element-wise — used when a product has matching repeated indices $(ii)$ on one operand |

The key insight is that every well-formed index pattern maps to one
concrete sparse-preserving operation. Patterns like $(ij, ij, ij)$ that
don't match the table are either simplified earlier (e.g. $(ij, ij, ij)
\to A \odot B$, element-wise matrix product) or emitted via a fallback
that produces a dense matrix.

## Application Examples

### Example 1: Power Flow Equations

The active power balance equation in power systems:

```{math}
P = e \odot (Ge - Bf) + f \odot (Be + Gf)
```

```python
from Solverz.sym_algebra.symbols import iVar, Para
from Solverz.sym_algebra.functions import Mat_Mul
from Solverz.sym_algebra.matrix_calculus import TensorExpr

e = iVar('e')
f = iVar('f')
G = Para('G', dim=2)
B = Para('B', dim=2)

P = e * (Mat_Mul(G, e) - Mat_Mul(B, f)) + f * (Mat_Mul(B, e) + Mat_Mul(G, f))
TP = TensorExpr(P)

print(TP.diff(e))  # diag(-B@f + G@e) + diag(e)@G + diag(f)@B
print(TP.diff(f))  # diag(B@e + G@f) + diag(e)@(-B) + diag(f)@G
```

### Example 2: Nonlinear Matrix Equations

```python
from Solverz.sym_algebra.functions import exp

A = Para('A', dim=2)
x = iVar('x')

expr = exp(Mat_Mul(A, x))
TE = TensorExpr(expr)
print(TE.diff(x))  # diag(exp(A@x))@A
```

### Example 3: Heat Network Equations

```python
mL = Para('mL', dim=2)
K = Para('K')
m = iVar('m')
from Solverz.sym_algebra.functions import Abs

expr = Mat_Mul(mL, K * Abs(m) * m)
TE = TensorExpr(expr)
print(TE.diff(m))  # mL@(diag(K*Abs(m)) + diag(K*m*Sign(m)))
```

## How numerical code is generated and optimized

When `module_printer` compiles an equation system that contains
`Mat_Mul`, it applies a **three-layer optimisation strategy** to make
the generated `F_`/`J_` functions run as fast as possible. All three
layers operate on the same insight: *at modelling time we already know
what the equations and their Jacobians look like* — so every
computation that does not depend on the current iterate $y$ can be
lifted out of the hot loop and paid for once, at code generation or
module load.

```{warning}
Everything below assumes **sparse matrix parameters are immutable**
after `model.create_instance()`. See the warning earlier in this
document. Every layer described here bakes in the matrix's sparsity
pattern or its `.data` values as a constant; mutating them in-place
will silently corrupt results without raising any error.
```

### Layer 1 — Precompute `Mat_Mul` in the `F_` wrapper

The code printer walks every residual RHS looking for `Mat_Mul(A, v)`
sub-trees, where `A` is a sparse parameter and `v` is any vector-valued
expression (a variable, a slice, or a larger expression). It performs
**three transformations at code gen time**:

1. **Hoist.** Each `Mat_Mul(A, v)` is replaced by a fresh placeholder
   variable `_mmN` (where `N` is a monotonically increasing integer
   unique across the whole system).
2. **Deduplicate.** Two `Mat_Mul` sub-trees that are structurally equal
   (same matrix, same operand expression after hoisting) are collapsed
   into a single placeholder. The power-flow example below computes
   `G_nr @ e` once even though both the P-balance and Q-balance
   residuals reference it.
3. **Emit precompute statements in the wrapper.** Each placeholder gets
   one line of the form `_mmN = A @ operand` inserted at the top of the
   `F_()` function body, *outside* the Numba-compiled inner functions.

The runtime effect: the *only* scipy.sparse matrix-vector product that
happens inside a Newton step is the one at the very top of `F_()`.
Every downstream computation (`inner_F`, `inner_F0`, `inner_F1`, …)
receives `_mmN` as a plain dense numpy array and can run under
`@njit(cache=True)`:

```python
def F_(y_, p_):
    e = y_[0:29]
    f = y_[29:58]
    G_nr = p_["G_nr"]           # csc_array, scipy.sparse
    B_nr = p_["B_nr"]
    _mm0 = (G_nr @ e)           # ← precomputed, one scipy SpMV
    _mm1 = (B_nr @ f)           # ← dedup: B_nr@f appears in 2 equations
    _mm2 = (B_nr @ e)
    _mm3 = (G_nr @ f)
    return inner_F(_F_, e, f, p_ref, q_ref, _mm0, _mm1, _mm2, _mm3, ...)

@njit(cache=True)
def inner_F(_F_, e, f, p_ref, q_ref, _mm0, _mm1, _mm2, _mm3, ...):
    _F_[0:29] = inner_F0(e, f, p_ref, _mm0, _mm1, _mm2, _mm3)
    _F_[29:53] = inner_F1(...)
    return _F_

@njit(cache=True)
def inner_F0(e, f, p_ref, q_ref, _mm0, _mm1, _mm2, _mm3):
    return e * (p_ref - _mm1 + _mm0) + f * (q_ref + _mm2 + _mm3) - Pinj
```

Why this matters: scipy.sparse matrix-vector products are already
implemented in C and fairly fast, but *constructing intermediate sparse
matrices* (e.g. `diag(e) @ G` returning a new csr_matrix) is not — the
construction allocates, copies, and eventually eliminates explicit
zeros. By forcing the sparse work into a handful of clean SpMVs and
leaving the rest to Numba, we get the best of both worlds.

### Layer 2 — Mutable matrix Jacobian blocks

The Jacobian is handled by a **separate, symbolic decomposition**
because its blocks are matrices (not vectors) whose structure depends
on the full expression. Consider the power-flow block

```{math}
\frac{\partial P}{\partial e} \;=\;
  \operatorname{diag}\!\bigl(G\,e - B\,f + p_\text{ref}\bigr)
  \;+\; \operatorname{diag}(e)\,G
  \;+\; \operatorname{diag}(f)\,B.
```

All three terms are *mutable* — they depend on $e$ and $f$ and must be
re-evaluated every Newton step. But their **sparsity pattern is
constant**: the union of the diagonal, the pattern of $G$, and the
pattern of $B$ — all fixed at modelling time.

Solverz exploits this in a pattern-match compiler:

1. **Parse the block into typed terms.** The symbolic analyser walks
   the SymPy expression tree (implemented in
   `Solverz/code_printer/python/module/mutable_mat_analyzer.py`). Each
   additive term is classified as one of:

   | Shape | Generated code contribution |
   |---|---|
   | `Diag(u_expr)` (diagonal term) | `data[out_pos] = u[src_idx]` |
   | `Mat_Mul(Diag(v), M)` (row-scale) | `data[out_pos] += v[src_row] * M_data[k]` |
   | `Mat_Mul(M, Diag(v))` (col-scale) | `data[out_pos] += v[src_col] * M_data[k]` |

   Unrecognised terms are deferred to a correct-but-slower fallback
   that evaluates the full sub-expression via scipy.sparse and extracts
   values by fancy indexing.

2. **Precompute the output sparsity pattern (once).** At model
   initialisation (inside `FormJac`, `Solverz/equation/equations.py`),
   every mutable-matrix block is evaluated *one* time with all
   variables perturbed to distinct non-zero values and all `Diag`
   nodes substituted with `SpDiag` (which emits `sps.diags` instead of
   the dense `np.diagflat`). This perturbed evaluation is the entire
   reason we go to the trouble of the `SpDiag` substitution: using
   non-zero variable values *guarantees* that no term accidentally
   collapses to zero at $y_0$ (which would happen in a flat-start power
   flow where $f = 0$ and `sps.diags(f)@B` is empty). The resulting
   sparse matrix gives us the **union of all terms' patterns** — the
   structural upper bound that the runtime block must fit into.

3. **Precompute the index mapping arrays (once).** For each recognised
   term, the analyser builds three numpy arrays:

   - `out_pos[k]` — the index into the block's output `data` array
     where the $k$-th nonzero contribution of this term lands.
   - `src_idx[k]` — for diagonal terms, the row $i$ in the inner
     vector $u$ to read from; for row-scale terms, the row of $M$'s
     $k$-th nonzero; for col-scale terms, the column.
   - `mat_data` (row/col-scale terms only) — a direct reference to
     `M.data` (which is why the matrix must be immutable).

   These arrays are stored inside `setting`, a per-module dict that
   also holds things like the CSC `row`, `col`, and initial `data`
   for the full Jacobian.

4. **Generate a dedicated Numba kernel per block.** For each mutable
   matrix block the code printer emits a new top-level function named
   `_mut_block_N`. Its signature takes the diag inner vectors, the
   base variables needed by any row/col-scale term, and each term's
   three mapping arrays. The body is a pure-Python / Numba scatter-add
   loop:

```python
# Module-level (loaded once from setting at import time):
_mb0_diag_out_0 = setting["_mb0_diag_out_0"]
_mb0_rs_out_0   = setting["_mb0_rs_out_0"]
_mb0_rs_src_0   = setting["_mb0_rs_src_0"]
_mb0_rs_dat_0   = setting["_mb0_rs_dat_0"]   # frozen G_nr.data
...

@njit(cache=True)
def _mut_block_0(_mb0_u0, e, f,
                 _mb_diag_out_0, _mb_diag_src_0,
                 _mb_rs_out_0,  _mb_rs_src_0,  _mb_rs_dat_0,
                 _mb_rs_out_1,  _mb_rs_src_1,  _mb_rs_dat_1):
    data = zeros(107)
    # Diagonal term: diag(G_nr@e - B_nr@f + p_ref)
    for i in range(_mb_diag_out_0.shape[0]):
        data[_mb_diag_out_0[i]] = _mb0_u0[_mb_diag_src_0[i]]
    # Row-scale term: diag(e) @ G_nr
    _v = e
    for i in range(_mb_rs_out_0.shape[0]):
        data[_mb_rs_out_0[i]] += _v[_mb_rs_src_0[i]] * _mb_rs_dat_0[i]
    # Row-scale term: diag(f) @ B_nr
    _v = f
    for i in range(_mb_rs_out_1.shape[0]):
        data[_mb_rs_out_1[i]] += _v[_mb_rs_src_1[i]] * _mb_rs_dat_1[i]
    return data
```

5. **Wire the kernel up in the `J_` wrapper.** For each block the
   wrapper emits: (a) one line per diag term that materialises its
   inner vector using scipy.sparse (e.g. `_mb0_u0 = p_ref - (B_nr@f) +
   (G_nr@e)`); (b) a call to `_mut_block_N(...)` passing the inner
   vectors, base variables, and mapping arrays; (c) an assignment of
   the returned `data` array into the appropriate slice of the overall
   Jacobian `_data_`.

   Crucially, each block's **scipy work is capped at one SpMV per diag
   term** — nothing else in the block touches sparse matrices at
   runtime. The data for row/col-scale terms comes straight from the
   pre-baked `_mb_rs_dat_k` arrays.

The net effect on a Newton step is that the Jacobian assembly cost is
dominated by (i) the constant `inner_J` call for element-wise blocks
and (ii) a handful of SpMVs for the diag inner vectors. The actual
mutable-matrix blocks themselves are assembled in
$\mathcal{O}(\text{nnz})$ vectorised Numba loops — no sparse matrix
allocation, no sparsity reconstruction, no format conversion.

#### Why not fancy indexing?

An earlier iteration of the pipeline used `expr.tocsr()[[row],[col]]`
fancy indexing — build the sparse matrix with `sps.diags` and read
back the nonzero values at the precomputed COO positions. It is
correct and elegant but slow: scipy has to construct several
intermediate sparse matrices per step, sum them (with explicit-zero
elimination), and then perform a linear-scan lookup for each output
position. On a 30-bus power flow case, each Jacobian call took ≈ 280
µs. The scatter-add Numba path drops that to ≈ 50 µs while producing
bit-identical results.

The fallback path still uses fancy indexing, so if you write an
equation whose Jacobian term structure the analyser doesn't recognise,
you get *correct* results — just slower. Profile with
`pytest --durations=10` and look for mutable-matrix blocks lingering
in the slow path if performance matters.

### Layer 3 — Selective `@njit`

All generated inner loops — `inner_F`, every per-equation
`inner_F{N}`, `inner_J`, every per-block `inner_J{N}`, and every
`_mut_block_{N}` — carry `@njit(cache=True)`. Only the `F_` / `J_`
wrappers remain pure Python; they host the scipy.sparse SpMV
precomputes, the module-level parameter unpacks, and the final
`sps.coo_array((data, (row, col)), shape).tocsc()` that packages the
Jacobian back into a scipy object for the downstream Newton solver.

The selectivity matters: Numba absolutely cannot digest a
`scipy.sparse.csc_array` argument, which is why the wrappers exist at
all. Anything that *can* run inside Numba is pushed there, at zero
syntactic cost to the user — they just write `Mat_Mul(A, x)` and
`module_printer` figures out the rest.

An important corollary for non-`Mat_Mul` models: if an equation system
contains **no** `Mat_Mul` at all, the entire precompute / mutable-
matrix machinery is disabled and `F_` / `J_` collapse back to the
original pre-0.8 architecture (direct `@njit` on `inner_F` receiving
only dense vectors and scalar params). Performance on pure
element-wise models is therefore unchanged by the Mat_Mul rewrite.

### Fallback path

If a Jacobian block contains a term whose shape the symbolic analyser
does not recognise — for example an unusual nested `Mat_Mul`, a
non-parameter matrix, or a matrix-valued non-linear expression — that
single block is handled by a slower but always-correct path:

```python
data[start:stop] = asarray(
    (sps.diags(u) + sps.diags(v)@M + ...)
    .tocsr()[[rows], [cols]]
).ravel()
```

This builds the block with scipy.sparse and reads its values at the
precomputed COO positions. It is $\mathcal{O}(\text{nnz} \log n)$ per
call instead of the $\mathcal{O}(\text{nnz})$ scatter-add path, but it
never misses a derivative term. Only the single affected block falls
back; the rest of the Jacobian still uses the fast path.

If you want to know whether your model is hitting the fallback,
profile with `pytest --durations=20` or `cProfile` and look for
`MutableMatJacDataModule` in the traces — its presence in the hot
`J_()` line indicates a fallback block.

### Performance

Measurements below are per `J_()` call on a 2025 MacBook Air (Apple
M4), averaged over 5000–10000 iterations.

**Case A — rectangular power flow on MATPOWER `case30`**
(58 unknowns, 450 Jacobian nonzeros, 6 mutable-matrix blocks, the flow
has meaningful $G\,e$, $B\,f$, $B\,e$, $G\,f$ combinations):

| Pipeline | `J_()` per call |
|---|---|
| Inline mode (lambdify + scipy SpMV) | ≈ 268 µs |
| Module printer, old `tocsr()[rows, cols]` | ≈ 283 µs |
| **Module printer, vectorized @njit** | **≈ 50 µs** |

**Case B — DHS hydraulic subproblem on `BarryIsland`**
(35 unknowns, 1 loop, 1 mutable-matrix block — the loop pressure
Jacobian contains two col-scale terms
$L\,\operatorname{diag}(K \odot |m|)$ and
$L\,\operatorname{diag}(K \odot m \odot \operatorname{sign}(m))$ whose
scaling vectors are non-trivial expressions of the variable):

| Pipeline | `J_()` per call |
|---|---|
| Element-wise inline (one scalar `Eqn` per node + loop) | ≈ 103 µs |
| Mat_Mul inline (lambdify) | ≈ 45 µs |
| **Mat_Mul module printer, vectorized @njit** | **≈ 28 µs** |

- **Element-wise vs Mat_Mul inline.** Writing the same hydraulic
  system as `V @ m - m_inj = 0` plus `L @ (K m |m|) = 0` (two vector
  equations total) instead of 35 scalar equations cuts inline time by
  more than half, because lambdify has far fewer expression nodes to
  walk at runtime.
- **Inline vs module printer.** Even on a small system (only 79
  nonzeros), the module printer wins once every mutable-matrix term
  reaches the vectorised scatter-add path. Here the loop-pressure
  Jacobian has two col-scale terms whose scaling vectors
  (``K * |m|`` and ``K * m * sign(m)``) are computed once per `J_`
  call as dense numpy expressions in the wrapper, then handed to a
  Numba kernel that does two 10-iteration scatter-add loops. The only
  remaining overhead is the final `coo_array(...).tocsc()` packing.

Rule of thumb: **module printer wins whenever the mutable-matrix
blocks successfully match the fast path**; it can lose to inline for
models that fall through to the fallback (scipy-sparse + fancy
indexing), or models small enough that the final COO-to-CSC packing
dominates. Profile with `%timeit mdl.J(y, mdl.p)` on a representative
iterate if in doubt, and check the generated `num_func.py` for any
lingering `MutableMatJacDataModule` (= fallback) calls.

The scatter-add path is strictly $\mathcal{O}(\text{nnz})$ in the block
contents, while the old fancy-indexing path was dominated by the cost
of constructing the intermediate sparse matrices — so the speed-up
scales with nnz and with the number of mutable-matrix blocks. For
models with many rows and few blocks the advantage is small; for
models with many blocks per variable (such as power flow, with
distinct $\partial/\partial e$ and $\partial/\partial f$ blocks for
both $P$ and $Q$ balances) the advantage compounds.

## API Reference

```{eval-rst}
.. autofunction:: Solverz.sym_algebra.matrix_calculus.MixedEquationDiff
.. autoclass:: Solverz.sym_algebra.matrix_calculus.TensorExpr
   :members: diff, visualize
```
