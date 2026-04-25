# Phase 0 findings

Empirical log of what works and what doesn't as we drive the smoke test
in `tests/test_loop_eqn.py` from "every stage fails" to "every stage
passes".

## Iteration 0 — sp.Matrix approach (abandoned)

Before pivoting to `LoopEqn`, the original Phase 0 plan tried to make
`Eqn('bc', sp.Matrix([Tsp_0[0], Tsp_1[0]]) - Mat_Mul(V, Ts))` work.
That smoke test (`tests/test_sp_matrix_in_eqn.py`, since deleted)
failed at sympy's expression construction:

```text
TypeError: unsupported operand type(s) for +: 'MutableDenseMatrix' and 'Mul'
  at sympy/matrices/matrixbase.py:3047  (sp.Matrix.__sub__ → self + (-a))
```

`sp.Matrix.__sub__` only accepts matrix-compatible operands; `Mat_Mul`
is a sympy `MatrixFunction` / `Function`, not a sympy Matrix. The
expression doesn't even build, let alone reach `obtain_TExpr`.

Workaround `sp.Add(sp.Matrix([...]), sp.Mul(-1, Mat_Mul(V, Ts)))`
constructs successfully, but follow-up probing showed Solverz's
`IdxVar` and `IdxPara` are opaque to sympy's `subs` and `Sum.doit()`:

- `(G[i, j] * x[j]).subs(j, 1)` returns the unchanged expression
  `G[i, j] * x[j]` — sympy can't traverse into Solverz's custom
  symbol types
- `Sum(x[j], (j, 0, 2)).doit()` collapses incorrectly to `3 * x[j]`
  instead of expanding to `x[0] + x[1] + x[2]`

Verdict: `sp.Matrix` + Solverz `IdxVar` is fundamentally unworkable.
Pivot to `LoopEqn` with sympy-native `IndexedBase` substrate.

## Iteration 1 — LoopEqn import failure (current)

`tests/test_loop_eqn.py` at `from Solverz import LoopEqn`:

```text
ImportError: cannot import name 'LoopEqn' from 'Solverz'
```

**Expected.** `LoopEqn` doesn't exist in stock 0.8.3. This is the
starting point for Hypothesis A.

### Substrate verification

Before diving into the implementation, I verified the underlying
sympy-native primitives work:

```python
import sympy as sp
i, j, k = sp.Idx('i'), sp.Idx('j'), sp.Idx('k')
G = sp.IndexedBase('G')
x = sp.IndexedBase('x')

# All work as expected:
sp.Sum(G[i,j]*x[j], (j, 0, 2)).doit()
# → G[i,0]*x[0] + G[i,1]*x[1] + G[i,2]*x[2]

(G[i,j] * x[j]).subs(j, 1)
# → G[i,1] * x[1]

sp.Sum(G[i,j]*x[j], (j, 0, 2)).diff(x[1])
# → Sum(KroneckerDelta(1, j) * G[i, j], (j, 0, 2))
# .doit() → Piecewise((G[i, 1], True), (0, True))

sp.Sum(G[i,j]*x[j], (j, 0, 5)).diff(x[k])
# → Sum(KroneckerDelta(j, k) * G[i, j], (j, 0, 5))
# .doit() → Piecewise((G[i, k], (k >= 0) & (k <= 5)), (0, True))
```

So differentiation via sympy on `IndexedBase` slots works correctly
and yields `Sum(KroneckerDelta * coefficient, ...)` which simplifies
via `.doit()` to the per-element coefficient (wrapped in a bounds
Piecewise). This is exactly the structure we need for the per-(eqn,
var) Jacobian block.

**Caveat**: `lambdify((G, x, i), Sum(G[i,j]*x[j], (j,0,4)), modules='numpy')`
fails with:

```text
PrintMethodNotImplementedError: Unsupported by NumPyPrinter: <class 'sympy.tensor.indexed.Idx'>
```

So sympy's standard `lambdify` cannot translate a `Sum` over an `Idx`
into numpy code. The `LoopEqn` printer must be **fully custom** — it
walks the body, replaces `Sum` with explicit Python `for`-loops, and
replaces `IndexedBase[i, j]` accesses with the corresponding Var/Param
indexing pattern. This is the meat of Hypothesis A.

## Hypothesis A — implement `LoopEqn` in Solverz core (PASSING)

Phase 0 minimum smoke test is GREEN. Full Solverz test suite (105
tests) still passes — no regressions. Recap of what landed:

### Step 1 — `LoopEqn(Eqn)` subclass

`Solverz/equation/eqn.py`:

- Stores `body`, `outer_index`, `n_outer`, `var_map`. Bypasses
  `Eqn.__init__` because `sympy.lambdify` cannot translate `Sum` over
  `Idx` (`PrintMethodNotImplementedError` from `NumPyPrinter`).
- Builds `SYMBOLS` from `var_map` (lex-sorted `iVar`/`Para` for the
  Var/Param the body references).
- `mixed_matrix_vector = False` — `LoopEqn` deliberately does not use
  `Mat_Mul`.

### Step 2 — custom F-side `NUM_EQN`

`_build_num_eqn()` in `eqn.py`:

- Walks the body via `_translate_loop_body()` and emits a Python
  source string of the form
  ```python
  def _loop_eqn_func(<lex-sorted arg names>):
      out = np.empty(n_outer)
      for i in range(n_outer):
          out[i] = <translated body>
      return out
  ```
- `_translate_loop_body` handles `Sum`, `Add`, `Mul`, `Pow`,
  `Indexed`, `Idx`, `Symbol`, and number literals. `Sum` becomes a
  Python generator-`sum` expression; `IndexedBase[...]` accesses are
  left as-is so they resolve to numpy fancy-indexing on the bound
  argument.

### Step 3 — F-side wiring through Solverz's inline printer

`Solverz/code_printer/python/utilities.py print_eqn_assignment` —
when `isinstance(eqn, LoopEqn)`, emit
```python
_F_[<addr>] = _sz_loop_<eqn_name>(B, G, ix_pin, iy_pin, ux, uy)
```
instead of the default `Assignment(_F_[addr], eqn.RHS)` (which would
trigger the `pycode` `Idx` failure).

`Solverz/code_printer/python/inline/inline_printer.py made_numerical`
— register the LoopEqn `NUM_EQN` callables in `custom_func` keyed by
`_sz_loop_<name>` before the `Solverzlambdify` call, so the
generated `F_` can resolve them.

### Step 4 — J-side via symbolic differentiation through sympy `diff`

`LoopEqn.derive_derivative()` in `eqn.py`:

- For each `var_map` entry that is a Solverz `Var` (NOT a `Param` —
  `Param(ParamBase, sSymBasic)` inherits from `sSymBasic`, so the
  filter must use `isinstance(_, Var)` directly, not `sSymBasic`),
  builds a fresh `Idx('_sz_loop_dk')`, constructs the target slot
  `IndexedBase(name)[k]`, and calls `sp.diff(body, target).doit()`.
- `sp.piecewise_fold(...)` normalizes any leading `Mul(-1, …)` into
  the Piecewise branches so the translator sees a uniform shape.
- `_translate_loop_jac()` strips the `Piecewise` bounds wrapper
  (`(0 ≤ k) & (k ≤ n - 1)` is always satisfied by construction) and
  recursively walks the result, replacing each
  `IndexedBase('M')[outer, k]` with `Para('M', dim=2)`. Currently
  only handles 2-D Param accessed with exactly two indices — anything
  else raises `NotImplementedError` so the next failure is loud.
- The translated Solverz expression is wrapped in a regular
  `EqnDiff(name=…, eqn=…, diff_var=iVar('ux'))`. Because the result
  is `Para` or `-Para`, `JacBlock.is_mutable_matrix` is False and the
  EXISTING constant-matrix code path inlines the data — no new J-side
  printer needed for this pattern.

### Bug fix in the smoke test

The original test used `G_dense` and `B_dense` with zero row sums (a
typical shunt-free admittance matrix), which made the resulting
Newton Jacobian rank-deficient (`det J = 0`, `cond ≈ 2.7e18`). Added
small diagonal shunts so `det J ≈ 395`, `cond ≈ 16`. The math the
LoopEqn machinery enforces is unchanged.

## Iteration 4 — Phase 0.4.3: module_printer(jit=True) JIT path (PASSING)

`tests/test_loop_eqn.py::test_loop_eqn_eps_minimal_jit_module`
exercises the full JIT-compiled module path that SolMuseum users
actually use (the IES benchmark is `module_printer(jit=True)`).

### Failure mode reproduced first

Same root cause as the inline F-side: ``pycode`` cannot translate
``Sum`` over ``Idx``. Stock ``print_sub_inner_F`` builds
``Return(eqn.RHS)`` for non-Mat_Mul equations and feeds it through
``pycode``. For a LoopEqn whose ``RHS`` *is* the
``Sum``-over-``IndexedBase`` body, this raises
``PrintMethodNotImplementedError: Unsupported by PythonCodePrinter:
Idx``.

### Implementation: `LoopEqn.print_njit_source`

`Solverz/equation/eqn.py`:

- New method ``LoopEqn.print_njit_source(func_name)`` returns a
  hand-built Python source string for the ``inner_F<N>``
  sub-function. Skips the AST + pycode pipeline entirely.
- New helper ``_translate_loop_body_njit(expr, state)`` is a
  Numba-compatible variant of ``_translate_loop_body``. Numba's
  ``@njit`` does NOT support generator expressions, so the
  ``sum(... for j in range(...))`` form from
  ``_build_num_eqn`` cannot be reused. Instead, each ``sympy.Sum``
  is hoisted into an explicit accumulator variable + explicit
  inner ``for`` loop, both placed inside the outer-loop body of
  the generated ``inner_F<N>``:
  ```python
  for i in range(n_outer):
      _sz_loop_acc_0 = 0.0
      for j in range(0, n_inner):
          _sz_loop_acc_0 += <translated body>
      out[i] = <translated outer expression with acc_0 substituted>
  ```
- Nested ``Sum``s (a Sum inside the body of another Sum) raise
  ``NotImplementedError`` — the inner accumulator would need to
  reset on each iteration of the outer dummy, which the linear
  prelude approach does not capture. SolMuseum's network modules
  don't currently use nested Sums; if a real use case appears, the
  fix is to emit a properly nested for-loop block instead of a
  flat prelude.

### Wiring: `print_sub_inner_F` LoopEqn branch

`Solverz/code_printer/python/module/module_printer.py`:

- New branch at the top of ``print_sub_inner_F``'s loop:
  ```python
  if isinstance(eqn, LoopEqn):
      arg_names = sorted(eqn.SYMBOLS.keys())
      args = [symbols(v, real=True) for v in arg_names]
      code_blocks.append(eqn.print_njit_source(f'inner_F{count}'))
      precompute_info.append({...})
      count += 1
      continue
  ```
- The lex-sorted arg names match what ``print_eqn_assignment``
  emits at the call site (``inner_F<N>(arg1, arg2, ...)``), so the
  dispatcher and the sub-function agree on the argument order.

### What the generated module looks like

```python
@njit(cache=True)
def inner_F0(B, G, ix_pin, iy_pin, ux, uy):
    out = np.empty(3)
    for i in range(3):
        _sz_loop_acc_0 = 0.0
        for j in range(0, 3):
            _sz_loop_acc_0 += ((ux[j] * G[i, j])
                               + (-1.0 * uy[j] * B[i, j]))
        out[i] = ((-1.0 * _sz_loop_acc_0) + ix_pin[i])
    return out
```

### J-side: nothing extra needed

Because the smoke test's per-(LoopEqn, Var) Jacobian blocks are
all bare ``±Para`` (constant matrices), the JacBlock classifier
hits the constant-data path and the entire J reduces to a
precomputed ``_data_`` blob loaded from the module's ``setting``
file. ``inner_J`` becomes a one-liner:

```python
@njit(cache=True)
def inner_J(_data_, ux, uy, B, G, ix_pin, iy_pin):
    return _data_
```

— exactly the best-case-scenario the LoopEqn refactor is supposed
to produce for the IES benchmark's eps_network dyn=True block.

### Compile-time evidence

Module compile (numba LLVM): **~1.0 s** for the entire 2-LoopEqn
model (3-bus eps rectangular current balance). The original
SolMuseum for-loop pattern would have produced ``2 × nb = 6``
separate ``inner_F<N>`` functions for this minimum case, each
compiled separately. With LoopEqn there are exactly 2.

## Iteration 3 — Phase 0.4.2: heat/gas mass continuity (PASSING, NO new code)

`tests/test_loop_eqn.py::test_loop_eqn_mass_continuity` exercises
the SolMuseum heat/gas pattern:

```python
body_mass = f_inj[i] + sum_p (V_in[i, p] - V_out[i, p]) * m_pipe[p]
```

where ``f_inj`` is a 1-D Param, ``V_in`` / ``V_out`` are 2-D
incidence Params, and ``m_pipe`` is a 1-D Var accessed at the inner
sum dummy ``p``. **The existing translator handles it without any
extension.** Walking through:

- `_translate_loop_body` already handles `IndexedBase[i]` (1-D) and
  `IndexedBase[i, p]` (2-D) — both translate to the bare Python
  expression which numpy evaluates as fancy indexing.
- `_translate_loop_jac` differentiates w.r.t. ``m_pipe[k]``:
  - ``Sum((V_in[i, p] - V_out[i, p]) * m_pipe[p], …).diff(m_pipe[k])``
    = ``Sum((V_in[i, p] - V_out[i, p]) * KroneckerDelta(p, k), …)``
  - ``.doit()`` collapses the KroneckerDelta-Sum →
    ``Piecewise(((V_in[i, k] - V_out[i, k]), bounds), (0, True))``
  - Walker strips Piecewise → ``V_in[i, k] - V_out[i, k]``
  - Walker translates each ``IndexedBase('M')[i, k]`` → ``Para('M',
    dim=2)``, reconstructs as ``Add(Para('V_in', dim=2),
    Mul(-1, Para('V_out', dim=2)))``.

JacBlock sees DeriExpr = ``V_in - V_out``, which is NOT a bare
``Para`` / ``-Para`` → ``is_mutable_matrix = True`` → mutable matrix
path emits ``(V_in - V_out)[[rows], [cols]].ravel().tolist()``.
Sparsity pattern from ``Value0.tocoo()`` correctly captures the 5
non-zero entries of the 3×3 incidence difference, so the printed
COO addresses are exactly those 5 positions — no wasted entries.

(Optimisation note: ``V_in - V_out`` is *actually* constant, so an
ideal printer would inline the data once instead of recomputing each
Newton step. JacBlock's classifier currently only recognises bare
``Para`` / ``-Para`` as constant. Generalising to "DeriExpr has no
free Var symbols" is a follow-up — small win, would help LoopEqn
mass-continuity blocks specifically. Deferred to Phase 0.5+.)

## Iteration 2 — Phase 0.4.1: outer-index Var residual (PASSING)

`tests/test_loop_eqn.py::test_loop_eqn_eps_dyn_with_var_residual`
exercises the actual ``SolMuseum/ae/eps_network.py:88-97``
``mdl(dyn=True)`` pattern, which has an outer-indexed **Var** as the
residual term:

```python
body_re = ix[i] - sum_j (G[i, j] * ux[j] - B[i, j] * uy[j])
```

`ix` is a Var, NOT a Param. The Jacobian block
``∂F[i]/∂ix[k]`` simplifies to ``KroneckerDelta(i, k)`` — i.e. the
``n_outer × n_outer`` identity matrix.

### Implementation: `_LoopJacEye(Function)`

`Solverz/equation/eqn.py`:

- `_LoopJacEye(n)` is a sympy `Function` subclass with `_pythoncode`
  / `_numpycode` returning `np.eye(<n>)`. Has no free symbols, so the
  standard `Eqn.lambdify` machinery wraps it as `lambda: np.eye(n)`.
- `_translate_loop_jac` now takes `n_outer` as an extra arg, and
  detects ``KroneckerDelta(a, b)`` with ``{a, b} == {outer_idx, k}``
  → emits ``_LoopJacEye(sp.Integer(n_outer))``.
- `JacBlock` sees a 2-D Value0 (`np.eye(3)`); because `DeriExpr` is a
  `Function` (not a `Para`), `is_mutable_matrix = True` and the
  mutable-matrix code path takes over. The kernel re-evaluates
  `np.eye(n)` every Newton step, but the cost is negligible (small n,
  result is one ~24-byte numpy alloc).

### Test setup detail

The new test makes the system square by adding two regular `Eqn`
"anchor" rows: `m.ix - m.ix_pin` and `m.iy - m.iy_pin`. This proves
that LoopEqn composes with regular Eqn in the same Model.

Both LoopEqn tests pass; full Solverz test suite is at 106 passing,
no regressions.

## Remaining gaps (Phase 0.4 / 0.5)

- `_translate_loop_jac` still doesn't handle:
  - **1-D Var/Param accessed at the inner sum dummy** — needed by
    heat_network and gas_network mass-continuity patterns of the
    form ``sum_p (V_in[node, p] * m[p] - V_out[node, p] * m[p])``.
  - **1-D Var/Param accessed at the outer index** (other than the
    ``±IndexedBase[outer]`` residual term that 0.4.1 added) —
    needed for ``Vm[i] * Vm[j] * (...)`` polar power balance
    products.
  - **Trig functions of variable differences** — needed for
    polar power flow (``cos(Va[i] - Va[j])``).
  - **Per-element Var-Var products** — the Jacobian becomes a
    mutable matrix that must be re-evaluated each step.
- Module printer (`module_printer(jit=True)`) path is untouched —
  `LoopEqn` only flows through the inline `made_numerical` path so
  far. The custom NUM_EQN uses Python's generator `sum(...)` which
  is incompatible with `@njit`; the JIT path will need a separate
  emitter that produces an explicit nested `for` loop.
- Variable-length per-pipe Vars (`M_j != M_k`), heat-network
  ``phi_eqn`` per-node-type variation (4 disjoint forms), and
  non-rectangular sums via graph-derived edge sets are deferred to
  Phase 0.5 — these are the design questions that the Codex review
  will grade.
