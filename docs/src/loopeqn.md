(loopeqn)=

# Indexed equations with `LoopEqn`

`LoopEqn` lets you write one equation template and have Solverz
instantiate it at every index of a set — the same idea as a `for`
loop around a scalar `Eqn`, except Solverz generates *one* compiled
sub-function that contains the loop, not one sub-function per
iteration. This page walks through the motivation, the minimal
syntax, and the four worked examples from the Solverz Cookbook that
cover the patterns you'll hit in practice.

## 1. Why `LoopEqn`?

A power-flow bus balance, a heat-network node energy balance, a gas
pipeline pressure drop — every one of these is "write the same
scalar residual for each index in some set." Before `LoopEqn`, the
obvious Solverz spelling was a Python loop that creates one
`Eqn` per bus:

```python
for i in range(nb):
    m.__dict__[f'balance_{i}'] = Eqn(f'balance_{i}', ...)
```

When `module_printer` renders this model, each scalar `Eqn` becomes
its own `@njit`-compiled sub-function inside `num_func.py`. The 33-
bus IEEE test feeder produces ≈70 of them. The 225-node network
scales linearly. On a cold Numba cache, the LLVM compile-time bill
dominates the warmup phase — minutes to get through the first
simulation step.

`LoopEqn` collapses all those scalar sub-functions into one:

```python
i = Idx('i', nb)
m.balance = LoopEqn(
    'balance',
    outer_index=i,
    body=body_expression,  # a scalar sympy expression in i
    model=m,
)
```

The generated module contains **one** `inner_F<N>` with an explicit
`for i in range(nb):` inside. One LLVM compilation instead of `nb`.
On the full 8996-DOF "Big" IES benchmark the cold-cache wall shrinks
from ≈300 s to ≈150 s — see [#134](https://github.com/smallbunnies/Solverz/issues/134)
for the measured table.

## 2. Hello `LoopEqn`

Solve $x_i^2 - r_i = 0$ for $i = 0, 1, 2$ with pinned right-hand side
values.

```python
import numpy as np
from Solverz import Eqn, Idx, LoopEqn, Model, Param, Var, module_printer, nr_method

n = 3
target = np.array([1.0, 2.0, 3.0])

m = Model()
m.x = Var('x', np.array([0.8, 2.3, 2.7]))          # warm-start
m.r = Param('r', target ** 2)

i = Idx('i', n)                                    # bounded index
m.eq = LoopEqn('eq', outer_index=i,
               body=m.x[i] ** 2 - m.r[i],
               model=m)

spf, y0 = m.create_instance()
import tempfile, importlib, sys
with tempfile.TemporaryDirectory() as d:
    module_printer(spf, y0, 'hello_loopeqn',
                   directory=d, jit=True).render()
    sys.path.insert(0, d)
    mdl = importlib.import_module('hello_loopeqn').mdl
    y   = importlib.import_module('hello_loopeqn').y
    sol = nr_method(mdl, y)

print(sol.y['x'])  # → [1., 2., 3.]
```

Three pieces to notice:

- `Idx('i', n)` is a shortcut for `sympy.Idx('i', (0, n-1))`. The
  bounded form carries its size on the index, so both `Sum` and
  `LoopEqn` can pick it up without repeating `n`.
- The **body** is a scalar sympy expression whose symbols are Solverz
  `Var` / `Param` references. Inside the body `m.x[i]` is a scalar:
  "the value of `x` at position `i`."
- Passing `model=m` lets `LoopEqn` resolve every `m.<name>` it sees
  in the body without the user having to list them. The other way
  (explicit `var_map={'x': m.x, 'r': m.r}`) is functionally
  equivalent — use whichever reads more clearly.

## 3. Sums inside bodies

When the residual for row `i` involves a sum over *another* axis —
e.g. a matrix-vector product — use `Sum`:

:::{math}
F_i \;=\; \sum_{j=0}^{n-1} A_{ij}\,x_j - b_i
:::

```python
from Solverz import Idx, LoopEqn, Model, Param, Sum, Var
import numpy as np

n = 3
m = Model()
m.x = Var('x', np.zeros(n))
m.A = Param('A', np.random.default_rng(0).random((n, n)), dim=2, sparse=True)
m.b = Param('b', np.ones(n))

i, j = Idx('i', n), Idx('j', n)
m.row_balance = LoopEqn(
    'row_balance',
    outer_index=i,
    body=Sum(m.A[i, j] * m.x[j], j) - m.b[i],
    model=m,
)
```

`Sum(expr, j)` is a shortcut for `sympy.Sum(expr, (j, 0, j.upper))`
when `j` is a bounded `Idx`. Pass an explicit `n` as the third
argument if `j` is unbounded: `Sum(expr, j, n)`.

Since `m.A` was declared with `sparse=True`, the body-printer
recognises the `A[i, j] * x[j]` pattern and emits a **CSR row walk**
instead of a dense `for j in range(n)` loop. You don't have to do
anything special — the per-row loop iterates only the non-zeros of
row `i`, which matters a lot for admittance-matrix-sized problems.

## 4. Worked example — polar power-flow P/Q balance

The Solverz Cookbook at
[`docs/source/ae/pf/src/pf_mdl_loopeqn.py`](https://cookbook.solverz.org/latest/ae/pf/pf.html)
solves the polar-form power-flow equations for the case30 benchmark
using `LoopEqn`. The active-power residual at a non-slack bus is

:::{math}
F^P_i \;=\; V_i \sum_{k=0}^{n_b-1} V_k \bigl(G_{ik}\cos(\theta_i - \theta_k)
                                  + B_{ik}\sin(\theta_i - \theta_k)\bigr)
          + P_{d,i} - P_{g,i}
\qquad i \in \text{PV} \cup \text{PQ}.
:::

The cookbook's implementation declares bus subsets via `int` Params
and drives the outer index through an indirect-index pattern:

```python
m.pv_pq_idx = Param('pv_pq_idx', pv_pq_arr, dtype=int)   # free buses
i_p = Idx('i_p', len(pv_pq_arr))
j   = Idx('j', nb)

body_P = (
    m.Vm_full[m.pv_pq_idx[i_p]]
    * Sum(m.Vm_full[j] * m.Gbus[m.pv_pq_idx[i_p], j]
          * cos(m.Va_full[m.pv_pq_idx[i_p]] - m.Va_full[j]), j)
    + m.Vm_full[m.pv_pq_idx[i_p]]
    * Sum(m.Vm_full[j] * m.Bbus[m.pv_pq_idx[i_p], j]
          * sin(m.Va_full[m.pv_pq_idx[i_p]] - m.Va_full[j]), j)
    + m.Pd[m.pv_pq_idx[i_p]] - m.Pg[m.pv_pq_idx[i_p]]
)
m.P_eqn = LoopEqn('P_eqn', outer_index=i_p, body=body_P, model=m)
```

Every access to a bus-indexed quantity goes through
`m.pv_pq_idx[i_p]`. That's five repetitions of the same index map.
Section 5 shows the `Set`-based rewrite that collapses them.

## 5. Iterating over a subset — `Set`

`Set` names an index set. Two construction forms:

```python
from Solverz import Set
import numpy as np

m.Bus  = Set('Bus', nb)                          # full range [0, nb)
m.PVPQ = Set('PVPQ', np.array(pv + pq, dtype=int))   # explicit subset
```

Accepted value types for the subset form: `int` (becomes a full
range `[0, n)`), any 1-D integer sequence (`np.ndarray[int]`,
`list[int]`, `tuple[int]`, `range`). Values must be non-negative and
duplicate-free.

Each `Set` exposes `.idx(name)` to produce a bounded `sympy.Idx`:

```python
i = m.PVPQ.idx('i')   # sp.Idx with range [0, len(PVPQ))
```

When the body uses such an `Idx`, `LoopEqn` automatically rewrites
every access `m.Var[i]` to `m.Var[PVPQ_param[i]]` — the indirect-
outer pattern that the sparsity analyser recognises. The cookbook's
P-balance body written with `Set` becomes:

```python
m.Bus  = Set('Bus', nb)
m.PVPQ = Set('PVPQ', np.array(pv + pq, dtype=int))

i = m.PVPQ.idx('i')
j = m.Bus.idx('j')

body_P = (
    m.Vm_full[i]
    * Sum(m.Vm_full[j] * m.Gbus[i, j]
          * cos(m.Va_full[i] - m.Va_full[j]), j)
    + m.Vm_full[i]
    * Sum(m.Vm_full[j] * m.Bbus[i, j]
          * sin(m.Va_full[i] - m.Va_full[j]), j)
    + m.Pd[i] - m.Pg[i]
)
m.P_eqn = LoopEqn('P_eqn', outer_index=i, body=body_P, model=m)
```

Two declarations (`m.Bus`, `m.PVPQ`) replace the three-object dance
(`Param` + `Idx` + five `m.pv_pq_idx[i_p]` expansions). Runtime
behaviour is identical — the `Set` is sugar that lowers to the same
indirect-outer CSR walk.

`m.Bus = Set('Bus', nb)` is an **identity** set — its values are
exactly `[0, nb)`, so no auxiliary Param is materialised and the
translator emits the direct `for j in range(nb):` loop (zero
overhead vs the plain `Idx('j', nb)` form).

## 6. Worked example — integrated energy system

The cookbook's
[`docs/source/dae/ies/src/test_ies.py`](https://cookbook.solverz.org/latest/dae/ies/ies.html)
wires the three network blocks (`eps_network`, `heat_network`,
`gas_network`) into one coupled DAE and integrates it with `Rodas`.
Each of the network blocks internally uses `LoopEqn` / `LoopOde`:
the heat network has `Mass_flow_continuity_sup` and `Ts_mixing` as
LoopEqns over the node set, and `heat_pipe_s` as a cross-pipe
LoopOde over all cell offsets. You don't write these by hand — you
call `heat_network(...).mdl(loopeqn=True)` and `m.add(...)` the
result — but the docs point you at the source because it's the
canonical example of how `LoopEqn` / `LoopOde` scale to a
production model. On the 8996-DOF Big system, using `loopeqn=True`
makes the full Rodas run **3.6× faster** than the scalar-`Eqn`
expansion, measured end-to-end (see the `bench` script alongside the
test).

## 7. `LoopOde`: the `LoopEqn` of time derivatives

Most users meet `LoopEqn` first, then `LoopOde` when they start
writing DAEs. `LoopOde` is the ODE sibling — same body template,
same `for`-loop codegen, but the left-hand side is $\dot{y}_i$
instead of a residual. Signature:

```python
LoopOde(name, outer_index=i, body=rhs, diff_var=m.y, model=m)
```

The heat pipe PDE in the cookbook IES example is a `LoopOde` whose
outer index iterates every `(pipe, segment)` cell. The `diff_var`
is the flat state vector `m.Tsp_all`. See
`SolMuseum/dae/heat_network.py::_mdl_loopeqn` for the assembly.

## 8. Use `module_printer(jit=True)`

`LoopEqn` is only fast with Numba JIT compilation. The generated
body contains an explicit Python `for` loop, and the only way to
turn it into a tight native loop is Numba. Running it through the
inline path `made_numerical(...)` — or `module_printer(jit=False)` —
executes the loop at Python interpreter speed, which on the Barry
IES benchmark is 3–7× slower per F/J call than the vectorised
lambdify path the legacy scalar-`Eqn` uses.

To keep users out of this trap, `made_numerical` raises
`NotImplementedError` when the equation system contains any
`LoopEqn`. The only supported render is:

```python
module_printer(spf, y0, name='my_model',
               directory='.', jit=True).render()
```

## 9. Performance expectations

From `Solverz-Cookbook/docs/source/ae/pf/src/bench_loopeqn_pf.py`
on case30 (29-bus polar PF, 53 residuals):

| metric | `loopeqn=False` | `loopeqn=True` | ratio |
|---|---:|---:|---:|
| Cold Numba compile | 45 s | 1.2 s | **38× faster** |
| `@njit` sub-functions | 54 F + 362 J | 4 F + 1 J | **416 → 5** |
| Hot F eval | 1.11 µs | 2.49 µs | 2.2× slower |
| Hot J eval | 52.8 µs | 37.2 µs | **1.4× faster** |

The warmup wins are structural — you get them for any LoopEqn
model. The F / J per-call ratios depend on the body shape. Bodies
with sparse CSR walkers (like PF) tend to beat the lambdify
baseline on J; dense element-wise bodies tend to come out close to
parity on F.

## 10. Troubleshooting

**`NotImplementedError: made_numerical does not support LoopEqn`**
The inline path runs `LoopEqn` bodies at Python speed, which is
much slower than the lambdify-vectorised fallback the legacy
`Eqn` uses. Switch the render to
`module_printer(spf, y0, ..., jit=True).render()` and import the
generated module.

**`ValueError: LoopEqn body references symbol 'X' but model has no attribute by that name.`**
The body uses `m.X[i]` but `m.X = Var('X', ...)` was declared
*after* the `LoopEqn` was constructed. Move all `Var` / `Param`
declarations above the `LoopEqn(...)` call.

**`IndexSet 'Sub': duplicate values not allowed`**
`Set` requires a non-repeating index sequence — the subset has to
be an injection into the target variable's index space. If you
need a gathered view that visits indices in any order (including
repeats), use a plain `Param('...', arr, dtype=int)` and reference
it by hand as `m.V[m.arr[i]]`.

**`NameError: name '<param>' is not defined` at runtime**
A symbol appears in the body but isn't in the lambdified function's
argument list. The usual cause is a `Set.idx` access on a `Var` /
`Param` not declared before the `LoopEqn`. Declare first, reference
second.

## Further reading

- {class}`Solverz.equation.eqn.LoopEqn` — full API.
- {class}`Solverz.equation.eqn.LoopOde` — ODE sibling.
- {class}`Solverz.equation.eqn.IndexSet` (`Solverz.Set`) — the set
  primitive.
- {ref}`LoopEqn translator appendix <loopeqn_translator>` — Phase J2
  coverage table + supported / unsupported body shapes, for readers
  extending the Jacobian translator itself.
- Tracking issues:
  [`#128`](https://github.com/smallbunnies/Solverz/issues/128) (prototype),
  [`#129`](https://github.com/smallbunnies/Solverz/issues/129) (`Set`),
  [`#133`](https://github.com/smallbunnies/Solverz/issues/133)
  (translator extensions),
  [`#134`](https://github.com/smallbunnies/Solverz/issues/134)
  (benchmark tracker).
