(extend_matrix_calculus)=

# Extending Matrix Calculus

This guide explains how to add new operations to the matrix-vector calculus module.
It is intended for developers and LLM-assisted development.

## Architecture Overview

The matrix calculus module consists of four main components:

```
User Expression (SymPy)
    │
    ▼
obtain_TExpr()          ← Convert SymPy nodes to tensor nodes
    │
    ▼
ToComGraph()            ← Build computation DAG
    │
    ▼
TensorExpr.diff()       ← Forward-mode differentiation on DAG
    │
    ▼
TMul2Mul()              ← Resolve tensor indices to matrix operations
    │
    ▼
Symbolic Derivative (SymPy)
```

**Key files:**
- `Solverz/sym_algebra/matrix_calculus.py` — tensor nodes, DAG, differentiation, TMul2Mul
- `Solverz/sym_algebra/functions.py` — symbolic function definitions with `fdiff()`
- `Solverz/sym_algebra/test/test_matrix_calculus.py` — tests

## Adding a New Element-wise Unary Function

Element-wise unary functions (like `exp`, `sin`, `tanh`) are automatically handled by
the `TUnaryFunc` node. You only need to:

### Step 1: Define the function in `functions.py`

```python
class tanh(UniVarFunc):
    r"""Hyperbolic tangent function."""

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 - tanh(self.args[0])**2
        raise ArgumentIndexError(self, argindex)

    def _numpycode(self, printer, **kwargs):
        return r'np.tanh(' + printer._print(self.args[0]) + r')'
```

**Key requirements:**
- Inherit from `UniVarFunc`
- Implement `fdiff()` returning the derivative **without** the chain rule
  (the chain rule is handled automatically by the tensor calculus)
- Implement `_numpycode()` for numerical code generation

### Step 2: No changes needed in `matrix_calculus.py`

The `TUnaryFunc` node automatically recognizes all `UniVarFunc` subclasses
via `isinstance(expr, UniVarFunc)` in `obtain_TExpr()`.

### Step 3: Add a test

```python
from Solverz.sym_algebra.functions import tanh

A = Para('A', dim=2)
x = iVar('x')
expr = tanh(Mat_Mul(A, x))
te = TensorExpr(expr)
# d/dx tanh(A@x) = diag(1 - tanh(A@x)**2) @ A
assert te.diff(x).__repr__() == 'diag(1 - tanh(A@x)**2)@A'
```

## Adding a New Node Type

For operations that are **not** element-wise unary functions, you need to create
a new tensor node class. Here is a step-by-step guide using a hypothetical
matrix trace operation as an example.

### Step 1: Create the T-node class

```python
class TTrace:
    """Trace node: tr(A) = sum of diagonal elements.
    
    If A has index (i,j), tr(A) is a scalar (no index).
    Derivative: d tr(A)/dA = I (identity matrix).
    """
    def __init__(self, expr, index: TensorIndex):
        self.index = index
        self.expr = expr
        self.args = expr.args

    def __repr__(self):
        return f"$Tr_{self.index}$"

    def __hash__(self):
        return hash(tuple([*self.args, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TTrace):
            return False
        return self.args == other.args and self.__repr__() == other.__repr__()
```

### Step 2: Add to `obtain_TExpr()`

```python
def obtain_TExpr(expr, index):
    # ... existing cases ...
    elif isinstance(expr, Trace):
        return TTrace(expr, index)
    # ...
```

### Step 3: Add DAG building logic in `ToComGraph()`

```python
elif isinstance(Texpr, TTrace):
    # tr(A): child A has matrix index, output is scalar
    child_index = TensorIndex([0, 1])  # assign fresh 2D index
    queue.append((Texpr.args[0], child_index, Texpr))
```

### Step 4: Add differentiation rule in `TensorExpr.diff()`

```python
elif isinstance(node, TTrace):
    succ = list(self.ComGraph.successors(node))
    Aprime = derivatives[succ[0]][0]
    Aprime_idx = derivatives[succ[0]][1]
    # d tr(A)/dA_ij = delta_ij, so the result picks the diagonal of A'
    # ... implement the specific derivative rule ...
    derivatives[node] = (result_expr, result_idx)
```

### Step 5: Add TMul2Mul rules if needed

If the differentiation produces new index combinations not already handled
by `TMul2Mul`, add them:

```python
def TMul2Mul(x, y, index):
    # ... existing cases ...
    # (new_pattern) → corresponding matrix operation
    if new_condition:
        return corresponding_matrix_expression
```

### Step 6: Add tests

Always test:
1. The derivative result matches the expected symbolic expression
2. The result is mathematically correct (verify against manual derivation
   or [matrixcalculus.org](https://www.matrixcalculus.org/))

## TMul2Mul Index Reference

Complete list of currently handled index patterns:

| s1 (x) | s2 (y) | s3 (result) | Operation | Condition |
|---------|--------|-------------|-----------|-----------|
| scalar | any | any | `x * y` | s1 or s2 is scalar |
| `i` | `i` | `i` | `x * y` | element-wise |
| `i` | `j` (j!=i) | `ij` | `x @ transpose(y)` | outer product |
| `i` | `ii` | `ii` | `Diag(x * y)` | vector → diagonal |
| `i` | `ij` | `ij` | `Diag(x) @ y` | i == j[0] |
| `i` | `ji` | `ji` | `y @ Diag(x)` | i == j[1] |
| `ii` | `i` | `i` | `Diag(x).diag * y` | requires x is Diag |
| `ii` | `ii` | `ii` | `Diag(x * y)` | diagonal × diagonal |
| `ii` | `ij` | `ij` | `Diag(x) @ y` | i == j[0] |
| `ij` | `j` | `i` | `x @ y` | matrix-vector |
| `ij` | `i` | `j` | `transpose(x) @ y` | |
| `ij` | `ij` | `ij` | `x * y` | element-wise matrix |
| `ij` | `jj` | `ij` | `x @ Diag(y)` | only if y is Diag |
| `ij` | `jk` | `ik` | `x @ y` | matrix multiply |
| `ij` | `kj` | `ik` | `x @ transpose(y)` | |
| `ji` | `jk` | `ik` | `transpose(x) @ y` | |

When an unhandled pattern is encountered, `TMul2Mul` raises `NotImplementedError`
with the specific index values, which helps identify the new rule needed.
