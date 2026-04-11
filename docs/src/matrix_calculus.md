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

### 1. Einstein Index Notation

Every tensor is labelled with indices:

- **Scalar**: no index (e.g., $\alpha$)
- **Vector** $x$: single index $x_i$
- **Matrix** $A$: two indices $A_{ij}$

Repeated indices in a product imply summation (contraction). For example,
$A_{ij} x_j$ means $\sum_j A_{ij} x_j = (Ax)_i$.

### 2. Computation DAG

The expression is parsed into a directed acyclic graph (DAG):

- **Leaf nodes**: variables ($x$, $y$) and parameters ($A$, $B$, $G$)
- **Internal nodes**: operations ($+$, $\times$, $@$, $f(\cdot)$)
- **Root node**: the output expression

For example, $e \odot (Ge - Bf)$ becomes:

```
        *[i]          ← element-wise multiply
       /    \
     e[i]   -[i]      ← subtraction
           /    \
       @[i]    @[i]    ← matrix-vector products
       / \     / \
     G[ij] e[j] B[ij] f[j]
```

### 3. Forward-Mode Automatic Differentiation

Derivatives propagate from **leaves to root** (pushforward):

1. **Leaf variables**: $\frac{\partial x_j}{\partial x_j} = 1$ (identity), $\frac{\partial x_j}{\partial y_k} = 0$

2. **Multiplication** (product rule in index notation):

```{math}
\tilde{C}_{s_3 s_4} = B_{s_2} \cdot \tilde{A}_{s_1 s_4} + A_{s_1} \cdot \tilde{B}_{s_2 s_4}
```

   where $s_1, s_2, s_3$ are the operand/result indices and $s_4$ is the derivative direction.

3. **Element-wise unary function** $f$:

```{math}
\frac{\partial f(A)}{\partial x} = f'(A) \odot \frac{\partial A}{\partial x}
```

4. **Addition**: $\frac{\partial (A + B)}{\partial x} = \frac{\partial A}{\partial x} + \frac{\partial B}{\partial x}$

### 4. TMul2Mul: Index Resolution

After differentiation, tensor index expressions are resolved back to matrix operations.
The key rules are:

| Index Pattern | Matrix Operation | Meaning |
|---------------|-----------------|---------|
| $(i, i, i)$ | $x \odot y$ | element-wise |
| $(i, j, ij)$ | $x \cdot y^T$ | outer product |
| $(ij, j, i)$ | $A y$ | matrix-vector product |
| $(ij, jk, ik)$ | $AB$ | matrix multiplication |
| $(i, ij, ij)$ | $\operatorname{diag}(x) A$ | diagonal scaling |
| $(ij, jj, ij)$ | $A \operatorname{diag}(y)$ | right diagonal scaling |
| $(i, ii, ii)$ | $\operatorname{diag}(x \odot y)$ | diagonal from element-wise |

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

## API Reference

```{eval-rst}
.. autofunction:: Solverz.sym_algebra.matrix_calculus.MixedEquationDiff
.. autoclass:: Solverz.sym_algebra.matrix_calculus.TensorExpr
   :members: diff, visualize
```
