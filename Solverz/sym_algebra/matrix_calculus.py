"""
Matrix-Vector Calculus Module
=============================

Automatic symbolic differentiation of mixed matrix-vector expressions using
Einstein index notation, following the approach of:

    Laue, Mitterreiter, Giesen. "A Simple and Efficient Tensor Calculus."
    Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

See also: https://www.matrixcalculus.org/

Functionality
-------------
This module differentiates expressions that mix matrix and vector operations,
producing symbolic Jacobian expressions suitable for Newton-type solvers.

Supported operations:
    - Matrix-vector multiply: ``Mat_Mul(A, x)``
    - Element-wise multiply: ``x * y``
    - Addition: ``x + y``
    - Absolute value: ``Abs(x)``
    - Element-wise unary functions: ``exp(x)``, ``sin(x)``, ``cos(x)``, ``ln(x)``
    - Power: ``x**n``  (constant exponent)
    - Transpose: ``transpose(A)``
    - Diagonal: ``Diag(x)``

Mathematics
-----------
Expressions are parsed into a directed acyclic graph (DAG) where:
    - Leaf nodes are variables (vectors) and parameters (vectors or matrices)
    - Internal nodes are operations (+, *, @, f(), etc.)
    - The root node is the output expression

Each node carries a **TensorIndex** using Einstein notation:
    - Scalar: no index (empty)
    - Vector x: single index ``x_i``
    - Matrix A: two indices ``A_{ij}``

Differentiation proceeds in **forward mode** (pushforward), from leaves to root:
    1. At each leaf: compute ``dx/ds`` (identity or zero)
    2. At multiplication nodes: product rule with index tracking
       ``C' = B *_(s2,s1s4,s3s4) A' + A *_(s1,s2s4,s3s4) B'``
    3. At element-wise unary function nodes: ``f(A)' = f'(A) * A'``
    4. At addition nodes: ``(A+B)' = A' + B'``

The ``TMul2Mul`` function resolves tensor index combinations back into concrete
matrix operations (Mat_Mul, Diag, transpose, element-wise multiply).

How to Extend
-------------
To add a new **element-wise unary function** (e.g., ``tanh``):
    1. Define the function class in ``functions.py`` inheriting ``UniVarFunc``
    2. Implement ``fdiff()`` returning the element-wise derivative
    3. ``TUnaryFunc`` will automatically handle it — no changes needed here
    4. Add a test in ``test_matrix_calculus.py``

To add a new **node type** (e.g., matrix inverse):
    1. Create a new T-node class (see ``TTranspose``, ``TDiag`` as examples)
    2. Add a branch in ``obtain_TExpr()`` to recognize the SymPy expression
    3. Add a branch in ``TensorExpr.diff()`` for the differentiation rule
    4. If new index combinations arise, add them to ``TMul2Mul()``
    5. Add tests

TMul2Mul Index Reference
-------------------------
    (scalar, any, any)    → scalar multiply
    (i, i, i)             → element-wise multiply
    (i, j, ij) where i!=j → outer product: x @ transpose(y)
    (i, ii, ii)           → vector * Diag: Diag(x * y)
    (i, ij, ij) where i=j[0] → Diag(x) @ A
    (ii, i, i)            → Diag extract: Diag(x).diag * y
    (ii, ii, ii)          → element-wise Diag * Diag
    (ii, ij, ij)          → Diag(x) @ A
    (ij, j, i)            → matrix-vector: A @ y
    (ij, i, j)            → transpose(A) @ y equivalent
    (ij, ij, ij)          → element-wise matrix multiply
    (ij, jj, ij)          → A @ Diag(y)
    (ij, ji, ii)          → element-wise A * B^T on diagonal
    (ij, jk, ik)          → matrix multiply: A @ B
    (ji, ij, ii)          → transpose product giving diagonal
"""
from __future__ import annotations

from typing import List, Union, Dict, Tuple

import networkx as nx
import numpy as np
import sympy as sp
from sympy import Expr, Mul, Add, S, Number, Integer, Symbol, Pow

from Solverz.sym_algebra.symbols import IdxVar, IdxPara, iVar, Para, iAliasVar
from Solverz.sym_algebra.functions import (
    Mat_Mul, Diag, transpose, Abs, Sign, UniVarFunc
)
from Solverz.num_api.module_parser import modules


# ---------------------------------------------------------------------------
#  Tensor node classes
# ---------------------------------------------------------------------------

class TMatrix:
    def __init__(self, expr: Expr, index: TensorIndex):
        self.name = expr.name
        self.index = index
        self.symbol = expr

    def __repr__(self):
        return f"{self.name}[{self.index}]"

    def __hash__(self):
        return hash(tuple([self.name, *self.index.index]))

    def __eq__(self, other):
        if not isinstance(other, TMatrix):
            return False
        return self.name == other.name and self.index.index_value == other.index.index_value


class TVector:
    def __init__(self, expr: Expr, index: TensorIndex):
        self.name = expr.name
        self.index = index
        self.symbol = expr

    def __repr__(self):
        return f"{self.name}[{self.index}]"

    def __hash__(self):
        return hash(tuple([self.name, *self.index.index]))

    def __eq__(self, other):
        if not isinstance(other, TVector):
            return False
        return self.name == other.name and self.index.index_value == other.index.index_value


class TMul:

    def __init__(self, expr: Union[Mul, Mat_Mul], index: TensorIndex):
        self.args = []
        self.expr = expr
        if not isinstance(expr, (Mul, Mat_Mul)):
            raise TypeError("Unsupported input expression")
        if len(expr.args) > 2:
            self.args += [expr.args[0]]
            self.args += [expr.func(*expr.args[1:])]
        else:
            self.args += [expr.args[0]]
            self.args += [expr.args[1]]
        dim1, dim2 = [obtain_dim(arg) for arg in self.args]
        if dim1 == 0:
            self.index = [TensorIndex(-1), index, index]
        else:
            if len(index.index) == 2:
                # TMul produces a matrix
                if isinstance(expr, Mul):
                    # dot product of two matrices
                    self.index: List[TensorIndex, TensorIndex, TensorIndex] = [index, index, index]
                elif isinstance(expr, Mat_Mul):
                    if isinstance(expr.args[0], Diag):
                        # diag(x)@A: x is vector with first index
                        self.index: List[TensorIndex, TensorIndex, TensorIndex] = \
                            [index, TensorIndex([index.index_value[0]]), index]
                    elif isinstance(expr.args[1], Diag):
                        # A@diag(x): x is vector with second index
                        self.index: List[TensorIndex, TensorIndex, TensorIndex] = \
                            [index, TensorIndex([index.index_value[1]]), index]
                    else:
                        # A@B
                        max_index = max(index.index_value)
                        self.index: List[TensorIndex, TensorIndex, TensorIndex] = \
                            [TensorIndex([index.index_value[0], max_index + 1]),
                             TensorIndex([max_index + 1, index.index_value[0]]),
                             index]
            elif len(index.index) == 1:
                if dim1 == 2:
                    if (isinstance(expr, Mat_Mul)
                            and isinstance(self.args[0], Diag)):
                        # Diag(x)@v = x*v (element-wise) for vector output
                        self.args = [self.args[0].args[0], self.args[1]]
                        self.index: List[TensorIndex, TensorIndex, TensorIndex] = \
                            [index, index, index]
                    else:
                        # A@y
                        max_index = max(index.index_value)
                        self.index: List[TensorIndex, TensorIndex, TensorIndex] = \
                            [TensorIndex([index.index_value[0], max_index + 1]),
                             TensorIndex([max_index + 1]),
                             index]
                elif dim1 == 1:
                    # y*x
                    self.index: List[TensorIndex, TensorIndex, TensorIndex] = \
                        [index, index, index]
            else:
                raise TypeError(f"Unsupported index length {len(index.index)}")

    def __repr__(self):
        return r"$*%s_{%s}$" % (self.index, self.args)

    def __hash__(self):
        return hash(tuple([*self.args, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TMul):
            return False
        return self.args == other.args and self.__repr__() == other.__repr__()


class TAdd:

    def __init__(self, expr: Add, index: TensorIndex):
        if not isinstance(expr, Add):
            raise TypeError("Input expression is not Add")

        self.index = index
        self.expr = expr
        self.args = expr.args

    def __repr__(self):
        return f"+[{self.index}]"

    def __hash__(self):
        return hash(tuple([*self.args, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TAdd):
            return False
        return self.args == other.args and self.__repr__() == other.__repr__()


class TAbs:
    """Element-wise absolute value node. Kept for backward compatibility;
    new code should prefer the general ``TUnaryFunc``."""

    def __init__(self, expr: Abs, index: TensorIndex):
        if not isinstance(expr, Abs):
            raise TypeError("Input expression is not Abs")

        self.index: TensorIndex = index
        self.expr = expr
        self.args = expr.args

    def __repr__(self):
        return f"$Abs_{self.index}$"

    def __hash__(self):
        return hash(tuple([*self.args, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TAbs):
            return False
        return self.args == other.args and self.__repr__() == other.__repr__()


class TUnaryFunc:
    """General element-wise unary function node.

    Handles any ``UniVarFunc`` subclass (exp, sin, cos, ln, Sign, heaviside, etc.)
    whose ``fdiff()`` is defined.

    Differentiation rule:  if C = f(A), then C' = f'(A) * A'  (element-wise).
    """

    def __init__(self, expr: UniVarFunc, index: TensorIndex):
        self.index: TensorIndex = index
        self.expr = expr
        self.func = type(expr)
        self.args = expr.args

    def __repr__(self):
        return f"${self.func.__name__}_{self.index}$"

    def __hash__(self):
        return hash(tuple([*self.args, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TUnaryFunc):
            return False
        return self.args == other.args and self.__repr__() == other.__repr__()


class TPow:
    """Element-wise power node for ``Pow(base, n)`` where n is a constant.

    Differentiation rule:  if C = A^n, then C' = n * A^(n-1) * A'  (element-wise).
    """

    def __init__(self, expr: Pow, index: TensorIndex):
        self.index: TensorIndex = index
        self.expr = expr
        self.base = expr.args[0]
        self.exp = expr.args[1]
        self.args = (self.base,)  # only the base is a child in the DAG

    def __repr__(self):
        return f"$Pow_{self.index}$"

    def __hash__(self):
        return hash(tuple([self.base, self.exp, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TPow):
            return False
        return self.base == other.base and self.exp == other.exp and self.__repr__() == other.__repr__()


class TTranspose:
    """Transpose node for ``transpose(A)``.

    If A has index ``(i,j)``, then ``transpose(A)`` has index ``(j,i)``.
    Differentiation rule: ``transpose(A)' = transpose(A')``, i.e. swap the
    first two indices of the derivative.
    """

    def __init__(self, expr: transpose, index: TensorIndex):
        self.index: TensorIndex = index
        self.expr = expr
        self.args = expr.args

    def __repr__(self):
        return f"$T_{self.index}$"

    def __hash__(self):
        return hash(tuple([*self.args, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TTranspose):
            return False
        return self.args == other.args and self.__repr__() == other.__repr__()


class TDiag:
    """Diagonal matrix node for ``Diag(x)`` appearing in user expressions.

    If x has index ``(i)``, then ``Diag(x)`` has index ``(i,i)`` (diagonal indices).
    Differentiation rule: ``Diag(x)' = Diag(x')``.
    """

    def __init__(self, expr: Diag, index: TensorIndex):
        self.index: TensorIndex = index
        self.expr = expr
        self.args = expr.args

    def __repr__(self):
        return f"$Diag_{self.index}$"

    def __hash__(self):
        return hash(tuple([*self.args, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TDiag):
            return False
        return self.args == other.args and self.__repr__() == other.__repr__()


class TNumber:

    def __init__(self, expr: Number):
        if not isinstance(expr, Number):
            raise TypeError("Input expression is not Number")
        self.name = expr.__repr__()
        self.index: TensorIndex = TensorIndex(-1)
        self.expr = expr
        self.args = expr.args

    def __repr__(self):
        return self.expr.__repr__()


# ---------------------------------------------------------------------------
#  TensorIndex
# ---------------------------------------------------------------------------

index_list = [chr(ord('i') + i) for i in range(18)]


class TensorIndex:

    def __init__(self, index_value):
        if isinstance(index_value, (int, float)):
            index_value = [index_value]
        self.index_value = index_value

    @property
    def index(self) -> str:
        return ''.join([index_list[idx] if idx >= 0 else '' for idx in self.index_value])

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError(f"Unsupported index type {type(item)}")
        else:
            return self.index[item]

    def __repr__(self):
        return self.index

    def __eq__(self, other):
        if not isinstance(other, TensorIndex):
            raise TypeError(f"Unsupported type {type(other)}")

        if self.index_value == other.index_value:
            return True
        else:
            return False


# ---------------------------------------------------------------------------
#  Helper: determine dimension of expression
# ---------------------------------------------------------------------------

def obtain_dim(expr) -> int:
    """Evaluate *expr* to decide its dimension (0=scalar, 1=vector, 2=matrix)."""
    if expr == S.NegativeOne or isinstance(expr, Number):
        return 0

    # Pow with constant exponent inherits the dimension of its base
    if isinstance(expr, Pow):
        return obtain_dim(expr.args[0])

    # UniVarFunc (element-wise) preserves the dimension of its argument
    if isinstance(expr, UniVarFunc):
        return obtain_dim(expr.args[0])

    # transpose swaps nothing dimension-wise (matrix stays matrix)
    if isinstance(expr, transpose):
        return obtain_dim(expr.args[0])

    # Diag promotes vector (dim=1) to matrix (dim=2)
    if isinstance(expr, Diag):
        return 2

    symbol_dict = dict()
    for symbol in list(expr.free_symbols):
        if isinstance(symbol, (iVar, IdxVar, iAliasVar)):
            symbol_dict[symbol] = np.ones((2, 1))
        elif isinstance(symbol, (Para, IdxPara)):
            if symbol.dim == 2:
                symbol_dict[symbol] = np.ones((2, 2))
            elif symbol.dim == 1:
                symbol_dict[symbol] = np.ones((2, 1))

    if not symbol_dict:
        return 0

    temp_expr = sp.lambdify(symbol_dict.keys(), expr, modules=modules)
    result = temp_expr(*symbol_dict.values())
    if np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0):
        return 0
    return result.shape[1]


# ---------------------------------------------------------------------------
#  Build computation graph
# ---------------------------------------------------------------------------

def obtain_TExpr(expr: Expr, index: TensorIndex):
    """Convert a SymPy Expr to a tensor node (TMatrix, TVector, TMul, etc.)."""

    if isinstance(expr, Add):
        return TAdd(expr, index)
    elif isinstance(expr, (Mat_Mul, Mul)):
        return TMul(expr, index)
    elif isinstance(expr, Abs):
        return TAbs(expr, index)
    elif isinstance(expr, UniVarFunc):
        return TUnaryFunc(expr, index)
    elif isinstance(expr, Pow):
        if isinstance(expr.args[1], Number):
            return TPow(expr, index)
        else:
            raise NotImplementedError(f"Pow with non-constant exponent {expr.args[1]} not supported")
    elif isinstance(expr, transpose):
        return TTranspose(expr, index)
    elif isinstance(expr, Diag):
        return TDiag(expr, index)
    elif isinstance(expr, (iVar, IdxVar, iAliasVar)):
        return TVector(expr, index)
    elif isinstance(expr, (Para, IdxPara)):
        if expr.dim == 2:
            return TMatrix(expr, index)
        elif expr.dim == 1:
            return TVector(expr, index)
        else:
            raise TypeError(f"Unsupported dim {expr.dim}")
    elif isinstance(expr, Number):
        return TNumber(expr)
    else:
        raise NotImplementedError(f"Unsupported expression type {type(expr)}: {expr}")


def ToComGraph(expr: Expr) -> nx.DiGraph:
    """Build a directed acyclic computation graph from a SymPy expression."""
    DG = nx.DiGraph()
    output_dim = obtain_dim(expr)
    if output_dim == 2:
        index = TensorIndex([0, 1])
    elif output_dim == 1:
        index = TensorIndex([0])
    elif output_dim == 0:
        index = TensorIndex([-1])
    else:
        raise TypeError(f"Unsupported output dim {output_dim}")
    queue = [(expr, index, None)]  # [(Expr, TensorIndex, predecessor)]

    while queue:
        current_expr, index, predecessor = queue.pop(0)

        Texpr = obtain_TExpr(current_expr, index)

        if predecessor is not None:
            DG.add_edge(predecessor, Texpr)
            DG.nodes[Texpr]['level'] = DG.nodes[predecessor]['level'] + 1
        else:
            DG.add_node(Texpr)
            DG.nodes[Texpr]['level'] = 0

        if isinstance(Texpr, TMul):
            queue.append((Texpr.args[0], TensorIndex(Texpr.index[0].index_value), Texpr))
            queue.append((Texpr.args[1], TensorIndex(Texpr.index[1].index_value), Texpr))
        elif isinstance(Texpr, TAdd):
            for subexpr in Texpr.args:
                queue.append((subexpr, TensorIndex(Texpr.index.index_value), Texpr))
        elif isinstance(Texpr, TAbs):
            queue.append((Texpr.args[0], TensorIndex(Texpr.index.index_value), Texpr))
        elif isinstance(Texpr, TUnaryFunc):
            queue.append((Texpr.args[0], TensorIndex(Texpr.index.index_value), Texpr))
        elif isinstance(Texpr, TPow):
            # only the base is a child
            queue.append((Texpr.base, TensorIndex(Texpr.index.index_value), Texpr))
        elif isinstance(Texpr, TTranspose):
            # child has swapped indices
            if len(index.index) == 2:
                child_index = TensorIndex([index.index_value[1], index.index_value[0]])
            else:
                child_index = TensorIndex(index.index_value)
            queue.append((Texpr.args[0], child_index, Texpr))
        elif isinstance(Texpr, TDiag):
            # Diag(x): x has single index = first of the diagonal pair
            if len(index.index) == 2 and index.index_value[0] == index.index_value[1]:
                child_index = TensorIndex([index.index_value[0]])
            else:
                child_index = TensorIndex([index.index_value[0]])
            queue.append((Texpr.args[0], child_index, Texpr))

    if not nx.is_directed_acyclic_graph(DG):
        raise ValueError("The expression contains cycles.")

    return DG


def draw_dag_as_tree(G, pos=None):
    """Visualize the computation DAG using Graphviz layout."""
    if pos is None:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    root_node = [node for node in G.nodes() if G.in_degree(node) == 0]
    leaf_node = [node for node in G.nodes() if G.out_degree(node) == 0]
    nx.draw_networkx_nodes(G, pos, nodelist=root_node, node_size=800, node_color='lightgreen')
    nx.draw_networkx_nodes(G, pos, nodelist=leaf_node, node_size=800)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15)

    import matplotlib.pyplot as plt
    plt.axis('off')
    plt.show()


# ---------------------------------------------------------------------------
#  TMul2Mul: resolve tensor index multiplication to matrix operations
# ---------------------------------------------------------------------------

def TMul2Mul(x, y, index: List[TensorIndex, TensorIndex, TensorIndex]):
    """Convert a tensor index multiplication back to concrete matrix operations.

    Parameters
    ----------
    x, y : symbolic expressions (the two operands)
    index : list of three TensorIndex [s1, s2, s3]
        s1 = index of x, s2 = index of y, s3 = index of the result

    Returns
    -------
    A SymPy expression using Mat_Mul, Diag, transpose, etc.
    """
    if x is S.Zero or y is S.Zero:
        return 0
    s1, s2, s3 = index[:]
    if s1 == TensorIndex(-1) or s2 == TensorIndex(-1):
        return x * y

    len1 = len(s1.index)
    len2 = len(s2.index)

    # --- s1 is vector (single index) ---
    if len1 < 2:
        if len2 < 2:
            # (i, j, ij) where i != j → outer product
            if s1.index_value[0] != s2.index_value[0]:
                return Mat_Mul(x, transpose(y))
            # (i, i, i) → element-wise
            if s1.index_value[0] == s2.index_value[0]:
                return x * y
        else:
            # s2 is matrix
            # (i, ii, ii) where all same → Diag(x * y_diag)
            if s2.index_value[0] == s2.index_value[1] == s1.index_value[0]:
                if isinstance(y, Diag):
                    return Diag(x * y.args[0])
                else:
                    return Diag(x * y)
            # (i, ij, ij) where i == j[0], i != j[1] → Diag(x) @ y
            if (s2.index_value[0] != s2.index_value[1]
                    and s1.index_value[0] == s2.index_value[0]):
                return Mat_Mul(Diag(x), y)
            # (i, ji, ji) where i == j[1], i != j[0] → y @ Diag(x)
            if (s2.index_value[0] != s2.index_value[1]
                    and s1.index_value[0] == s2.index_value[1]):
                return Mat_Mul(y, Diag(x))

    # --- s1 is matrix (two indices) ---
    else:
        if len2 < 2:
            # s2 is vector
            # (ii, i, i) where all same → Diag extract * vector
            if (s1.index_value[0] == s1.index_value[1] == s2.index_value[0]):
                if isinstance(x, Diag):
                    return Diag(x.args[0] * y)
                # If x is not Diag, fall through to other cases
            # (ij, j, i) where i != j → matrix-vector: A @ y
            if (s1.index_value[0] != s1.index_value[1]
                    and s1.index_value[1] == s2.index_value[0]):
                return Mat_Mul(x, y)
            # (ij, i, j) where i != j, result is j → transpose(A) @ y
            if (s1.index_value[0] != s1.index_value[1]
                    and s1.index_value[0] == s2.index_value[0]):
                return Mat_Mul(transpose(x), y)
        else:
            # Both matrix
            # (ij, ij, ij) → element-wise
            if s1.index_value == s2.index_value and s1.index_value[0] != s1.index_value[1]:
                return x * y
            # (ii, ii, ii) → element-wise diagonal
            if (s1.index_value == s2.index_value
                    and s1.index_value[0] == s1.index_value[1]):
                if isinstance(x, Diag) and isinstance(y, Diag):
                    return Diag(x.args[0] * y.args[0])
                elif isinstance(x, Diag):
                    return Diag(x.args[0] * y)
                elif isinstance(y, Diag):
                    return Diag(x * y.args[0])
                else:
                    return Diag(x * y)
            # (ii, ij, ij) where i == j[0] → Diag(x) @ y
            if (s1.index_value[0] == s1.index_value[1]
                    and s2.index_value[0] != s2.index_value[1]
                    and s1.index_value[0] == s2.index_value[0]):
                if isinstance(x, Diag):
                    return Mat_Mul(x, y)
                else:
                    return Mat_Mul(Diag(x), y)
            # (ij, jj, ij) → A @ Diag(y)
            if (s1 == s3
                    and s2.index_value[0] == s2.index_value[1]
                    and s1.index_value[1] == s2.index_value[0]):
                if isinstance(y, Number):
                    return y * x
                elif isinstance(y, Diag):
                    return Mat_Mul(x, y)
                # If y is not Number or Diag, fall through to (ij, jk, ik)
            # (ij, jk, ik) → matrix multiply: A @ B
            if (s1.index_value != s2.index_value
                    and s1.index_value[1] == s2.index_value[0]):
                return Mat_Mul(x, y)
            # (ij, kj, ik) → A @ transpose(B)
            if (s1.index_value != s2.index_value
                    and s1.index_value[1] == s2.index_value[1]
                    and s1.index_value[0] == s3.index_value[0]):
                return Mat_Mul(x, transpose(y))
            # (ji, jk, ik) → transpose(A) @ B
            if (s1.index_value != s2.index_value
                    and s1.index_value[0] == s2.index_value[0]
                    and s1.index_value[1] == s3.index_value[0]
                    and s2.index_value[1] == s3.index_value[1]):
                return Mat_Mul(transpose(x), y)

    raise NotImplementedError(
        f"Unsupported TMul2Mul index combination: "
        f"s1={s1.index}({s1.index_value}), "
        f"s2={s2.index}({s2.index_value}), "
        f"s3={s3.index}({s3.index_value}), "
        f"x={x}, y={y}"
    )


# ---------------------------------------------------------------------------
#  TensorExpr: computation graph + differentiation
# ---------------------------------------------------------------------------

class TensorExpr:

    def __init__(self, expr: Expr):
        self.ComGraph = ToComGraph(expr)
        self.index = TensorIndex([0])

    def visualize(self):
        """Visualize the computation DAG."""
        draw_dag_as_tree(self.ComGraph)

    def diff(self, s):
        """Differentiate the expression w.r.t. symbol *s* using forward mode.

        Returns a SymPy expression representing the derivative.
        """
        leaf_node = [node for node in self.ComGraph.nodes() if self.ComGraph.out_degree(node) == 0]
        index_s = None
        for node in leaf_node:
            if hasattr(node, 'name') and node.name == s.name:
                index_s = node.index
        if index_s is None:
            raise TypeError(f"Cannot find index of {s}")

        highest_level = np.max([attr['level'] for node, attr in self.ComGraph.nodes(data=True)])
        derivatives: Dict = dict()
        for level in range(highest_level, -1, -1):
            current_nodes = [node for node, attr in self.ComGraph.nodes(data=True) if attr['level'] == level]
            for node in current_nodes:
                if isinstance(node, (TMatrix, TVector)):
                    derivatives[node] = (node.symbol.diff(s), node.index)

                elif isinstance(node, TAdd):
                    succ = list(self.ComGraph.successors(node))
                    derivatives[node] = (Add(*[derivatives[arg][0] for arg in succ]),
                                         derivatives[succ[0]][1])

                elif isinstance(node, (TAbs, TUnaryFunc, TPow)):
                    # Element-wise function: f(A)' = f'(A) * A'
                    succ = list(self.ComGraph.successors(node))
                    Aprime = derivatives[succ[0]][0]
                    Aprime_idx = derivatives[succ[0]][1]

                    # Compute f'(A)
                    if isinstance(node, TAbs):
                        fprimeA = Sign(node.args[0])
                    elif isinstance(node, TUnaryFunc):
                        fprimeA = node.expr.fdiff(1)
                    else:  # TPow
                        n = node.exp
                        fprimeA = n * node.base ** (n - 1)

                    s_f = node.index  # function output index
                    result_idx = TensorIndex(s_f.index_value + index_s.index_value)

                    # For leaf children, derivative index == node index (compact form).
                    # Use old concatenation approach to produce Diag wrapping.
                    # For intermediate children, derivative index is already the full
                    # combined index (includes the variable direction).
                    if len(Aprime_idx.index_value) <= len(s_f.index_value):
                        # Leaf case: extend index with variable direction
                        derivatives[node] = (
                            TMul2Mul(fprimeA, Aprime, [s_f, result_idx, result_idx]),
                            result_idx
                        )
                    else:
                        # Intermediate case: Aprime already has full derivative index
                        derivatives[node] = (
                            TMul2Mul(fprimeA, Aprime, [s_f, Aprime_idx, result_idx]),
                            result_idx
                        )

                elif isinstance(node, TTranspose):
                    # transpose(A)' w.r.t. x:
                    # If A' has indices (j,i,...), then transpose(A)' has (i,j,...)
                    succ = list(self.ComGraph.successors(node))
                    Aprime = derivatives[succ[0]][0]
                    Aprime_idx = derivatives[succ[0]][1]
                    if Aprime is S.Zero or Aprime == Integer(0):
                        derivatives[node] = (Integer(0), node.index)
                    else:
                        # The child had swapped indices; the derivative inherits the node's index
                        # For transpose, derivative is simply transpose(A')
                        if len(Aprime_idx.index_value) >= 2:
                            new_idx = TensorIndex([Aprime_idx.index_value[1], Aprime_idx.index_value[0]]
                                                  + Aprime_idx.index_value[2:])
                            derivatives[node] = (transpose(Aprime), new_idx)
                        else:
                            derivatives[node] = (Aprime, Aprime_idx)

                elif isinstance(node, TDiag):
                    # Diag(x)' = Diag(x')
                    succ = list(self.ComGraph.successors(node))
                    Aprime = derivatives[succ[0]][0]
                    Aprime_idx = derivatives[succ[0]][1]
                    if Aprime is S.Zero or Aprime == Integer(0):
                        derivatives[node] = (Integer(0), node.index)
                    else:
                        # x' has index (i, s4...), Diag(x') has index (ii, s4...)
                        new_idx = TensorIndex([Aprime_idx.index_value[0], Aprime_idx.index_value[0]]
                                              + Aprime_idx.index_value[1:])
                        derivatives[node] = (Diag(Aprime), new_idx)

                elif isinstance(node, TMul):
                    succ = list(self.ComGraph.successors(node))
                    s4 = index_s
                    s1 = node.index[0]
                    s2 = node.index[1]
                    s3 = node.index[2]
                    A = node.args[0]
                    B = node.args[1]
                    Aprime = derivatives[succ[0]][0]
                    Bprime = derivatives[succ[1]][0]
                    s1s4 = TensorIndex(s1.index_value + s4.index_value)
                    s3s4 = TensorIndex(s3.index_value + s4.index_value)
                    s2s4 = TensorIndex(s2.index_value + s4.index_value)
                    derivatives[node] = (
                        TMul2Mul(B, Aprime, [s2, s1s4, s3s4])
                        + TMul2Mul(A, Bprime, [s1, s2s4, s3s4]),
                        s3s4
                    )

                elif isinstance(node, TNumber):
                    derivatives[node] = (Integer(0), TensorIndex(-1))

        root_node = [node for node in self.ComGraph.nodes() if self.ComGraph.in_degree(node) == 0]
        return derivatives[root_node[0]][0]


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def MixedEquationDiff(expr: Expr, symbol: Symbol):
    """Calculate the derivative of a mixed matrix-vector expression.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to differentiate (may contain Mat_Mul, Diag, etc.)
    symbol : sympy.Symbol
        The variable to differentiate with respect to

    Returns
    -------
    sympy.Expr
        The symbolic derivative expression
    """
    return TensorExpr(expr).diff(symbol)
