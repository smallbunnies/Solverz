from __future__ import annotations

from typing import List, Union, Dict, Tuple

import networkx as nx
import numpy as np
import sympy as sp
from sympy import Expr, Mul, Add, S, Number, Integer, Symbol

from Solverz.sym_algebra.symbols import IdxVar, IdxPara, iVar, Para
from Solverz.sym_algebra.functions import Mat_Mul, Diag, transpose, Abs, Sign
from Solverz.num_api.module_parser import modules


class TMatrix:
    def __init__(self, expr: Expr, index: TensorIndex):
        self.name = expr.name
        self.index = index
        self.symbol = expr

    def __repr__(self):
        return f"{self.name}[{self.index}]"

    # __hash__ and __eq__ functions ensure that TMatrix and TVector objects are unique in computation graph.
    def __hash__(self):
        return hash(tuple([self.name, *self.index.index]))

    def __eq__(self, other):
        if not isinstance(other, TMatrix):
            return False
        else:
            if self.name == other.name and self.index.index_value == other.index.index_value:
                return True
            else:
                return False


class TVector:
    def __init__(self, expr: Expr, index: TensorIndex):
        self.name = expr.name
        self.index = index
        self.symbol = expr

    def __repr__(self):
        return f"{self.name}[{self.index}]"

    # __hash__ and __eq__ functions ensure that TMatrix and TVector objects are unique in computation graph.
    def __hash__(self):
        return hash(tuple([self.name, *self.index.index]))

    def __eq__(self, other):
        if not isinstance(other, TVector):
            return False
        else:
            if self.name == other.name and self.index.index_value == other.index.index_value:
                return True
            else:
                return False


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
                        # diag(x)@A
                        self.index: List[TensorIndex, TensorIndex, TensorIndex] = [index, index[0], index]
                    elif isinstance(expr.args[1], Diag):
                        # A@diag(x)
                        self.index: List[TensorIndex, TensorIndex, TensorIndex] = [index, index[1], index]
                    else:
                        # A@B
                        max_index = max(index.index_value)
                        self.index: List[TensorIndex, TensorIndex, TensorIndex] = \
                            [TensorIndex([index.index_value[0], max_index + 1]),
                             TensorIndex([max_index + 1, index.index_value[0]]),
                             index]
            elif len(index.index) == 1:
                if dim1 == 2:
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
        # To add self.args here is to differentiate TMul objects with the same index, in which case, graphviz
        # treats the nodes with the same __repr__() value as one node.
        return r"$*%s_{%s}$" % (self.index, self.args)

    # __hash__ and __eq__ functions ensure that TMatrix and TVector objects are unique in computation graph.
    def __hash__(self):
        return hash(tuple([*self.args, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TMul):
            return False
        else:
            if self.args == other.args and self.__repr__() == other.__repr__():
                return True
            else:
                return False


class TAdd:

    def __init__(self, expr: Add, index: TensorIndex):
        if not isinstance(expr, Add):
            raise TypeError("Input expression is not Add")

        self.index = index
        self.expr = expr
        self.args = expr.args

    def _eval_derivative(self, s):
        pass

    def __repr__(self):
        return f"+[{self.index}]"

    def __hash__(self):
        return hash(tuple([*self.args, self.__repr__()]))

    def __eq__(self, other):
        if not isinstance(other, TAdd):
            return False
        else:
            if self.args == other.args and self.__repr__() == other.__repr__():
                return True
            else:
                return False


class TAbs:

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
        else:
            if self.args == other.args and self.__repr__() == other.__repr__():
                return True
            else:
                return False


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


def obtain_dim(expr) -> int:
    """
    # Evaluate the expr to decide its dimension
    """
    if expr == S.NegativeOne or isinstance(expr, Number):
        return 0

    symbol_dict = dict()
    for symbol in list(expr.free_symbols):
        if isinstance(symbol, (iVar, IdxVar)):
            symbol_dict[symbol] = np.ones((2, 1))
        elif isinstance(symbol, (Para, IdxPara)):
            if symbol.dim == 2:
                symbol_dict[symbol] = np.ones((2, 2))
            elif symbol.dim == 1:
                symbol_dict[symbol] = np.ones((2, 1))
    temp_expr = sp.lambdify(symbol_dict.keys(), expr, modules=modules)
    return temp_expr(*symbol_dict.values()).shape[1]


def obtain_TExpr(expr: Expr, index: TensorIndex):
    """
    Convert Expr to TOperator and its args

    """

    if isinstance(expr, Add):
        Texpr = TAdd(expr, index)
    elif isinstance(expr, (Mat_Mul, Mul)):
        Texpr = TMul(expr, index)
    elif isinstance(expr, Abs):
        Texpr = TAbs(expr, index)
    elif isinstance(expr, (iVar, IdxVar)):
        Texpr = TVector(expr, index)
    elif isinstance(expr, (Para, IdxPara)):
        if expr.dim == 2:
            Texpr = TMatrix(expr, index)
        elif expr.dim == 1:
            Texpr = TVector(expr, index)
        else:
            TypeError(f"Unsupported dim {expr.dim}")
    elif isinstance(expr, Number):
        Texpr = TNumber(expr)
    else:
        raise NotImplementedError(f"Unsupported expression type {type(expr)}")
    return Texpr


def ToComGraph(expr: Expr) -> nx.DiGraph:
    DG = nx.DiGraph()
    output_dim = obtain_dim(expr)
    if output_dim == 2:
        index = TensorIndex([0, 1])
    elif output_dim == 1:
        index = TensorIndex([0])
    else:
        raise TypeError("Unsupported output dim")
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

    # Check for and remove any cycles to ensure acyclic graph
    if not nx.is_directed_acyclic_graph(DG):
        raise ValueError("The expression contains cycles.")

    return DG


def draw_dag_as_tree(G, pos=None):
    # using Graphviz and pygraphviz to draw directed acyclic expression graph as a 'tree'

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


def TMul2Mul(x, y, index: List[TensorIndex, TensorIndex, TensorIndex]):
    if x is S.Zero or y is S.Zero:
        return 0
    s1, s2, s3 = index[:]
    if s1 == TensorIndex(-1) or s2 == TensorIndex(-1):
        return x * y
    if len(s1.index) < 2:
        if len(s2.index) < 2:
            # i, j, ij
            if s1.index_value[0] != s2.index_value[0]:
                return Mat_Mul(x, transpose(y))
            # i, i, i
            if s1.index_value[0] == s2.index_value[0]:
                return x * y
        else:
            # j, jj, jj
            if s2.index_value[0] == s2.index_value[1] == s1.index_value[0]:
                if isinstance(y, Diag):
                    return Diag(x * y.args[0])
                else:
                    return Diag(x * y)
            # i, ij, ij
            if s2.index_value[0] != s2.index_value[1] and s1.index_value[0] == s2.index_value[0]:
                return Mat_Mul(Diag(x), y)
    else:
        if len(s2.index) < 2:
            # jj, j, jj
            if s1.index_value[0] == s1.index_value[1] == s2.index_value[0]:
                if isinstance(x, Diag):
                    return Diag(x.args[0] * y)
            # ij, j, i
            if s1.index_value[0] != s1.index_value[1] and s1.index_value[1] == s2.index_value[0]:
                return Mat_Mul(x, y)
        else:
            # ij, ij, ij
            if s1.index_value == s2.index_value and s1.index_value[0] != s1.index_value[1]:
                return x * y
            # ij, jj, ij
            if s1 == s3 and s2.index_value[0] == s2.index_value[1] and s1.index_value[1] == s2.index_value[0]:
                if isinstance(y, Number):
                    return y * x
                elif isinstance(y, Diag):
                    return Mat_Mul(x, y)
            # ij, jk, ik
            if s1.index_value != s2.index_value and s1.index_value[1] == s2.index_value[0]:
                return Mat_Mul(x, y)
        return NotImplementedError("Unsupport index type!")


class TensorExpr:

    def __init__(self, expr: Expr):
        self.ComGraph = ToComGraph(expr)
        self.index = TensorIndex([0])

    def visualize(self):
        """
        To visualize TensorExpr
        """
        draw_dag_as_tree(self.ComGraph)

    def diff(self, s):

        leaf_node = [node for node in self.ComGraph.nodes() if self.ComGraph.out_degree(node) == 0]
        index_s = None
        for node in leaf_node:
            if node.name == s.name:
                index_s = node.index
        if index_s is None:
            raise TypeError(f"Cant find index of {s}")

        highest_level = np.max([attr['level'] for node, attr in self.ComGraph.nodes(data=True)])
        derivatives: Dict[Union[TMatrix, TVector, TMul, TAdd, TAbs, TNumber], Tuple[Expr, TensorIndex]] = dict()
        for level in range(highest_level, -1, -1):
            current_nodes = [node for node, attr in self.ComGraph.nodes(data=True) if attr['level'] == level]
            for node in current_nodes:
                if isinstance(node, (TMatrix, TVector)):
                    derivatives[node] = (node.symbol.diff(s), node.index)
                elif isinstance(node, TAdd):
                    succ = list(self.ComGraph.successors(node))
                    derivatives[node] = (Add(*[derivatives[arg][0] for arg in succ]), derivatives[succ[0]][1])
                elif isinstance(node, TAbs):
                    succ = list(self.ComGraph.successors(node))
                    s1 = derivatives[succ[0]][1]
                    s2 = index_s
                    Aprime = derivatives[succ[0]][0]
                    fprimeA = Sign(node.args[0])
                    s1s2 = TensorIndex(s1.index_value + s2.index_value)

                    derivatives[node] = (TMul2Mul(fprimeA, Aprime, [s1, s1s2, s1s2]),
                                         s1s2)
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
                    derivatives[node] = (TMul2Mul(B, Aprime, [s2, s1s4, s3s4]) + TMul2Mul(A, Bprime, [s1, s2s4, s3s4]),
                                         s3s4)
                elif isinstance(node, TNumber):
                    derivatives[node] = (Integer(0), TensorIndex(-1))
        root_node = [node for node in self.ComGraph.nodes() if self.ComGraph.in_degree(node) == 0]
        return derivatives[root_node[0]][0]


def MixedEquationDiff(expr: Expr, symbol: Symbol):
    """
    Calculate the derivatives of mixed-Matrix-Vector equations

    """
    return TensorExpr(expr).diff(symbol)
