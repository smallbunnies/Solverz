"""Analyzer extensions for ``_LoopJacSelectMat`` as a sparse-matrix
operand (#133 groundwork). These assert that ``_classify_matmul`` and
``_sparse_matrix_nnz`` accept ``Diag(var) @ SelectMat``,
``SelectMat @ Diag(var)``, and the ``Mat_Mul(SelectMat, Para)`` /
``Mat_Mul(Para, SelectMat)`` composites the LoopEqn translator emits
for indirect-outer / non-identity-map Pattern 4.
"""
import numpy as np
import sympy as sp

from Solverz.code_printer.python.module.mutable_mat_analyzer import (
    _classify_matmul, _sparse_matrix_nnz,
)
from Solverz.equation.eqn import _LoopJacSelectMat
from Solverz.sym_algebra.functions import Diag, Mat_Mul
from Solverz.sym_algebra.symbols import Para, iVar


def _mk_select(col_map, n_outer, n_diff):
    return _LoopJacSelectMat(
        sp.Tuple(*[sp.Integer(int(c)) for c in col_map]),
        sp.Integer(n_outer),
        sp.Integer(n_diff),
    )


def test_classify_matmul_diag_at_select_mat():
    sel = _mk_select([2, 0, 3], 3, 4)
    v = iVar('v')
    expr = Mat_Mul(Diag(v), sel)
    result = _classify_matmul(expr)
    assert result is not None
    kind, var_expr, matrix_expr, sign = result
    assert kind == 'row_scale'
    assert var_expr == v
    assert sign == 1


def test_classify_matmul_select_mat_at_diag():
    sel = _mk_select([1, 0], 2, 3)
    v = iVar('v')
    expr = Mat_Mul(sel, Diag(v))
    result = _classify_matmul(expr)
    assert result is not None
    kind, var_expr, matrix_expr, sign = result
    assert kind == 'col_scale'
    assert var_expr == v


def test_classify_matmul_diag_at_composite():
    sel = _mk_select([0, 1, 2], 3, 3)
    P = Para('P', dim=2)
    v = iVar('v')
    expr = Mat_Mul(Diag(v), Mat_Mul(sel, P))
    result = _classify_matmul(expr)
    assert result is not None
    kind, var_expr, matrix_expr, sign = result
    assert kind == 'row_scale'
    assert isinstance(matrix_expr, Mat_Mul)


def test_sparse_matrix_nnz_select_mat():
    sel = _mk_select([2, 0, 3], 3, 4)
    rows, cols, data = _sparse_matrix_nnz(sel, PARAM={})
    # The COO order for a (3,4) CSC -> tocoo is column-major:
    # column 0 has row 1 (from col_map[1]=0), column 2 has row 0 (col_map[0]=2),
    # column 3 has row 2 (col_map[2]=3). Sort-check by sets instead.
    assert set(zip(rows.tolist(), cols.tolist())) == {
        (0, 2), (1, 0), (2, 3),
    }
    assert np.all(data == 1.0)


def test_classify_matmul_biscale_nested_right():
    """``Diag(u) @ Mat_Mul(Matrix, Diag(v))`` → biscale."""
    from Solverz.code_printer.python.module.mutable_mat_analyzer import (
        _classify_matmul_biscale,
    )
    sel = _mk_select([2, 0, 3], 3, 4)
    u = iVar('u')
    v = iVar('v')
    expr = Mat_Mul(Diag(u), Mat_Mul(sel, Diag(v)))
    result = _classify_matmul_biscale(expr)
    assert result is not None
    u_expr, matrix_expr, v_expr, sign = result
    assert u_expr == u
    assert v_expr == v
    assert sign == 1


def test_classify_matmul_biscale_nested_left():
    """``Mat_Mul(Diag(u), Matrix) @ Diag(v)`` → biscale."""
    from Solverz.code_printer.python.module.mutable_mat_analyzer import (
        _classify_matmul_biscale,
    )
    P = Para('P', dim=2)
    u = iVar('u')
    v = iVar('v')
    expr = Mat_Mul(Mat_Mul(Diag(u), P), Diag(v))
    result = _classify_matmul_biscale(expr)
    assert result is not None


def test_sparse_matrix_nnz_select_at_para(tmp_path):
    from Solverz.equation.param import Param
    P = Para('P', dim=2)
    P_obj = Param('P', np.array([[10.0, 11.0, 12.0],
                                  [20.0, 21.0, 22.0],
                                  [30.0, 31.0, 32.0]]),
                  dim=2, sparse=True)
    PARAM = {'P': P_obj}
    sel = _mk_select([2, 0], 2, 3)  # pick rows 2 and 0 of P
    composite = Mat_Mul(sel, P)
    rows, cols, data = _sparse_matrix_nnz(composite, PARAM)
    # sel @ P = P[[2, 0], :] — shape (2, 3):
    #   row 0 = P[2, :] = [30, 31, 32]
    #   row 1 = P[0, :] = [10, 11, 12]
    expected = {
        (0, 0): 30.0, (0, 1): 31.0, (0, 2): 32.0,
        (1, 0): 10.0, (1, 1): 11.0, (1, 2): 12.0,
    }
    got = {(int(r), int(c)): float(d)
           for r, c, d in zip(rows, cols, data)}
    assert got == expected
