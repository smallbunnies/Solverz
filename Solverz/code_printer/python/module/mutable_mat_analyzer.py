"""Mutable matrix Jacobian block analyzer.

Decomposes a SymPy derivative expression for a mutable-matrix Jacobian
block (such as ``Diag(G@e - B@f + p_ref) + Diag(e)@G + Diag(f)@B`` from
power flow) into a sum of typed *terms*. For each term we precompute the
mapping from its source indices to positions in the output CSC data array,
so that the runtime can build the block data with pure numpy/Numba loops
instead of constructing intermediate scipy sparse matrices at every call.

Recognised term shapes (after flattening ``Mul`` coefficients into a
``sign`` ∈ {+1, −1}):

1. ``Diag(inner)``
   Contributes at positions ``(i, i)`` for each ``i`` where ``(i, i)`` is
   in the output sparsity. Runtime: ``data[out_pos] = sign * inner[src_i]``.

2. ``Mat_Mul(Diag(var), Matrix)``  (row-scaled sparse matrix)
   The ``Matrix`` must be a fixed ``Para`` (immutable after modelling).
   Each nnz ``(r, c)`` in ``Matrix`` lands at ``pos_lookup[(r, c)]`` with
   value ``sign * var[r] * Matrix.data[k]``. Runtime scatter-add over nnz.

3. ``Mat_Mul(Matrix, Diag(var))``  (column-scaled sparse matrix, rare)
   Symmetric to case 2 but scaled by ``var[c]`` instead of ``var[r]``.

Everything else falls back to scipy sparse evaluation with fancy indexing
(slow but correct).

The analyzer is pure data — it doesn't touch scipy.sparse at runtime.
That's the whole point: the expensive sparse-matrix construction happens
zero times per Newton step.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sympy import Add, Mul, S, Symbol, Expr, sympify, pycode

from Solverz.sym_algebra.functions import Diag, Mat_Mul
from Solverz.sym_algebra.symbols import Para, iVar, IdxVar


class MutableMatBlockMapping:
    """Precomputed runtime mapping for one mutable matrix Jacobian block.

    Stores the decomposition of a derivative expression into terms plus
    the index arrays each term will use at runtime. All attributes are
    plain numpy arrays / SymPy expressions — no scipy.sparse objects —
    so they pickle cleanly and compile under Numba.

    Attributes
    ----------
    n_out : int
        Number of entries in the block's contribution to ``_data_``.
    diag_terms : list of dict
        Each dict has ``sign`` (±1), ``inner_expr`` (SymPy), ``out_pos``
        (ndarray[int64]), ``src_idx`` (ndarray[int64]).
    row_scale_terms : list of dict
        Each dict has ``sign``, ``var_name`` (str, name of the vector
        variable), ``matrix_name`` (str, name of the sparse Para),
        ``out_pos`` (ndarray), ``src_row`` (ndarray), ``mat_data_ref``
        (ndarray — the matrix's constant .data values, baked in).
    col_scale_terms : list of dict
        Symmetric to row_scale but with ``src_col`` instead.
    fallback_expr : SymPy Expr or None
        Sum of any unrecognised terms; evaluated via sparse fancy indexing.
    fallback_out_row, fallback_out_col : ndarray or None
        COO positions used by the fallback expression at runtime.
    """

    __slots__ = ('n_out', 'diag_terms', 'row_scale_terms',
                 'col_scale_terms', 'fallback_expr',
                 'fallback_out_row', 'fallback_out_col', 'has_fallback')

    def __init__(self, n_out: int):
        self.n_out = n_out
        self.diag_terms: List[Dict] = []
        self.row_scale_terms: List[Dict] = []
        self.col_scale_terms: List[Dict] = []
        self.fallback_expr: Optional[Expr] = None
        self.fallback_out_row: Optional[np.ndarray] = None
        self.fallback_out_col: Optional[np.ndarray] = None
        self.has_fallback: bool = False


def _extract_sign_and_core(term: Expr) -> Tuple[int, Expr]:
    """Return ``(sign, core)`` where ``core`` has no leading ±1 coefficient.

    Handles ``-Diag(x)`` → ``(-1, Diag(x))`` and ``Mul(-1, Mat_Mul(...))`` →
    ``(-1, Mat_Mul(...))``. A non-unit numeric coefficient like 2*Diag(x)
    is not unwrapped — it gets absorbed into the diag term later.
    """
    if isinstance(term, Mul):
        args = list(term.args)
        sign = 1
        rest: List[Expr] = []
        for a in args:
            if a == S.NegativeOne:
                sign *= -1
            else:
                rest.append(a)
        if len(rest) == 1:
            return sign, rest[0]
        return sign, Mul(*rest) if rest else S.One
    return 1, term


def _unwrap_negated_para(expr: Expr) -> Tuple[int, Optional[Para]]:
    """If ``expr`` is ``Para`` or ``-Para``, return ``(sign, Para)``.

    Handles the common case of ``diag(v) @ (-B)`` where SymPy represents
    ``-B`` as ``Mul(-1, B)``. For anything else, returns ``(1, None)``.
    """
    if isinstance(expr, Para):
        return 1, expr
    if isinstance(expr, Mul):
        sign = 1
        rest: List[Expr] = []
        for a in expr.args:
            if a == S.NegativeOne:
                sign *= -1
            else:
                rest.append(a)
        if len(rest) == 1 and isinstance(rest[0], Para):
            return sign, rest[0]
    return 1, None


def _classify_matmul(mm: Mat_Mul) -> Optional[Tuple[str, Expr, Para, int]]:
    """Classify a Mat_Mul as row-scale / col-scale / None.

    Returns ``(kind, var_expr, matrix_para, extra_sign)`` where kind is
    ``'row_scale'`` or ``'col_scale'``, ``extra_sign`` ∈ {+1, −1} captures
    a ``Mul(-1, Matrix)`` factor, or ``None`` if the shape is not
    recognised.
    """
    args = list(mm.args)
    if len(args) != 2:
        return None
    left, right = args
    # Diag(var) @ (±Matrix)
    if isinstance(left, Diag):
        sign, para = _unwrap_negated_para(right)
        if para is not None:
            return ('row_scale', left.args[0], para, sign)
    # (±Matrix) @ Diag(var)
    if isinstance(right, Diag):
        sign, para = _unwrap_negated_para(left)
        if para is not None:
            return ('col_scale', right.args[0], para, sign)
    return None


def _sparse_matrix_nnz(para: Para, PARAM) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (coo_row, coo_col, data) for a sparse Para's current value.

    The caller promises the matrix is immutable after modelling (see the
    documentation), so these arrays reflect the baked-in runtime state.
    """
    value = PARAM[para.name].get_v_t(0)
    coo = value.tocoo()
    return (np.asarray(coo.row, dtype=np.int64),
            np.asarray(coo.col, dtype=np.int64),
            np.asarray(coo.data, dtype=np.float64))


def analyze_mutable_mat_expr(expr: Expr,
                              value0_row: np.ndarray,
                              value0_col: np.ndarray,
                              PARAM,
                              eqn_size: int) -> MutableMatBlockMapping:
    """Decompose ``expr`` and build a :class:`MutableMatBlockMapping`.

    Parameters
    ----------
    expr : SymPy Expr
        The mutable-matrix Jacobian block expression, in SpDeriExpr form
        (already Mat_Mul-aware, Diag uses the original ``Diag`` class).
    value0_row, value0_col : ndarray
        Output sparsity pattern (one entry per nnz in the block's Value0).
        Obtained from the perturbed ``Value0.tocoo()`` — this is the full
        structural union of all term patterns.
    PARAM : dict
        The equation system's PARAM dict (needed to fetch sparse matrix
        .data/.indices at analysis time).
    eqn_size : int
        The block's equation-axis length (number of rows).

    Returns
    -------
    MutableMatBlockMapping
        Pre-computed mapping with typed term lists. Any terms that
        don't fit the supported shapes are dropped into ``fallback_expr``.
    """
    n_out = len(value0_row)
    pos_lookup: Dict[Tuple[int, int], int] = {
        (int(r), int(c)): i
        for i, (r, c) in enumerate(zip(value0_row, value0_col))
    }
    mapping = MutableMatBlockMapping(n_out)

    fallback_pieces: List[Expr] = []

    def handle(term: Expr):
        sign, core = _extract_sign_and_core(term)
        # Distribute Mat_Mul over Add in the operand: the matrix-calculus
        # engine typically emits ``L @ (Diag(u) + Diag(v))`` for the
        # derivative of ``L @ (u ⊙ m)``, and we need each ``Diag`` term to
        # be classified independently. Apply the identity
        # ``A @ (X + Y) = A @ X + A @ Y`` (and the symmetric form) and
        # recurse into each summand.
        if isinstance(core, Mat_Mul) and len(core.args) == 2:
            left, right = core.args
            if isinstance(right, Add):
                for sub in right.args:
                    handle(sign * Mat_Mul(left, sub))
                return
            if isinstance(left, Add):
                for sub in left.args:
                    handle(sign * Mat_Mul(sub, right))
                return
        # Diag(inner)
        if isinstance(core, Diag):
            inner = core.args[0]
            # Identify output positions on the diagonal (row == col)
            out_pos_list = []
            src_idx_list = []
            for i in range(eqn_size):
                k = pos_lookup.get((i, i))
                if k is not None:
                    out_pos_list.append(k)
                    src_idx_list.append(i)
            mapping.diag_terms.append({
                'sign': sign,
                'inner_expr': inner,
                'out_pos': np.asarray(out_pos_list, dtype=np.int64),
                'src_idx': np.asarray(src_idx_list, dtype=np.int64),
            })
            return
        # Mat_Mul(Diag, Matrix) or Mat_Mul(Matrix, Diag)
        if isinstance(core, Mat_Mul):
            classification = _classify_matmul(core)
            if classification is not None:
                kind, var_expr, matrix_para, extra_sign = classification
                sign = sign * extra_sign
                try:
                    mat_row, mat_col, mat_data = _sparse_matrix_nnz(matrix_para, PARAM)
                except Exception:
                    fallback_pieces.append(term)
                    return
                out_pos_list = []
                src_list = []
                # Build term's nnz → output mapping
                for k, (r, c) in enumerate(zip(mat_row, mat_col)):
                    out_k = pos_lookup.get((int(r), int(c)))
                    if out_k is None:
                        # This element is not in the init pattern — shouldn't
                        # happen when Value0 is computed via SpDiag + union,
                        # but skip safely if it does.
                        continue
                    out_pos_list.append(out_k)
                    src_list.append(int(r) if kind == 'row_scale' else int(c))
                entry = {
                    'sign': sign,
                    'var_expr': var_expr,
                    'matrix_name': matrix_para.name,
                    'out_pos': np.asarray(out_pos_list, dtype=np.int64),
                    'src': np.asarray(src_list, dtype=np.int64),
                    'mat_data': mat_data,
                }
                if kind == 'row_scale':
                    mapping.row_scale_terms.append(entry)
                else:
                    mapping.col_scale_terms.append(entry)
                return
        # Anything else — fall back to sparse evaluation
        fallback_pieces.append(term)

    if isinstance(expr, Add):
        for t in expr.args:
            handle(t)
    else:
        handle(expr)

    if fallback_pieces:
        mapping.fallback_expr = Add(*fallback_pieces) if len(fallback_pieces) > 1 else fallback_pieces[0]
        mapping.fallback_out_row = np.asarray(value0_row, dtype=np.int64)
        mapping.fallback_out_col = np.asarray(value0_col, dtype=np.int64)
        mapping.has_fallback = True

    return mapping


def generate_block_function_code(fn_name: str,
                                   mapping: MutableMatBlockMapping,
                                   diag_arg_names: List[str],
                                   rs_arg_names: List[str],
                                   cs_arg_names: List[str]) -> str:
    """Generate a @njit function that builds one mutable matrix block's
    data array.

    The generated kernel takes **all** dense vectors — diag inner
    vectors AND row/col-scale vectors — as arguments, pre-computed by
    the J_ wrapper. This lets the kernel be completely free of scipy
    sparse objects and base variable slicing, so it can compile
    cleanly under Numba regardless of how complex the original
    ``var_expr`` inside each ``Diag(...)`` was.

    Parameters
    ----------
    fn_name : str
        Name of the generated function, e.g. ``'_mut_block_0'``.
    mapping : MutableMatBlockMapping
        The pre-computed decomposition of the block.
    diag_arg_names : list of str
        Per-diag-term inner vector argument names. Length =
        ``len(mapping.diag_terms)``.
    rs_arg_names : list of str
        Per-row-scale-term scaling vector argument names. Length =
        ``len(mapping.row_scale_terms)``. For a term
        ``Mat_Mul(Diag(v_expr), M)`` this vector is ``v_expr`` evaluated
        as a dense vector in the wrapper, and the kernel reads
        ``rsv[src[k]]`` per nonzero of M.
    cs_arg_names : list of str
        Per-col-scale-term scaling vector argument names. Same role as
        row-scale but for ``Mat_Mul(M, Diag(v_expr))``.

    Returns
    -------
    str
        Full function source (no decorator; the decorator is added
        later by the module generator). Ready to be appended to the
        generated module file.
    """
    # Build arg list: all dense vectors first, then per-term mapping
    # arrays. No base variables needed — every term reads from a
    # pre-computed dense vector.
    args: List[str] = []
    args.extend(diag_arg_names)
    args.extend(rs_arg_names)
    args.extend(cs_arg_names)

    body_lines: List[str] = []
    body_lines.append(f'    data = zeros({mapping.n_out})')

    # Diag terms — ADDITIVE update at diagonal output positions. Using
    # ``+=`` (not ``=``) is critical when the expression has multiple
    # independent diag contributions on the same diagonal entries, e.g.
    # ``Diag(A@x) + Diag(B@y)`` — both terms land on the same (i, i)
    # positions and must accumulate. ``data`` starts at zero, so the
    # additive update is also correct for the first term in isolation.
    for ti, t in enumerate(mapping.diag_terms):
        out_name = f'_sz_mb_diag_out_{ti}'
        src_name = f'_sz_mb_diag_src_{ti}'
        args.extend([out_name, src_name])
        sign = t['sign']
        u_name = diag_arg_names[ti]
        body_lines.append(f'    for i in range({out_name}.shape[0]):')
        if sign == 1:
            body_lines.append(f'        data[{out_name}[i]] += {u_name}[{src_name}[i]]')
        else:
            body_lines.append(f'        data[{out_name}[i]] -= {u_name}[{src_name}[i]]')

    # Row-scale terms: data[out[k]] += sign * rsv[src[k]] * mat_data[k]
    for ti, t in enumerate(mapping.row_scale_terms):
        out_name = f'_sz_mb_rs_out_{ti}'
        src_name = f'_sz_mb_rs_src_{ti}'
        dat_name = f'_sz_mb_rs_dat_{ti}'
        args.extend([out_name, src_name, dat_name])
        sign = t['sign']
        rsv_name = rs_arg_names[ti]
        body_lines.append(f'    for i in range({out_name}.shape[0]):')
        if sign == 1:
            body_lines.append(
                f'        data[{out_name}[i]] += {rsv_name}[{src_name}[i]] * {dat_name}[i]')
        else:
            body_lines.append(
                f'        data[{out_name}[i]] -= {rsv_name}[{src_name}[i]] * {dat_name}[i]')

    # Col-scale terms: data[out[k]] += sign * csv[src[k]] * mat_data[k]
    for ti, t in enumerate(mapping.col_scale_terms):
        out_name = f'_sz_mb_cs_out_{ti}'
        src_name = f'_sz_mb_cs_src_{ti}'
        dat_name = f'_sz_mb_cs_dat_{ti}'
        args.extend([out_name, src_name, dat_name])
        sign = t['sign']
        csv_name = cs_arg_names[ti]
        body_lines.append(f'    for i in range({out_name}.shape[0]):')
        if sign == 1:
            body_lines.append(
                f'        data[{out_name}[i]] += {csv_name}[{src_name}[i]] * {dat_name}[i]')
        else:
            body_lines.append(
                f'        data[{out_name}[i]] -= {csv_name}[{src_name}[i]] * {dat_name}[i]')

    body_lines.append('    return data')

    signature = f'def {fn_name}(' + ', '.join(args) + '):'
    return signature + '\n' + '\n'.join(body_lines) + '\n'


def _var_base_name(var_expr: Expr) -> str:
    """Return the base variable name for use in the function signature.

    For ``iVar('e')`` → ``'e'``; for ``IdxVar('e[5:29]')`` → ``'e'``.
    The full access expression (including slice) is produced separately
    by :func:`_var_access_expr`.
    """
    if isinstance(var_expr, iVar):
        return var_expr.name
    if isinstance(var_expr, IdxVar):
        return var_expr.name0
    raise NotImplementedError(
        f"var_expr inside Diag(...) must be a plain variable or slice, "
        f"got {type(var_expr).__name__}: {var_expr}")


def _var_access_expr(var_expr: Expr, var_arg_names: Dict[str, str]) -> str:
    """Return the runtime access expression inside the @njit function body.

    For ``iVar('e')`` → ``'e'``; for ``IdxVar('e[5:29]')`` → ``'e[5:29]'``
    (a numpy slice). The base name is remapped through ``var_arg_names``
    if the caller renamed arguments in the signature.
    """
    if isinstance(var_expr, iVar):
        return var_arg_names.get(var_expr.name, var_expr.name)
    if isinstance(var_expr, IdxVar):
        base = var_arg_names.get(var_expr.name0, var_expr.name0)
        idx = var_expr.index
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else ''
            stop = idx.stop if idx.stop is not None else ''
            step = f':{idx.step}' if idx.step is not None else ''
            return f'{base}[{start}:{stop}{step}]'
        if isinstance(idx, (int, np.integer)):
            return f'{base}[{int(idx)}]'
    raise NotImplementedError(
        f"var_expr inside Diag(...) must be a plain variable or slice, "
        f"got {type(var_expr).__name__}: {var_expr}")
