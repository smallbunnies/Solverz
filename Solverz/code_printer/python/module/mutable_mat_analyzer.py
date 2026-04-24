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
from sympy import Add, Mul, S, Expr

from Solverz.sym_algebra.functions import Diag, Mat_Mul
from Solverz.sym_algebra.symbols import Para


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
                 'col_scale_terms', 'biscale_terms', 'fallback_expr',
                 'fallback_out_row', 'fallback_out_col', 'has_fallback')

    def __init__(self, n_out: int):
        self.n_out = n_out
        self.diag_terms: List[Dict] = []
        self.row_scale_terms: List[Dict] = []
        self.col_scale_terms: List[Dict] = []
        # biscale: ``Diag(u) @ Matrix @ Diag(v)`` — each nnz ``(r, c, d)``
        # contributes ``u[r] * v[c] * d`` to output ``(r, c)``. Each entry
        # has ``sign``, ``u_expr``, ``v_expr``, ``matrix_name``,
        # ``out_pos``, ``src_row``, ``src_col``, ``mat_data``.
        self.biscale_terms: List[Dict] = []
        self.fallback_expr: Optional[Expr] = None
        self.fallback_out_row: Optional[np.ndarray] = None
        self.fallback_out_col: Optional[np.ndarray] = None
        self.has_fallback: bool = False


def _extract_sign_and_core(term: Expr) -> Tuple[float, Expr]:
    """Return ``(coeff, core)`` where ``core`` has no leading numeric
    scalar and ``coeff`` is the product of all ``Integer`` / ``Rational``
    / ``Float`` / ``Number`` factors (including ``-1``).

    Examples:
    * ``-Diag(x)`` → ``(-1, Diag(x))``
    * ``2*Diag(x)`` → ``(2, Diag(x))``
    * ``-2 * Mat_Mul(A, B)`` → ``(-2, Mat_Mul(A, B))``
    """
    from sympy import Number, Integer, Rational, Float

    if isinstance(term, Mul):
        args = list(term.args)
        coeff: float = 1.0
        rest: List[Expr] = []
        for a in args:
            if isinstance(a, (Integer, Rational, Float, Number)):
                coeff = coeff * float(a)
            else:
                rest.append(a)
        if len(rest) == 1:
            return coeff, rest[0]
        return coeff, (Mul(*rest) if rest else S.One)
    return 1.0, term


def _is_supported_matrix_operand(expr: Expr) -> bool:
    """True if ``expr`` is a constant sparse matrix operand that the
    analyzer can materialise to COO data — ``Para``, ``_LoopJacSelectMat``,
    or a ``Mat_Mul`` chain of such leaves (optional ``-1`` scalar)."""
    from Solverz.equation.eqn import _LoopJacSelectMat

    if isinstance(expr, Para):
        return True
    if isinstance(expr, _LoopJacSelectMat):
        return True
    if isinstance(expr, Mul):
        rest = [a for a in expr.args if a != S.NegativeOne]
        if len(rest) == 1:
            return _is_supported_matrix_operand(rest[0])
        return False
    if isinstance(expr, Mat_Mul):
        return all(_is_supported_matrix_operand(a) for a in expr.args)
    return False


def _unwrap_sign_and_matrix(expr: Expr) -> Tuple[int, Optional[Expr]]:
    """If ``expr`` or ``-expr`` reduces to a supported constant matrix
    operand (``Para`` / ``_LoopJacSelectMat`` / ``Mat_Mul`` chain
    thereof), return ``(sign, core)``. Otherwise ``(1, None)``.

    Strip a top-level ``Mul(-1, ...)`` wrapper *before* consulting
    :func:`_is_supported_matrix_operand` — that predicate also tolerates
    a leading ``-1`` recursively (so negated leaves inside
    ``Mat_Mul.args`` keep matching), and short-circuiting on it first
    would silently drop the outer sign.
    """
    if isinstance(expr, Mul):
        sign = 1
        rest: List[Expr] = []
        for a in expr.args:
            if a == S.NegativeOne:
                sign *= -1
            else:
                rest.append(a)
        if len(rest) == 1 and _is_supported_matrix_operand(rest[0]):
            return sign, rest[0]
        return 1, None
    if _is_supported_matrix_operand(expr):
        return 1, expr
    return 1, None


def _matrix_chain_expr(args: List[Expr]) -> Expr:
    """Build a single constant-matrix expression from a list of
    supported operands, unwrapping a single-item list."""
    if len(args) == 1:
        return args[0]
    return Mat_Mul(*args)


def _classify_matmul(mm: Mat_Mul) -> Optional[Tuple[str, Expr, Expr, int]]:
    """Classify a Mat_Mul as row-scale / col-scale / None.

    Returns ``(kind, var_expr, matrix_expr, extra_sign)`` where
    ``matrix_expr`` is any constant-matrix operand supported by
    :func:`_sparse_matrix_nnz` (``Para``, ``_LoopJacSelectMat``, or a
    ``Mat_Mul`` chain of constants). ``extra_sign`` ∈ {+1, −1}.

    Handles variadic ``Mat_Mul`` args so that
    ``Mat_Mul(SelectMat, Para, Diag(v))`` (a 3-arg flat form) is
    recognised as col-scale with ``matrix = SelectMat @ Para``.
    Returns ``None`` for shapes with ``Diag`` on both ends (those
    belong to :func:`_classify_matmul_biscale`).
    """
    args = list(mm.args)
    if len(args) < 2:
        return None
    left_diag = args[0] if isinstance(args[0], Diag) else None
    right_diag = args[-1] if isinstance(args[-1], Diag) else None

    if left_diag is None and right_diag is None:
        return None
    if left_diag is not None and right_diag is not None:
        # biscale territory
        return None

    if left_diag is not None:
        matrix_candidate = _matrix_chain_expr(args[1:])
        sign, matrix = _unwrap_sign_and_matrix(matrix_candidate)
        if matrix is not None:
            return ('row_scale', left_diag.args[0], matrix, sign)
        return None
    # right_diag not None
    matrix_candidate = _matrix_chain_expr(args[:-1])
    sign, matrix = _unwrap_sign_and_matrix(matrix_candidate)
    if matrix is not None:
        return ('col_scale', right_diag.args[0], matrix, sign)
    return None


def _classify_matmul_biscale(mm: Mat_Mul) -> Optional[Tuple[Expr, Expr, Expr, int]]:
    """Classify a Mat_Mul as ``Diag(u) @ M @ Diag(v)`` where ``M`` is a
    supported constant-matrix operand (``Para``, ``_LoopJacSelectMat``,
    or a ``Mat_Mul`` chain of such).

    Accepts all of:

    * Flat N-arg form ``Mat_Mul(Diag(u), ..., Diag(v))``.
    * 2-arg nested forms
      ``Mat_Mul(Diag(u), Mat_Mul(Matrix, Diag(v)))`` /
      ``Mat_Mul(Mat_Mul(Diag(u), Matrix), Diag(v))``.
    * The above with a leading ``-1`` on either side wrapped as
      ``Mul(-1, Mat_Mul(...))``.
    """
    def _peek(side: Expr):
        """Unwrap ``-1 * x`` / ``c * x``, returning ``(extra_sign, x)``
        where ``extra_sign`` is the numeric coefficient (for now only
        ``+1``/``-1``; non-unit scalars should have been absorbed into
        ``coeff`` by ``_extract_sign_and_core`` at the outer level)."""
        if isinstance(side, Mul):
            rest = [a for a in side.args if a != S.NegativeOne]
            neg = len(side.args) - len(rest)
            sign = -1 if neg % 2 else 1
            if len(rest) == 1:
                return sign, rest[0]
        return 1, side

    args = list(mm.args)
    # Flat: Diag(u) @ <matrix chain> @ Diag(v), len(args) >= 3
    if (len(args) >= 3
            and isinstance(args[0], Diag)
            and isinstance(args[-1], Diag)):
        sign, matrix = _unwrap_sign_and_matrix(
            _matrix_chain_expr(args[1:-1])
        )
        if matrix is not None:
            return (args[0].args[0], matrix, args[-1].args[0], sign)
    if len(args) != 2:
        return None
    left, right = args
    # Diag(u) @ (±(Matrix_chain @ Diag(v)))
    if isinstance(left, Diag):
        inner_sign, inner = _peek(right)
        if isinstance(inner, Mat_Mul):
            r_args = list(inner.args)
            if len(r_args) >= 2 and isinstance(r_args[-1], Diag):
                sign, matrix = _unwrap_sign_and_matrix(
                    _matrix_chain_expr(r_args[:-1])
                )
                if matrix is not None:
                    return (left.args[0], matrix, r_args[-1].args[0],
                            inner_sign * sign)
    # (±(Diag(u) @ Matrix_chain)) @ Diag(v)
    if isinstance(right, Diag):
        inner_sign, inner = _peek(left)
        if isinstance(inner, Mat_Mul):
            l_args = list(inner.args)
            if len(l_args) >= 2 and isinstance(l_args[0], Diag):
                sign, matrix = _unwrap_sign_and_matrix(
                    _matrix_chain_expr(l_args[1:])
                )
                if matrix is not None:
                    return (l_args[0].args[0], matrix, right.args[0],
                            inner_sign * sign)
    return None


def _matrix_debug_name(expr: Expr) -> str:
    """Human-readable tag for a matrix operand — used only for debug
    prints / exception messages (``matrix_name`` in the analyzer
    entry). Analyzer correctness depends on ``mat_data`` / ``src``,
    not this string."""
    from Solverz.equation.eqn import _LoopJacSelectMat

    if isinstance(expr, Para):
        return expr.name
    if isinstance(expr, _LoopJacSelectMat):
        return '_LoopJacSelectMat'
    if isinstance(expr, Mat_Mul):
        return '@'.join(_matrix_debug_name(a) for a in expr.args)
    if isinstance(expr, Mul):
        return '*'.join(_matrix_debug_name(a) if not isinstance(a, (int, float))
                        else str(a) for a in expr.args)
    return type(expr).__name__


def _sparse_matrix_nnz(
    matrix_expr: Expr, PARAM
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(coo_row, coo_col, data)`` for any constant sparse
    matrix expression the analyzer accepts.

    Supports:

    * ``Para`` — looked up in ``PARAM`` and its value's COO extracted.
    * ``_LoopJacSelectMat`` — the selection matrix is built inline from
      its ``col_map`` argument (row ``i`` carries a single 1 at column
      ``col_map[i]``).
    * ``Mat_Mul`` of supported operands — evaluated by chaining
      scipy-sparse multiplies once at analysis time. The result's
      COO form is returned.

    Signs are not handled here; the caller extracts them via
    :func:`_unwrap_sign_and_matrix`.
    """
    from scipy.sparse import csc_array, csr_array
    from Solverz.equation.eqn import _LoopJacSelectMat

    def materialise(expr: Expr):
        if isinstance(expr, Para):
            value = PARAM[expr.name].get_v_t(0)
            if not hasattr(value, 'tocoo'):
                return csc_array(np.asarray(value))
            return value
        if isinstance(expr, _LoopJacSelectMat):
            col_map_tuple, n_outer_sym, n_diff_sym = expr.args
            cols = np.asarray(
                [int(c) for c in col_map_tuple.args], dtype=np.int64)
            n_outer = int(n_outer_sym)
            n_diff = int(n_diff_sym)
            rows = np.arange(n_outer, dtype=np.int64)
            data = np.ones(n_outer, dtype=np.float64)
            return csc_array((data, (rows, cols)), shape=(n_outer, n_diff))
        if isinstance(expr, Mul):
            # Strip leading -1 — sign is handled by the caller.
            rest = [a for a in expr.args if a != S.NegativeOne]
            if len(rest) == 1:
                return materialise(rest[0])
            raise TypeError(
                f"_sparse_matrix_nnz: unexpected Mul {expr!r}")
        if isinstance(expr, Mat_Mul):
            factors = [materialise(a) for a in expr.args]
            result = factors[0]
            for nxt in factors[1:]:
                result = result @ nxt
            return result
        raise TypeError(
            f"_sparse_matrix_nnz: unsupported operand {type(expr).__name__}")

    mat = materialise(matrix_expr)
    coo = mat.tocoo()
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
        # Mat_Mul with a biscale shape ``Diag(u) @ M @ Diag(v)`` (tried
        # before the row/col-scale classifier because the two-scale case
        # is a strict superset of either single-scale case — if only one
        # side is ``Diag``, the biscale matcher returns ``None`` and we
        # fall through to ``_classify_matmul``).
        if isinstance(core, Mat_Mul):
            biscale = _classify_matmul_biscale(core)
            if biscale is not None:
                u_expr, matrix_expr, v_expr, extra_sign = biscale
                sign = sign * extra_sign
                try:
                    mat_row, mat_col, mat_data = _sparse_matrix_nnz(
                        matrix_expr, PARAM)
                except Exception:
                    fallback_pieces.append(term)
                    return
                out_pos_list = []
                row_src = []
                col_src = []
                mat_data_filtered = []
                for k, (r, c) in enumerate(zip(mat_row, mat_col)):
                    out_k = pos_lookup.get((int(r), int(c)))
                    if out_k is None:
                        continue
                    out_pos_list.append(out_k)
                    row_src.append(int(r))
                    col_src.append(int(c))
                    mat_data_filtered.append(mat_data[k])
                mapping.biscale_terms.append({
                    'sign': sign,
                    'u_expr': u_expr,
                    'v_expr': v_expr,
                    'matrix_name': _matrix_debug_name(matrix_expr),
                    'out_pos': np.asarray(out_pos_list, dtype=np.int64),
                    'src_row': np.asarray(row_src, dtype=np.int64),
                    'src_col': np.asarray(col_src, dtype=np.int64),
                    'mat_data': np.asarray(mat_data_filtered, dtype=np.float64),
                })
                return
        # Mat_Mul(Diag, Matrix) or Mat_Mul(Matrix, Diag)
        if isinstance(core, Mat_Mul):
            classification = _classify_matmul(core)
            if classification is not None:
                kind, var_expr, matrix_expr, extra_sign = classification
                sign = sign * extra_sign
                try:
                    mat_row, mat_col, mat_data = _sparse_matrix_nnz(matrix_expr, PARAM)
                except Exception:
                    fallback_pieces.append(term)
                    return
                out_pos_list = []
                src_list = []
                mat_data_filtered = []
                # Build term's nnz → output mapping
                for k, (r, c) in enumerate(zip(mat_row, mat_col)):
                    out_k = pos_lookup.get((int(r), int(c)))
                    if out_k is None:
                        # This element is not in the init pattern —
                        # shouldn't happen when Value0 is the structural
                        # union, but skip safely if it does.
                        continue
                    out_pos_list.append(out_k)
                    src_list.append(int(r) if kind == 'row_scale' else int(c))
                    mat_data_filtered.append(mat_data[k])
                entry = {
                    'sign': sign,
                    'var_expr': var_expr,
                    'matrix_name': _matrix_debug_name(matrix_expr),
                    'out_pos': np.asarray(out_pos_list, dtype=np.int64),
                    'src': np.asarray(src_list, dtype=np.int64),
                    'mat_data': np.asarray(mat_data_filtered, dtype=np.float64),
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
                                   cs_arg_names: List[str],
                                   bs_u_arg_names: List[str] = None,
                                   bs_v_arg_names: List[str] = None) -> str:
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
    if bs_u_arg_names is None:
        bs_u_arg_names = []
    if bs_v_arg_names is None:
        bs_v_arg_names = []
    args: List[str] = []
    args.extend(diag_arg_names)
    args.extend(rs_arg_names)
    args.extend(cs_arg_names)
    args.extend(bs_u_arg_names)
    args.extend(bs_v_arg_names)

    body_lines: List[str] = []
    body_lines.append(f'    data = zeros({mapping.n_out})')

    # Diag terms — ADDITIVE update at diagonal output positions. Using
    # ``+=`` (not ``=``) is critical when the expression has multiple
    # independent diag contributions on the same diagonal entries, e.g.
    # ``Diag(A@x) + Diag(B@y)`` — both terms land on the same (i, i)
    # positions and must accumulate. ``data`` starts at zero, so the
    # additive update is also correct for the first term in isolation.
    def _scale_prefix(sign: float) -> str:
        """Return an expression prefix that scales the RHS by ``sign``:
        ``+= X`` for +1, ``-= X`` for -1, ``+= c * X`` for any other
        numeric coefficient."""
        if abs(sign - 1.0) < 1e-12:
            return '        data[{out}[i]] += '
        if abs(sign + 1.0) < 1e-12:
            return '        data[{out}[i]] -= '
        return f'        data[{{out}}[i]] += {float(sign)!r} * '

    for ti, t in enumerate(mapping.diag_terms):
        out_name = f'_sz_mb_diag_out_{ti}'
        src_name = f'_sz_mb_diag_src_{ti}'
        args.extend([out_name, src_name])
        sign = t['sign']
        u_name = diag_arg_names[ti]
        prefix = _scale_prefix(sign).format(out=out_name)
        body_lines.append(f'    for i in range({out_name}.shape[0]):')
        body_lines.append(prefix + f'{u_name}[{src_name}[i]]')

    # Row-scale terms: data[out[k]] += sign * rsv[src[k]] * mat_data[k]
    for ti, t in enumerate(mapping.row_scale_terms):
        out_name = f'_sz_mb_rs_out_{ti}'
        src_name = f'_sz_mb_rs_src_{ti}'
        dat_name = f'_sz_mb_rs_dat_{ti}'
        args.extend([out_name, src_name, dat_name])
        sign = t['sign']
        rsv_name = rs_arg_names[ti]
        prefix = _scale_prefix(sign).format(out=out_name)
        body_lines.append(f'    for i in range({out_name}.shape[0]):')
        body_lines.append(prefix + f'{rsv_name}[{src_name}[i]] * {dat_name}[i]')

    # Col-scale terms: data[out[k]] += sign * csv[src[k]] * mat_data[k]
    for ti, t in enumerate(mapping.col_scale_terms):
        out_name = f'_sz_mb_cs_out_{ti}'
        src_name = f'_sz_mb_cs_src_{ti}'
        dat_name = f'_sz_mb_cs_dat_{ti}'
        args.extend([out_name, src_name, dat_name])
        sign = t['sign']
        csv_name = cs_arg_names[ti]
        prefix = _scale_prefix(sign).format(out=out_name)
        body_lines.append(f'    for i in range({out_name}.shape[0]):')
        body_lines.append(prefix + f'{csv_name}[{src_name}[i]] * {dat_name}[i]')

    # Biscale terms: data[out[k]] += sign * u[src_row[k]] * v[src_col[k]] * mat_data[k]
    for ti, t in enumerate(mapping.biscale_terms):
        out_name = f'_sz_mb_bs_out_{ti}'
        row_name = f'_sz_mb_bs_row_{ti}'
        col_name = f'_sz_mb_bs_col_{ti}'
        dat_name = f'_sz_mb_bs_dat_{ti}'
        args.extend([out_name, row_name, col_name, dat_name])
        sign = t['sign']
        u_name = bs_u_arg_names[ti]
        v_name = bs_v_arg_names[ti]
        prefix = _scale_prefix(sign).format(out=out_name)
        body_lines.append(f'    for i in range({out_name}.shape[0]):')
        body_lines.append(
            prefix + f'{u_name}[{row_name}[i]] '
            f'* {v_name}[{col_name}[i]] * {dat_name}[i]'
        )

    body_lines.append('    return data')

    signature = f'def {fn_name}(' + ', '.join(args) + '):'
    return signature + '\n' + '\n'.join(body_lines) + '\n'


