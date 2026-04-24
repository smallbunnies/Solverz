"""LoopEqn Jacobian pipeline (Phase J1).

Takes a raw ``sp.diff`` result on a LoopEqn body and produces a
Solverz expression suitable for the standard ``JacBlock`` machinery.

Two-step pipeline:

1. :func:`canonicalize_kronecker` — manual delta collapse on the raw
   diff. Never calls ``sympy.Sum.doit()`` because ``.doit()`` will
   unroll an ``(j, 0, nb-1)``-bounded ``Sum`` into ``nb`` explicit
   terms whenever the Sum contains a ``KroneckerDelta(k, outer)``
   that can't be collapsed via the sum dummy — the very case we need
   to keep as a template.
2. :func:`loop_jac_to_solverz_expr` — classify the canonicalized
   expression into the constant-shape categories that the current
   ``JacBlock`` + ``is_constant_matrix_deri`` path already handles:

   - Constant identity: ``KroneckerDelta(outer, k)`` → ``_LoopJacEye``
   - Constant 2-D Param entry: ``Indexed(Param, outer, k)`` → ``Para``
   - Para arithmetic: ``Add`` / ``Mul`` thereof (``V_in - V_out``)

Anything that does NOT reduce to one of the above raises
``NotImplementedError``. Phase J2 will extend the pipeline with
``DiagTerm`` / ``RowScaleTerm`` / ``ColScaleTerm`` / ``BilinearEntry``
categories and emit per-block kernels via the existing
``mut_mat_mappings`` / ``mut_mat_block_funcs`` pipeline.

Why a separate module
---------------------
The J-side logic is big enough to warrant its own file, both for
readability (eqn.py was close to 1500 lines) and so the
``canonicalize_kronecker`` helper can be unit-tested in isolation
from the LoopEqn wrapper. The legacy in-line ``_translate_loop_jac``
is gone once ``LoopEqn.derive_derivative`` is rewired to this
module.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import sympy as sp
from sympy.functions.special.tensor_functions import KroneckerDelta

from Solverz.sym_algebra.symbols import Para
from Solverz.utilities.type_checker import is_zero


def canonicalize_kronecker(expr: sp.Expr,
                            outer_idx: sp.Idx,
                            diff_idx: sp.Idx,
                            _expanded: bool = False) -> sp.Expr:
    """Canonicalize a raw ``sp.diff`` output so every ``Sum`` is
    either collapsed (``KroneckerDelta`` on the sum dummy dropped
    along with the Sum) or has its ``KroneckerDelta`` factors pulled
    out in front.

    **Never** calls ``sympy.Sum.doit()`` — that would unroll a
    bounded ``(dummy, 0, n - 1)`` Sum into ``n`` explicit terms
    whenever the Sum contains a ``KroneckerDelta(diff_idx, outer)``
    factor (the ``outer`` argument doesn't match the dummy so the
    Sum cannot collapse via the delta; ``.doit()`` then falls back
    to expanding the range). That's exactly the case we need to
    keep as a template: the Sum stays symbolic, and the classifier
    downstream interprets the structure.

    Strategy (per Sum):

    1. Recurse into the body first — nested Sums get canonicalized
       bottom-up.
    2. Distribute ``Sum(a + b, dummy)`` → ``Sum(a, dummy) +
       Sum(b, dummy)`` so each summand can be classified
       independently.
    3. In each ``Mul`` summand, split factors into
       ``KroneckerDelta``s and "other":

       - If any delta's second argument equals the Sum dummy
         → collapse: substitute ``dummy := other_arg`` in the
         remaining factors, drop the Sum and the delta.
       - If all deltas' arguments are non-dummy → pull them out in
         front of the Sum: ``Sum(δ * rest, dummy) = δ * Sum(rest,
         dummy)``.

    The result is a sum-of-products form where every KroneckerDelta
    is either (a) gone (collapsed) or (b) a top-level coefficient.
    """
    if not _expanded:
        # Push multiplication inside addition at the top level so
        # ``(δ(k,i) - δ(k,j)) * rest`` becomes
        # ``δ(k,i)*rest - δ(k,j)*rest`` BEFORE we try to classify
        # factors. Without this expansion, the per-summand
        # KroneckerDelta detection inside ``_canonicalize_sum``
        # never sees a standalone delta factor — the delta is
        # hidden inside an ``Add`` subexpression and never gets
        # collapsed. This arises naturally from ``sp.diff`` on
        # trig/exponential bodies (probe 6 of the investigation)
        # where the chain rule produces ``d(f(a-b))/dx =
        # f'(a-b) * (∂a/∂x - ∂b/∂x)`` with both partial derivatives
        # being ``KroneckerDelta``s.
        expr = sp.expand_mul(expr)
        return canonicalize_kronecker(expr, outer_idx, diff_idx,
                                      _expanded=True)
    if isinstance(expr, sp.Add):
        return sp.Add(*(
            canonicalize_kronecker(a, outer_idx, diff_idx,
                                   _expanded=True)
            for a in expr.args
        ))
    if isinstance(expr, sp.Mul):
        return sp.Mul(*(
            canonicalize_kronecker(a, outer_idx, diff_idx,
                                   _expanded=True)
            for a in expr.args
        ))
    if isinstance(expr, sp.Sum):
        return _canonicalize_sum(expr, outer_idx, diff_idx)
    # Piecewise can appear if the upstream caller did ``.doit()``;
    # for Phase J1 we defensively unwrap the first non-zero branch.
    if isinstance(expr, sp.Piecewise):
        for branch_expr, _cond in expr.args:
            if not is_zero(branch_expr):
                return canonicalize_kronecker(branch_expr,
                                              outer_idx, diff_idx,
                                              _expanded=True)
        return sp.S.Zero
    return expr


def _canonicalize_sum(sum_node: sp.Sum,
                      outer_idx: sp.Idx,
                      diff_idx: sp.Idx) -> sp.Expr:
    """Canonicalize a single ``Sum`` node. See
    :func:`canonicalize_kronecker` for the high-level strategy.
    """
    body = sum_node.args[0]
    limits = sum_node.args[1:]
    if len(limits) != 1:
        raise NotImplementedError(
            f"canonicalize_kronecker: multi-dummy Sum {sum_node} "
            f"— not supported in Phase J1"
        )
    dummy, lo, hi = limits[0]

    # Recurse into the body first so nested Sums are already
    # canonicalized by the time we inspect the top-level factors.
    body = canonicalize_kronecker(body, outer_idx, diff_idx)

    # Sum over Add → Add of Sums (linearity).
    if isinstance(body, sp.Add):
        return sp.Add(*(
            _canonicalize_sum(sp.Sum(term, (dummy, lo, hi)),
                              outer_idx, diff_idx)
            for term in body.args
        ))

    # Body is now a single Mul (or atom). Pull out KroneckerDeltas.
    if isinstance(body, sp.Mul):
        factors = list(body.args)
    else:
        factors = [body]

    dummy_collapse = None   # (dummy_sym, substitute_to) if a delta matches dummy
    pulled_deltas = []      # deltas whose second arg is NOT dummy
    other_factors = []
    for f in factors:
        if isinstance(f, KroneckerDelta):
            a, b = f.args
            # KroneckerDelta is symmetric: δ(a,b) = δ(b,a). Either
            # side may be the dummy.
            if a == dummy or b == dummy:
                if dummy_collapse is not None:
                    # Two deltas on the same dummy in one Mul —
                    # unusual; would imply the other args must be
                    # equal for the term to be non-zero. Leave as a
                    # NotImplementedError for now.
                    raise NotImplementedError(
                        f"canonicalize_kronecker: multiple "
                        f"KroneckerDelta factors on sum dummy "
                        f"{dummy!r} in {body}"
                    )
                other_side = b if a == dummy else a
                dummy_collapse = (dummy, other_side)
            else:
                # Only pull the delta out if it doesn't reference the
                # Sum dummy at all.  A delta like
                # ``δ(k, map_param[p_p])`` *contains* the dummy ``p_p``
                # inside the ``Indexed`` argument but doesn't *equal*
                # it, so the exact-equality check above misses it.
                # Pulling such a delta out of the Sum creates a stale
                # reference to ``p_p`` that incorrectly evaluates to
                # the last loop-iteration value in generated code.
                if dummy in f.free_symbols:
                    other_factors.append(f)
                else:
                    pulled_deltas.append(f)
        else:
            other_factors.append(f)

    remaining_body = (sp.Mul(*other_factors)
                      if other_factors else sp.S.One)

    if dummy_collapse is not None:
        # Dummy → other_side in the remaining body, drop the Sum +
        # the matched delta.
        dummy_sym, replacement = dummy_collapse
        collapsed = remaining_body.subs(dummy_sym, replacement)
        # Re-multiply any other (pulled) deltas.
        for d in pulled_deltas:
            collapsed = collapsed * d
        return collapsed

    # No delta matched the dummy: pull out any non-dummy deltas and
    # keep the Sum alive around the remaining body.
    inner = remaining_body
    result: sp.Expr = sp.Sum(inner, (dummy, lo, hi))
    for d in pulled_deltas:
        result = d * result
    return result


def loop_jac_to_solverz_expr(expr: sp.Expr,
                              outer_idx: sp.Idx,
                              diff_idx: sp.Idx,
                              n_outer: int,
                              var_map: Dict[str, object],
                              n_diff: int = 0) -> sp.Expr:
    """Translate a *canonicalized* LoopEqn Jacobian expression back
    to a Solverz expression that ``FormJac`` and ``JacBlock`` can
    process through the existing constant / mutable-matrix paths.

    Phase J1 — **constant** patterns handled via
    ``is_constant_matrix_deri``:

    - ``KroneckerDelta(outer, diff)`` → ``_LoopJacEye(n_outer)``
    - ``Indexed(Param, outer, diff)`` with ``Param`` a 2-D ``Param``
      → ``Para(name, dim=2)``
    - ``Add`` / ``Mul`` / numeric coeffs over the above

    Phase J2 — **mutable** patterns handled via the existing
    ``mutable_mat_analyzer`` (Solverz ``Diag`` / ``Mat_Mul``
    decomposition into diag / row-scale / col-scale terms):

    - ``δ(outer, diff) * Sum(Param[outer, dummy] * Var[dummy], dummy)``
      (DiagTerm) → ``Diag(Mat_Mul(Para(name, dim=2), iVar(name)))``
    - ``Indexed(Param, outer, diff) * Indexed(Var, outer)``
      (RowScale) → ``Mat_Mul(Diag(iVar(name)), Para(name, dim=2))``
    - ``Indexed(Param, outer, diff) * Indexed(Var, diff)``
      (ColScale) → ``Mat_Mul(Para(name, dim=2), Diag(iVar(name)))``

    Phase J3 (deferred):

    - ``Indexed(Param, outer, diff) * h(Var[outer], Var[diff])``
      with arbitrary ``h`` (e.g. ``cos(Va[i] - Va[k])``) — requires
      a LoopEqn-native per-entry scatter kernel.
    - Fully dense bilinear blocks with no ``Param`` carrier.
    """
    from Solverz.equation.eqn import _LoopJacEye
    from Solverz.equation.param import ParamBase

    # Top-level: distribute Add. Each addend is classified and
    # translated independently; the result is reassembled with sp.Add.
    if isinstance(expr, sp.Add):
        return sp.Add(*(
            loop_jac_to_solverz_expr(t, outer_idx, diff_idx,
                                      n_outer, var_map, n_diff=n_diff)
            for t in expr.args
        ))

    # Numeric constant — pass through.
    if isinstance(expr, (sp.Integer, sp.Float, sp.Rational, sp.Number)):
        return expr

    # Single-atom cases that don't need the Mul-splitting classifier.
    if isinstance(expr, KroneckerDelta):
        result = _translate_kronecker_delta(
            expr, outer_idx, diff_idx, n_outer, var_map, n_diff=n_diff)
        if result is not None:
            return result
        raise NotImplementedError(
            f"LoopEqn J translator: unsupported KroneckerDelta args "
            f"{expr.args}"
        )

    if isinstance(expr, sp.Indexed):
        return _translate_indexed_param(
            expr, outer_idx, diff_idx, var_map
        )

    # Everything from here is a Mul (or a Sum without sign — fall
    # through to classifier) representing one term of the sum.
    # Extract the sign, examine the remaining factors, and classify.
    return _classify_and_translate_term(
        expr, outer_idx, diff_idx, n_outer, var_map, n_diff=n_diff
    )


def _translate_kronecker_delta(
        expr: KroneckerDelta,
        outer_idx: sp.Idx,
        diff_idx: sp.Idx,
        n_outer: int,
        var_map: Dict[str, object],
        n_diff: int = 0,
):
    """Translate a ``KroneckerDelta`` to a constant J2 expression.

    Returns ``None`` if the pattern is not recognized.

    Recognized patterns:

    - ``KD(outer, diff)`` → ``_LoopJacEye(n_outer)``
    - ``KD(diff, map_param[outer])`` → ``_LoopJacSelectMat(...)``
      where *map_param* is a 1-D integer ``Param`` that maps each
      outer index to a column.  The result is a sparse selection
      matrix with one 1 per row.
    """
    import numpy as np
    from Solverz.equation.eqn import _LoopJacEye, _LoopJacSelectMat
    from Solverz.equation.param import ParamBase

    outer_name = _name_of(outer_idx)
    diff_name = _name_of(diff_idx)

    a0, a1 = expr.args

    # --- direct diagonal: KD(outer, diff) ---
    if ({_name_of(a0), _name_of(a1)}
            == {outer_name, diff_name}):
        return _LoopJacEye(sp.Integer(n_outer))

    # --- indirect diagonal: KD(diff, map[outer]) ---
    for arg_d, arg_m in [(a0, a1), (a1, a0)]:
        if not (isinstance(arg_d, sp.Idx)
                and _name_of(arg_d) == diff_name):
            continue
        if not isinstance(arg_m, sp.Indexed):
            continue
        if len(arg_m.indices) != 1:
            continue
        inner = arg_m.indices[0]
        if not (isinstance(inner, sp.Idx)
                and _name_of(inner) == outer_name):
            continue
        map_name = arg_m.base.name
        map_obj = var_map.get(map_name)
        if not isinstance(map_obj, ParamBase):
            continue
        if getattr(map_obj, 'dim', None) != 1:
            continue
        # Build the selection matrix node.  Use the caller-supplied
        # n_diff (the actual variable size) when available; fall back
        # to max(col_map)+1 which is a safe lower bound.
        col_map = np.asarray(map_obj.v, dtype=np.int64).reshape(-1)
        nd = n_diff if n_diff > 0 else (int(col_map.max()) + 1 if len(col_map) else 0)
        return _LoopJacSelectMat(
            sp.Tuple(*[sp.Integer(int(c)) for c in col_map]),
            sp.Integer(n_outer),
            sp.Integer(nd),
        )

    return None


def _translate_indexed_param(e: sp.Indexed,
                              outer_idx: sp.Idx,
                              diff_idx: sp.Idx,
                              var_map: Dict[str, object]) -> sp.Expr:
    """Translate a bare ``Indexed(Param, (outer, diff))`` node — the
    constant 2-D-Param case — into ``Para(name, dim=2)``.

    Used both by the top-level walker (for unwrapped single-term
    expressions) and by the per-term classifier inside
    :func:`_classify_and_translate_term`.
    """
    from Solverz.equation.param import ParamBase

    base_name = e.base.name
    sol_obj = var_map.get(base_name)
    if sol_obj is None:
        raise NotImplementedError(
            f"LoopEqn J translator: IndexedBase {base_name!r} has "
            f"no var_map entry"
        )
    if not (isinstance(sol_obj, ParamBase) and sol_obj.dim == 2):
        raise NotImplementedError(
            f"LoopEqn J translator: bare IndexedBase {base_name!r} "
            f"of {type(sol_obj).__name__} dim="
            f"{getattr(sol_obj, 'dim', None)} — expected a 2-D Param"
        )
    if len(e.indices) != 2:
        raise NotImplementedError(
            f"LoopEqn J translator: expected 2 indices on "
            f"{base_name!r}, got {e}"
        )
    idx_names = {_name_of(ix) for ix in e.indices}
    if idx_names != {_name_of(outer_idx), _name_of(diff_idx)}:
        raise NotImplementedError(
            f"LoopEqn J translator: 2-D Param {base_name!r} accessed "
            f"with indices {e.indices} — expected [outer, diff] "
            f"(in either order)"
        )
    return Para(sol_obj.name, dim=2)


def _classify_and_translate_term(term: sp.Expr,
                                  outer_idx: sp.Idx,
                                  diff_idx: sp.Idx,
                                  n_outer: int,
                                  var_map: Dict[str, object],
                                  n_diff: int = 0) -> sp.Expr:
    """Classify a single *term* (post canonicalize_kronecker) into
    one of the Phase J1 / J2 shapes and translate it to a Solverz
    expression.

    A term is either a bare factor or a ``Mul`` of factors. We
    bucket the factors into KroneckerDeltas / Indexed / Sums /
    numeric coefficients / other, and then match against the known
    shapes in priority order (most specific first).
    """
    from Solverz.equation.eqn import _LoopJacEye
    from Solverz.equation.param import ParamBase
    from Solverz.sym_algebra.functions import Diag, Mat_Mul
    from Solverz.sym_algebra.symbols import iVar

    # Split leading ±1 / numeric coefficients from "real" factors.
    if isinstance(term, sp.Mul):
        factors = list(term.args)
    else:
        factors = [term]

    coeff = sp.S.One
    delta_factors = []
    indexed_factors = []
    sum_factors = []
    other_factors = []
    for f in factors:
        if isinstance(f, KroneckerDelta):
            delta_factors.append(f)
        elif isinstance(f, sp.Indexed):
            indexed_factors.append(f)
        elif isinstance(f, sp.Sum):
            sum_factors.append(f)
        elif isinstance(f, (sp.Integer, sp.Float, sp.Rational, sp.Number)):
            coeff = coeff * f
        else:
            other_factors.append(f)

    outer_name = _name_of(outer_idx)
    diff_name = _name_of(diff_idx)

    # --- Phase J1 shapes ---------------------------------------------

    # Constant identity / selection: KroneckerDelta(...) * numeric
    if (len(delta_factors) == 1
            and len(indexed_factors) == 0
            and len(sum_factors) == 0
            and not other_factors):
        d = delta_factors[0]
        result = _translate_kronecker_delta(
            d, outer_idx, diff_idx, n_outer, var_map, n_diff=n_diff)
        if result is not None:
            return coeff * result
        raise NotImplementedError(
            f"LoopEqn J translator: unsupported KroneckerDelta {d}"
        )

    # Constant 2-D Param entry: Indexed(Param, outer, diff) * numeric
    if (len(indexed_factors) == 1
            and len(delta_factors) == 0
            and len(sum_factors) == 0
            and not other_factors):
        return coeff * _translate_indexed_param(
            indexed_factors[0], outer_idx, diff_idx, var_map
        )

    # --- Phase J2 shapes ---------------------------------------------

    # RowScale / ColScale: two Indexed factors — one a 2-D Param
    # accessed as [outer, diff] (or [diff, outer]), the other a 1-D
    # Var accessed as [outer] (RowScale) or [diff] (ColScale).
    if (len(indexed_factors) == 2
            and len(delta_factors) == 0
            and len(sum_factors) == 0
            and not other_factors):
        param_idx, var_idx = _split_param_var_indexed(
            indexed_factors, var_map, outer_name, diff_name
        )
        if param_idx is not None and var_idx is not None:
            param_name = param_idx.base.name
            var_name = var_idx.base.name
            var_sol_obj = var_map[var_name]
            if not isinstance(var_sol_obj.symbol, iVar):
                raise NotImplementedError(
                    f"LoopEqn J translator: row/col scale with "
                    f"non-iVar {var_name!r}"
                )
            var_sym = var_sol_obj.symbol
            # Classify as RowScale or ColScale based on the Var's
            # sole index.
            var_index_name = _name_of(var_idx.indices[0])
            param_expr = Para(param_name, dim=2)
            if var_index_name == outer_name:
                return coeff * Mat_Mul(Diag(var_sym), param_expr)
            if var_index_name == diff_name:
                return coeff * Mat_Mul(param_expr, Diag(var_sym))
            raise NotImplementedError(
                f"LoopEqn J translator: Var {var_name}[{var_idx.indices[0]}] "
                f"— index must be outer or diff"
            )

    # DiagTermWithSum: KroneckerDelta(...) * Sum(...)
    # Direct diagonal KD(outer,diff)*Sum → Diag(Mat_Mul(...))
    # Indirect KD(diff,map[outer])*Sum → SelectMat * Diag(Sum_scalar)
    #   — but the Sum is a scalar per outer iteration, so the
    #   overall expression stays a selection matrix scaled per-row.
    #   For now we only handle the direct diagonal form; the indirect
    #   KD * Sum is left to Phase J3 (still correctly sparse via the
    #   Sum-KD sparsity analyzer).
    if (len(delta_factors) == 1
            and len(sum_factors) >= 1
            and len(indexed_factors) == 0
            and not other_factors):
        d = delta_factors[0]
        d_names = {_name_of(d.args[0]), _name_of(d.args[1])}
        if d_names == {outer_name, diff_name} and len(sum_factors) == 1:
            inner_mm = _sum_to_matmul(
                sum_factors[0], outer_idx, var_map
            )
            return coeff * Diag(inner_mm)
        # Indirect KD with identity map (``KD(diff, map[outer])`` where
        # ``map == arange(n_outer)`` and ``n_outer == n_diff``) behaves
        # exactly like the direct KD — collapse to ``Diag(inner_mm)``.
        # Non-identity indirect KD falls through to Phase J3.
        if len(sum_factors) == 1:
            identity_indirect = _indirect_kd_is_identity(
                d, outer_idx, diff_idx, n_outer, var_map, n_diff,
            )
            if identity_indirect:
                inner_mm = _sum_to_matmul(
                    sum_factors[0], outer_idx, var_map
                )
                return coeff * Diag(inner_mm)
        raise NotImplementedError(
            f"LoopEqn J translator: DiagTerm with unsupported "
            f"KroneckerDelta {d}"
        )

    # Pattern 4 (#133): Sum-KD → sliced Param.
    # ``Sum(f(outer, dummy) * KD(diff, map[dummy]), (dummy, ...))`` collapses
    # via ``dummy := inv_map[diff]`` to ``f(outer, inv_map[diff])``. When
    # ``map`` is the identity, the result is ``f(outer, diff)`` — a bare
    # 2-D Param access that ``_translate_indexed_param`` already handles.
    if (len(sum_factors) == 1
            and len(delta_factors) == 0
            and len(indexed_factors) == 0
            and not other_factors):
        collapsed = _try_sum_kd_collapse(
            sum_factors[0], outer_idx, diff_idx,
            n_outer, var_map, n_diff,
        )
        if collapsed is not None:
            return coeff * collapsed

    # No shape matched — Phase J3 territory.
    raise NotImplementedError(
        f"LoopEqn J translator: cannot classify term {term!r}. "
        f"Factors: deltas={len(delta_factors)}, "
        f"indexed={len(indexed_factors)}, sums={len(sum_factors)}, "
        f"other={len(other_factors)}. Reserved for Phase J3 "
        f"(LoopEqn-native per-entry kernel emitter)."
    )


def _try_sum_kd_collapse(
    sum_node: sp.Sum,
    outer_idx: sp.Idx,
    diff_idx: sp.Idx,
    n_outer: int,
    var_map: Dict[str, object],
    n_diff: int,
):
    """Pattern 4 (#133): recognise

        Sum(f(outer, dummy) * KroneckerDelta(diff, map[dummy]),
            (dummy, lo, hi))

    and collapse it via ``dummy := inv_map[diff]``.

    **Identity map** (``map[p] == p``): the result is
    ``f(outer, diff)`` — typically a bare ``Indexed(Param[outer, diff])``
    that downstream translation lowers to ``Para(name, dim=2)``.

    **Non-identity map** (bijection ``[0, M) → subset of [0, n_diff)``):
    the body must simplify to a single 2-D ``Param[outer, dummy]``
    (after dropping the KD). The collapsed expression is emitted as

        ``Mat_Mul(Para(V, dim=2), _LoopJacSelectMat(map, M, n_diff))``

    — a constant sparse matrix product. The sparsity analyzer already
    reports the right nnz positions for the Sum-KD shape, and the
    ``is_constant_matrix_deri`` predicate routes the block to the
    constant-value fast path (Value0 baked into ``_data_`` at
    module-build time).

    Returns the translated Solverz expression, or ``None`` when the
    shape doesn't match.
    """
    from Solverz.equation.eqn import _LoopJacSelectMat
    from Solverz.equation.param import ParamBase
    from Solverz.sym_algebra.functions import Mat_Mul
    from Solverz.sym_algebra.symbols import Para

    if len(sum_node.args) != 2:
        return None
    body, sum_range = sum_node.args
    if len(sum_range) != 3:
        return None
    dummy = sum_range[0]
    dummy_name = _name_of(dummy)
    diff_name = _name_of(diff_idx)

    factors = list(body.args) if isinstance(body, sp.Mul) else [body]

    kd_factor = None
    map_obj = None
    for f in factors:
        if not isinstance(f, KroneckerDelta):
            continue
        a0, a1 = f.args
        for arg_d, arg_m in ((a0, a1), (a1, a0)):
            if not (isinstance(arg_d, sp.Idx) and _name_of(arg_d) == diff_name):
                continue
            if not isinstance(arg_m, sp.Indexed):
                continue
            if len(arg_m.indices) != 1:
                continue
            inner = arg_m.indices[0]
            if not (isinstance(inner, sp.Idx) and _name_of(inner) == dummy_name):
                continue
            base_name = arg_m.base.name
            mobj = var_map.get(base_name)
            if not (isinstance(mobj, ParamBase) and mobj.dim == 1):
                continue
            if kd_factor is not None:
                return None  # two KDs — ambiguous, skip
            kd_factor = f
            map_obj = mobj
    if kd_factor is None:
        return None

    map_v = np.asarray(map_obj.v, dtype=np.int64).reshape(-1)
    M = len(map_v)
    nd = n_diff if n_diff > 0 else (int(map_v.max()) + 1 if M else 0)
    if nd <= 0 or M == 0:
        return None
    # map must be an injection into [0, n_diff) — otherwise two dummies
    # would collide at the same diff column and the collapse would
    # double-count.
    if int(map_v.min()) < 0 or int(map_v.max()) >= nd:
        return None
    if len(set(map_v.tolist())) != M:
        return None

    is_identity = (M == nd and np.array_equal(map_v, np.arange(nd)))

    # Identity map: substitute dummy := diff and let the downstream
    # translator lower the resulting bare Indexed(Param, outer, diff).
    if is_identity:
        other_factors = [f for f in factors if f is not kd_factor]
        new_body = sp.Mul(*other_factors) if other_factors else sp.S.One
        substituted = new_body.subs(dummy, diff_idx)
        if not isinstance(substituted, sp.Indexed):
            return None
        if substituted.base.name not in var_map:
            return None
        sol_obj = var_map[substituted.base.name]
        if not (isinstance(sol_obj, ParamBase) and sol_obj.dim == 2):
            return None
        val_shape = sol_obj.value.shape if sol_obj.value is not None else None
        if val_shape is None or len(val_shape) != 2:
            return None
        if len(substituted.indices) != 2:
            return None
        row_idx, col_idx = substituted.indices
        # Column axis must be the bare diff index.
        if not (isinstance(col_idx, sp.Idx)
                and _name_of(col_idx) == diff_name):
            return None
        if val_shape[1] != nd:
            return None
        # Resolve the row axis to a concrete (n_outer,) gather map.
        row_select = _resolve_row_gather(
            row_idx, outer_idx, n_outer, var_map, val_shape[0],
        )
        if row_select is None:
            return None
        row_map_v, row_is_identity = row_select
        para_expr = Para(substituted.base.name, dim=2)
        if row_is_identity and val_shape[0] == n_outer:
            return para_expr
        select_row = _LoopJacSelectMat(
            sp.Tuple(*[sp.Integer(int(r)) for r in row_map_v]),
            sp.Integer(n_outer),
            sp.Integer(val_shape[0]),
        )
        return Mat_Mul(select_row, para_expr)

    # Non-identity map: require that the body (minus the KD) is a single
    # Indexed(V, outer, dummy) 2-D-Param access. Emit
    # ``Mat_Mul(Para(V, dim=2), _LoopJacSelectMat(map, M, n_diff))``.
    other_factors = [f for f in factors if f is not kd_factor]
    if len(other_factors) != 1:
        return None
    indexed_v = other_factors[0]
    if not isinstance(indexed_v, sp.Indexed):
        return None
    if len(indexed_v.indices) != 2:
        return None
    base_name = indexed_v.base.name
    if base_name not in var_map:
        return None
    v_obj = var_map[base_name]
    if not (isinstance(v_obj, ParamBase) and v_obj.dim == 2):
        return None
    # V's indices must be {outer, dummy} in either order so that the
    # matrix-side Mat_Mul composition picks up the right axes.
    outer_name = _name_of(outer_idx)
    idx_names = {_name_of(ix) for ix in indexed_v.indices}
    if idx_names != {outer_name, dummy_name}:
        return None
    val_shape = v_obj.value.shape if v_obj.value is not None else None
    if val_shape is None or len(val_shape) != 2:
        return None
    # V must be (n_outer, M): n_outer rows, one column per dummy entry.
    if val_shape[0] != n_outer or val_shape[1] != M:
        return None

    select_mat = _LoopJacSelectMat(
        sp.Tuple(*[sp.Integer(int(c)) for c in map_v]),
        sp.Integer(M),
        sp.Integer(nd),
    )
    return Mat_Mul(Para(base_name, dim=2), select_mat)


def _indirect_kd_is_identity(
    d: KroneckerDelta,
    outer_idx: sp.Idx,
    diff_idx: sp.Idx,
    n_outer: int,
    var_map: Dict[str, object],
    n_diff: int,
) -> bool:
    """True when ``d`` has the form ``KD(diff, map[outer])`` (or the
    symmetric ``KD(map[outer], diff)``) with ``map`` a 1-D integer
    ``Param`` of length ``n_outer`` whose values are ``[0, 1, …,
    n_outer-1]`` and ``n_outer == n_diff``.

    In that configuration the indirect delta is numerically identical
    to the direct ``KD(outer, diff)``, so any pattern built on the
    direct form (e.g. Phase J2 DiagTerm) applies unchanged.
    """
    from Solverz.equation.param import ParamBase

    outer_name = _name_of(outer_idx)
    diff_name = _name_of(diff_idx)
    a0, a1 = d.args
    for arg_d, arg_m in ((a0, a1), (a1, a0)):
        if not (isinstance(arg_d, sp.Idx) and _name_of(arg_d) == diff_name):
            continue
        if not isinstance(arg_m, sp.Indexed):
            continue
        if len(arg_m.indices) != 1:
            continue
        inner = arg_m.indices[0]
        if not (isinstance(inner, sp.Idx) and _name_of(inner) == outer_name):
            continue
        map_name = arg_m.base.name
        map_obj = var_map.get(map_name)
        if not (isinstance(map_obj, ParamBase) and map_obj.dim == 1):
            continue
        map_v = np.asarray(map_obj.v, dtype=np.int64).reshape(-1)
        if len(map_v) != n_outer:
            continue
        if n_diff != n_outer:
            continue
        if np.array_equal(map_v, np.arange(n_outer)):
            return True
    return False


def _resolve_row_gather(
    row_idx: sp.Expr,
    outer_idx: sp.Idx,
    n_outer: int,
    var_map: Dict[str, object],
    source_row_count: int,
):
    """Return ``(row_map_1d, is_identity)`` describing how the outer
    axis gathers rows of the source 2-D Param, or ``None`` when the
    row-index shape isn't recognised.

    Three supported cases:

    1. ``row_idx == outer_idx`` — the LoopEqn's outer iterates rows
       ``0..n_outer-1``. ``row_map = np.arange(n_outer)`` and
       ``is_identity = (n_outer == source_row_count)``.

    2. ``row_idx == Indexed(row_map_param, outer_idx)`` — row axis is
       gathered via a 1-D integer Param of length ``n_outer``; values
       must lie in ``[0, source_row_count)``.

    3. ``row_idx == Integer(k)`` with ``0 <= k < source_row_count`` —
       a constant row slice. Treated as ``row_map = [k] * n_outer``
       (every outer iteration picks the same row). Rare; covered for
       completeness.
    """
    from Solverz.equation.param import ParamBase

    outer_name = _name_of(outer_idx)

    if isinstance(row_idx, sp.Idx) and _name_of(row_idx) == outer_name:
        row_map = np.arange(n_outer, dtype=np.int64)
        is_identity = (n_outer == source_row_count)
        return row_map, is_identity

    if isinstance(row_idx, sp.Indexed):
        if len(row_idx.indices) != 1:
            return None
        inner = row_idx.indices[0]
        if not (isinstance(inner, sp.Idx) and _name_of(inner) == outer_name):
            return None
        map_name = row_idx.base.name
        map_obj = var_map.get(map_name)
        if not (isinstance(map_obj, ParamBase) and map_obj.dim == 1):
            return None
        row_map = np.asarray(map_obj.v, dtype=np.int64).reshape(-1)
        if len(row_map) != n_outer:
            return None
        if int(row_map.min()) < 0 or int(row_map.max()) >= source_row_count:
            return None
        is_identity = np.array_equal(row_map, np.arange(source_row_count))
        return row_map, is_identity

    if isinstance(row_idx, (sp.Integer, int)):
        k = int(row_idx)
        if k < 0 or k >= source_row_count:
            return None
        row_map = np.full(n_outer, k, dtype=np.int64)
        return row_map, False

    return None


def _split_param_var_indexed(indexed_factors,
                              var_map: Dict[str, object],
                              outer_name: str,
                              diff_name: str):
    """Given two ``Indexed`` factors, identify which is the 2-D
    ``Param`` accessed as ``[outer, diff]`` (or ``[diff, outer]``)
    and which is the 1-D ``Var`` / ``Param`` accessed as a single
    index. Returns ``(param_idx, var_idx)`` or ``(None, None)`` if
    neither factor is a clean 2-D Param.
    """
    from Solverz.equation.param import ParamBase
    from Solverz.variable.ssymbol import Var

    if len(indexed_factors) != 2:
        return None, None

    def is_2d_param(idx_node):
        sol = var_map.get(idx_node.base.name)
        return (isinstance(sol, ParamBase)
                and sol.dim == 2
                and len(idx_node.indices) == 2)

    def indices_match_outer_diff(idx_node):
        names = {_name_of(ix) for ix in idx_node.indices}
        return names == {outer_name, diff_name}

    def is_1d_var(idx_node):
        sol = var_map.get(idx_node.base.name)
        return (isinstance(sol, Var)
                and len(idx_node.indices) == 1)

    a, b = indexed_factors
    if (is_2d_param(a) and indices_match_outer_diff(a)
            and is_1d_var(b)):
        return a, b
    if (is_2d_param(b) and indices_match_outer_diff(b)
            and is_1d_var(a)):
        return b, a
    return None, None


def _sum_to_matmul(sum_node: sp.Sum,
                    outer_idx: sp.Idx,
                    var_map: Dict[str, object]) -> sp.Expr:
    """Recognise ``Sum(Param[outer, dummy] * Var[dummy], (dummy, 0,
    n-1))`` and translate to ``Mat_Mul(Para(param), iVar(var))``.

    This is the matmul-fingerprint used by the DiagTerm classifier.
    The mutable-matrix analyzer downstream will pre-compute the
    ``Mat_Mul`` as a dense vector in the ``J_`` wrapper via
    ``SolCF.csc_matvec`` (if the Param is sparse) and pass it to
    ``inner_J`` as a ``diag_term`` scaling vector.

    Limitations:
    - Exactly one 2-D ``Param`` factor indexed ``[outer, dummy]`` or
      ``[dummy, outer]``
    - Exactly one 1-D ``Var`` factor indexed ``[dummy]``
    - Any other factors must be pure numeric coefficients (which
      get multiplied into the final Mat_Mul)
    """
    from Solverz.equation.param import ParamBase
    from Solverz.sym_algebra.functions import Mat_Mul
    from Solverz.sym_algebra.symbols import iVar
    from Solverz.variable.ssymbol import Var

    if len(sum_node.args) != 2:
        raise NotImplementedError(
            f"_sum_to_matmul: unsupported Sum shape {sum_node}"
        )
    body = sum_node.args[0]
    dummy, _lo, _hi = sum_node.args[1]
    dummy_name = _name_of(dummy)
    outer_name = _name_of(outer_idx)

    if isinstance(body, sp.Mul):
        factors = list(body.args)
    else:
        factors = [body]

    coeff = sp.S.One
    param_node = None
    var_node = None
    for f in factors:
        if isinstance(f, (sp.Integer, sp.Float, sp.Rational, sp.Number)):
            coeff = coeff * f
            continue
        if isinstance(f, sp.Indexed):
            base_name = f.base.name
            sol = var_map.get(base_name)
            if sol is None:
                raise NotImplementedError(
                    f"_sum_to_matmul: Indexed {base_name!r} has no "
                    f"var_map entry"
                )
            if (isinstance(sol, ParamBase) and sol.dim == 2
                    and len(f.indices) == 2):
                idx_names = {_name_of(ix) for ix in f.indices}
                if idx_names != {outer_name, dummy_name}:
                    raise NotImplementedError(
                        f"_sum_to_matmul: 2-D Param {base_name!r} "
                        f"indexed {f.indices} — expected "
                        f"[{outer_name}, {dummy_name}]"
                    )
                if param_node is not None:
                    raise NotImplementedError(
                        f"_sum_to_matmul: multiple 2-D Params in "
                        f"Sum body — not supported"
                    )
                param_node = f
                continue
            if (isinstance(sol, Var) and len(f.indices) == 1
                    and _name_of(f.indices[0]) == dummy_name):
                if var_node is not None:
                    raise NotImplementedError(
                        f"_sum_to_matmul: multiple 1-D Vars at "
                        f"dummy — not supported"
                    )
                var_node = f
                continue
            raise NotImplementedError(
                f"_sum_to_matmul: unexpected Indexed factor {f}"
            )
        raise NotImplementedError(
            f"_sum_to_matmul: non-Indexed / non-numeric factor {f} "
            f"in Sum body — reserved for Phase J3"
        )

    if param_node is None or var_node is None:
        raise NotImplementedError(
            f"_sum_to_matmul: expected one 2-D Param and one 1-D "
            f"Var in Sum body, got param={param_node}, var={var_node}"
        )

    var_sol_obj = var_map[var_node.base.name]
    return coeff * Mat_Mul(
        Para(param_node.base.name, dim=2),
        var_sol_obj.symbol,
    )


def _probe_kron_entries(expr: sp.Expr,
                       outer_idx: sp.Idx,
                       n_outer: int,
                       n_diff: int) -> list | None:
    """Numerically evaluate a KroneckerDelta argument at each outer
    index value.

    Used when the argument contains ``outer_idx`` in a form too
    complex for ``_classify_axis`` — e.g.
    ``off + k - 1 + 2 * KroneckerDelta(k, 0)``.  Substitutes
    ``outer_idx → Integer(i)`` for each ``i`` in ``[0, n_outer)``
    and checks whether the result reduces to a concrete integer
    column index in ``[0, n_diff)``.

    Returns a list of ``(row, col)`` tuples, or ``None`` if any
    evaluation fails to reduce.
    """
    entries: list = []
    for i in range(n_outer):
        try:
            val = expr.subs(outer_idx, sp.Integer(i))
            col = int(val)
        except (TypeError, ValueError):
            return None
        if 0 <= col < n_diff:
            entries.append((i, col))
    return entries


def compute_loop_jac_sparsity(canonical: sp.Expr,
                                outer_idx: sp.Idx,
                                diff_idx: sp.Idx,
                                var_map: Dict[str, object],
                                n_outer: int,
                                n_diff: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the **structural** sparsity pattern of a LoopEqn
    Jacobian block from its canonicalized derivative expression.

    Walks the canonical ``Add`` terms and, for each term,
    identifies which ``(row, col)`` positions in the ``n_outer ×
    n_diff`` block can be structurally non-zero:

    - **Diagonal**: a term containing ``KroneckerDelta(outer,
      diff)`` (possibly multiplied by other factors) contributes
      only to the diagonal positions ``(i, i)`` for each ``i``
      where both indices are valid.
    - **Param sparsity**: a term containing
      ``Indexed(Param, outer, diff)`` or
      ``Indexed(Param, diff, outer)`` with ``Param`` a 2-D
      ``ParamBase`` contributes at the Param's stored nnz
      positions (from ``param.v.tocoo()``). Transposed indexing
      swaps row/col appropriately. Dense Params contribute the
      full ``n_outer × n_diff`` block.
    - **Dense fallback**: any other term depending on both
      ``outer_idx`` and ``diff_idx`` in a shape the analyzer
      can't classify forces the block back to the full dense
      pattern (conservatively safe).

    The returned ``(row_arr, col_arr)`` is the **sorted union** of
    all term patterns in row-major order, so downstream
    ``JacBlock.ParseSp`` / ``csc_array`` conversion sees a
    deterministic COO layout.

    Why this matters
    ----------------
    Without proper sparsity parsing, a LoopEqnDiff's Value0 ends
    up as a ``csc_array(dense_ndarray)`` with all ``n_outer ×
    n_diff`` entries marked non-zero — even when the true
    structure is e.g. diagonal (n entries) or follows a sparse
    network-admittance pattern (~5n entries for a power grid).
    The global Jacobian inherits this false density, ``inner_J``
    writes n² entries per iteration, and the Newton step's
    ``scipy.sparse.spsolve`` runs on a ``density = 1.0`` matrix
    — defeating every sparse-code optimization downstream.
    """
    from Solverz.equation.param import ParamBase

    outer_name = _name_of(outer_idx)
    diff_name = _name_of(diff_idx)

    def _classify_axis(expr):
        """Return ``('outer', None)`` if ``expr`` is the outer Idx,
        ``('diff', None)`` if ``expr`` is the diff Idx,
        ``('indirect_outer', map_name)`` if ``expr`` is
        ``Indexed(map_param, outer_idx)`` where ``map_param`` is a
        1-D int ``Param``, or ``('outer_shifted', shift)`` if
        ``expr`` is ``outer_idx + integer_constant`` (e.g.
        ``off + k + 1``). Otherwise ``(None, None)`` — the axis is
        not structurally classifiable and the term will fall through
        to the dense fallback or the numerical probing path.
        """
        if isinstance(expr, sp.Idx):
            if _name_of(expr) == outer_name:
                return 'outer', None
            if _name_of(expr) == diff_name:
                return 'diff', None
            return None, None
        if isinstance(expr, sp.Indexed):
            if len(expr.indices) != 1:
                return None, None
            inner = expr.indices[0]
            if not isinstance(inner, sp.Idx) or _name_of(inner) != outer_name:
                return None, None
            map_name = expr.base.name
            map_obj = var_map.get(map_name)
            if not isinstance(map_obj, ParamBase):
                return None, None
            if getattr(map_obj, 'dim', None) != 1:
                return None, None
            return 'indirect_outer', map_name
        if isinstance(expr, sp.Add):
            has_outer = False
            shift = 0
            all_simple = True
            for arg in expr.args:
                if (isinstance(arg, sp.Idx)
                        and _name_of(arg) == outer_name):
                    if has_outer:
                        all_simple = False
                        break
                    has_outer = True
                elif isinstance(arg, (sp.Integer, sp.Rational,
                                      sp.Number)):
                    shift += int(arg)
                else:
                    all_simple = False
                    break
            if has_outer and all_simple:
                return 'outer_shifted', shift
        return None, None

    def _map_values(map_name):
        """Return the materialised 1-D int array for an indirect
        outer row map. Assumes the caller has already validated
        dim=1 and ParamBase via ``_classify_axis``.
        """
        return np.asarray(var_map[map_name].v, dtype=np.int64).reshape(-1)

    # Each classified term contributes one or more (row, col)
    # pattern fragments; we accumulate them here and union at
    # the end.
    all_positions: set = set()
    has_dense_fallback = False

    if isinstance(canonical, sp.Add):
        terms = list(canonical.args)
    else:
        terms = [canonical]

    for term in terms:
        factors = list(term.args) if isinstance(term, sp.Mul) else [term]

        # Look for a top-level KroneckerDelta factor whose two
        # arguments classify as {outer, diff} or
        # {indirect_outer, diff}. The direct ``δ(outer, diff)``
        # form yields a plain diagonal; the indirect
        # ``δ(row_map[outer], diff)`` form yields entries at
        # ``(i, row_map.v[i])`` — exactly the positions where
        # ``_sz_loop_dk == row_map.v[i]``.
        has_diag_kron = False
        has_indirect_diag_kron = False
        indirect_diag_map: str = ''
        has_shifted_diag_kron = False
        shifted_diag_shift: int = 0
        has_probed_kron = False
        probed_kron_entries: list = []
        for f in factors:
            if isinstance(f, KroneckerDelta):
                a0 = f.args[0]
                a1 = f.args[1]
                kind0, map0 = _classify_axis(a0)
                kind1, map1 = _classify_axis(a1)
                kinds = {kind0, kind1}
                if kinds == {'outer', 'diff'}:
                    has_diag_kron = True
                    break
                if kinds == {'indirect_outer', 'diff'}:
                    has_indirect_diag_kron = True
                    indirect_diag_map = map0 if kind0 == 'indirect_outer' else map1
                    break
                if 'outer_shifted' in kinds and 'diff' in kinds:
                    has_shifted_diag_kron = True
                    shifted_diag_shift = (
                        map0 if kind0 == 'outer_shifted' else map1
                    )
                    break
                diff_side = (0 if kind0 == 'diff'
                             else 1 if kind1 == 'diff'
                             else -1)
                if diff_side >= 0:
                    other = a1 if diff_side == 0 else a0
                    other_kind = kind1 if diff_side == 0 else kind0
                    if (other_kind is None
                            and outer_idx in other.free_symbols):
                        entries = _probe_kron_entries(
                            other, outer_idx, n_outer, n_diff)
                        if entries is not None:
                            has_probed_kron = True
                            probed_kron_entries = entries
                            break

        # Look for ``Indexed(Param, axis0, axis1)`` where
        # ``{axis0_kind, axis1_kind}`` is one of
        # ``{outer, diff}`` or ``{indirect_outer, diff}``. The
        # indirect-outer case uses the row-map to select a SUBSET
        # of the Param's rows and emits entries at
        # ``(i, col)`` for every col where
        # ``Param[row_map.v[i], col]`` is structurally nonzero.
        param_hits = []
        for idx_node in term.atoms(sp.Indexed):
            base_name = idx_node.base.name
            sol = var_map.get(base_name)
            if not isinstance(sol, ParamBase):
                continue
            if sol.dim != 2 or len(idx_node.indices) != 2:
                continue
            ax0 = idx_node.indices[0]
            ax1 = idx_node.indices[1]
            kind0, map0 = _classify_axis(ax0)
            kind1, map1 = _classify_axis(ax1)
            if kind0 is None or kind1 is None:
                continue
            kinds = {kind0, kind1}
            if kinds == {'outer', 'diff'}:
                val = sol.v
                if hasattr(val, 'tocoo'):
                    coo = val.tocoo()
                    r = np.asarray(coo.row, dtype=np.int64)
                    c = np.asarray(coo.col, dtype=np.int64)
                else:
                    arr = np.asarray(val)
                    r, c = np.nonzero(np.ones_like(arr, dtype=bool))
                    r = r.astype(np.int64)
                    c = c.astype(np.int64)
                # ``[diff, outer]`` transposes row/col.
                if kind0 == 'diff':
                    r, c = c, r
                param_hits.append((r, c))
            elif kinds == {'indirect_outer', 'diff'}:
                # Indirect outer: row is ``map_name[outer]`` for
                # some known row map. Look up the Param's stored
                # columns for each ``row = map.v[i]`` and emit
                # ``(i, col)``.
                if kind0 == 'indirect_outer':
                    map_name = map0
                    swapped = False
                else:
                    map_name = map1
                    swapped = True
                row_map = _map_values(map_name)
                val = sol.v
                if hasattr(val, 'tocsr'):
                    csr = val.tocsr()
                    sub_rows_i = []
                    sub_cols = []
                    for i_outer, real_row in enumerate(row_map):
                        start = int(csr.indptr[real_row])
                        stop = int(csr.indptr[real_row + 1])
                        cols_in_row = csr.indices[start:stop]
                        for cc in cols_in_row:
                            sub_rows_i.append(i_outer)
                            sub_cols.append(int(cc))
                    r = np.array(sub_rows_i, dtype=np.int64)
                    c = np.array(sub_cols, dtype=np.int64)
                else:
                    # Dense 2-D Param: every column of each selected
                    # row is a candidate structural nonzero. This is
                    # still better than the full ``n_outer * n_diff``
                    # dense fallback because we only include the
                    # subset rows the outer index actually visits.
                    arr = np.asarray(val)
                    n_cols_v = arr.shape[1]
                    rows_i = np.repeat(
                        np.arange(len(row_map), dtype=np.int64), n_cols_v)
                    cols_i = np.tile(
                        np.arange(n_cols_v, dtype=np.int64), len(row_map))
                    r, c = rows_i, cols_i
                if swapped:
                    r, c = c, r
                param_hits.append((r, c))
            else:
                continue

        if has_diag_kron:
            n = min(n_outer, n_diff)
            diag = np.arange(n, dtype=np.int64)
            all_positions.update(zip(diag.tolist(), diag.tolist()))

        if has_indirect_diag_kron:
            row_map = _map_values(indirect_diag_map)
            for i_outer, real_col in enumerate(row_map):
                if 0 <= int(real_col) < n_diff:
                    all_positions.add((int(i_outer), int(real_col)))

        if has_shifted_diag_kron:
            for i in range(n_outer):
                col = i + shifted_diag_shift
                if 0 <= col < n_diff:
                    all_positions.add((i, col))

        if has_probed_kron:
            all_positions.update(probed_kron_entries)

        if param_hits:
            for r, c in param_hits:
                all_positions.update(zip(r.tolist(), c.tolist()))

        # --- Sum-KD pattern: a factor is ``Sum(... * KD(diff,
        # map[dummy]) * Param2D[row_expr, dummy] * ..., (dummy,
        # lo, hi))``.  The KD collapses the sum to specific
        # columns; a 2-D Param further restricts which dummy
        # values contribute per row.
        has_sum_kd = False
        for f in term.atoms(sp.Sum):
            if not isinstance(f, sp.Sum):
                continue
            sum_body = f.args[0]
            sum_lim = f.args[1:]
            if len(sum_lim) != 1:
                continue
            sdummy, slo, shi = sum_lim[0]
            s_range = int(shi) - int(slo) + 1
            # Find KD(diff_idx, expr_of_dummy) inside the Sum body
            sbody_factors = (list(sum_body.args)
                             if isinstance(sum_body, sp.Mul)
                             else [sum_body])
            col_map_name = None
            for sf in sbody_factors:
                if not isinstance(sf, KroneckerDelta):
                    continue
                sa0, sa1 = sf.args
                # One side must be the diff_idx, the other an
                # Indexed(param, dummy).
                for sarg_d, sarg_m in [(sa0, sa1), (sa1, sa0)]:
                    if not (isinstance(sarg_d, sp.Idx)
                            and _name_of(sarg_d) == diff_name):
                        continue
                    if not isinstance(sarg_m, sp.Indexed):
                        continue
                    if len(sarg_m.indices) != 1:
                        continue
                    sinner = sarg_m.indices[0]
                    if not (isinstance(sinner, sp.Idx)
                            and sinner == sdummy):
                        continue
                    mname = sarg_m.base.name
                    mobj = var_map.get(mname)
                    if isinstance(mobj, ParamBase) and mobj.dim == 1:
                        col_map_name = mname
                        break
                if col_map_name is not None:
                    break
            if col_map_name is None:
                continue
            # We found the column-map param.  Now look for a 2-D
            # Param with axes (outer_expr, dummy) inside the Sum
            # body to restrict which dummy values contribute per
            # row.
            col_map = _map_values(col_map_name)
            sparse_param = None
            sp_row_map_name = None
            for sf in sbody_factors:
                if not isinstance(sf, sp.Indexed):
                    continue
                sbase = sf.base.name
                sobj = var_map.get(sbase)
                if not isinstance(sobj, ParamBase):
                    continue
                if sobj.dim != 2 or len(sf.indices) != 2:
                    continue
                sax0, sax1 = sf.indices
                # One axis must be the Sum dummy, the other the
                # outer index (possibly indirect).
                if sax1 == sdummy:
                    kind_row, map_row = _classify_axis(sax0)
                elif sax0 == sdummy:
                    kind_row, map_row = _classify_axis(sax1)
                else:
                    continue
                if kind_row in ('outer', 'indirect_outer'):
                    sparse_param = sobj
                    sp_row_map_name = map_row  # None for direct
                    break
            # Build sparsity entries.
            if sparse_param is not None and hasattr(
                    sparse_param.v, 'tocsr'):
                csr = sparse_param.v.tocsr()
                row_map = (_map_values(sp_row_map_name)
                           if sp_row_map_name else
                           np.arange(n_outer, dtype=np.int64))
                for i_outer in range(n_outer):
                    real_row = int(row_map[i_outer])
                    start = int(csr.indptr[real_row])
                    stop = int(csr.indptr[real_row + 1])
                    for pos in range(start, stop):
                        pp = int(csr.indices[pos])
                        col = int(col_map[pp])
                        if 0 <= col < n_diff:
                            all_positions.add((i_outer, col))
            else:
                # No 2-D Param filter — every dummy value
                # contributes to every row.
                unique_cols = set(int(c) for c in col_map
                                  if 0 <= int(c) < n_diff)
                for i_outer in range(n_outer):
                    for col in unique_cols:
                        all_positions.add((i_outer, col))
            has_sum_kd = True
            # Don't break — union sparsity from all Sum-KD atoms

        if (not has_diag_kron
                and not has_indirect_diag_kron
                and not has_shifted_diag_kron
                and not has_probed_kron
                and not param_hits
                and not has_sum_kd):
            has_dense_fallback = True
            break

    def _dense_fallback():
        # Column-major dense pattern to match ``csc_array.tocoo()``
        # internal ordering (columns first, then rows within each
        # column). The downstream JacBlock extracts CooRow / CooCol
        # from ``Value0.tocoo()``, which scipy returns in this
        # same order; keeping the kernel output in the same order
        # lets the J_ wrapper write ``data[addr_slice] = kernel(...)``
        # with no fancy-indexing reorder.
        cols = np.repeat(np.arange(n_diff, dtype=np.int64), n_outer)
        rows = np.tile(np.arange(n_outer, dtype=np.int64), n_diff)
        return rows, cols

    if has_dense_fallback:
        import warnings
        warnings.warn(
            f"compute_loop_jac_sparsity: dense fallback triggered "
            f"for {n_outer}×{n_diff} block ({n_outer * n_diff} nnz). "
            f"Consider pre-computing index expressions as Params to "
            f"help the sparsity analyzer.",
            stacklevel=3)
        return _dense_fallback()

    if not all_positions:
        # Defensive: no terms produced structural positions. Fall
        # back to dense to guarantee correctness.
        return _dense_fallback()

    # Sort by (col, row) — column-major, matching csc_array's
    # internal storage so ``csc_array((data, (row, col)), shape)
    # .tocoo()`` returns the indices in the same order we pass
    # to the kernel builder.
    sorted_positions = sorted(all_positions, key=lambda p: (p[1], p[0]))
    row_arr = np.array([p[0] for p in sorted_positions], dtype=np.int64)
    col_arr = np.array([p[1] for p in sorted_positions], dtype=np.int64)
    return row_arr, col_arr


def build_loop_jac_kernel_source(func_name: str,
                                   canonical: sp.Expr,
                                   outer_idx: sp.Idx,
                                   diff_idx: sp.Idx,
                                   nnz: int,
                                   symbols_list,
                                   var_map: Dict[str, object],
                                   row_arr_param: str = '_sz_row_arr',
                                   col_arr_param: str = '_sz_col_arr') -> str:
    """Generate Python source for a **sparse** LoopEqn Jacobian
    block kernel.

    The kernel is a single ``for _sz_idx in range(nnz):`` loop that
    reads the current ``(i, k)`` position from pre-computed row /
    column index arrays, evaluates the canonicalized diff expression
    at that position, and writes the result to a flat
    ``data[_sz_idx]``. The output is a 1-D ``ndarray`` of length
    ``nnz`` whose layout matches the order the caller supplied in
    ``row_arr_param`` / ``col_arr_param``. Callers (``LoopEqnDiff``
    and the JIT ``print_J`` wrapper) are responsible for providing
    the row / col arrays in column-major (csc) order so the
    downstream ``Value0.tocoo()`` extraction aligns with the
    kernel's output.

    Per-element ``(i, k)`` values are translated through
    :func:`Solverz.equation.eqn._translate_loop_body_njit`, which
    handles ``sympy.Indexed``, ``sympy.Sum`` (inner for-loop with
    accumulator), ``KroneckerDelta`` (lowered to a Python
    conditional), and the Phase J2 function map
    (``sin``/``cos``/``Abs``/``Sign``/``heaviside``/etc).

    Parameters
    ----------
    func_name : str
        Name of the generated function, e.g. ``"_loop_jac_kernel_0"``.
    canonical : sympy.Expr
        Canonicalized Jacobian expression (output of
        :func:`canonicalize_kronecker`).
    outer_idx : sympy.Idx
        The LoopEqn's outer index — its name is used inside the
        body translator as the row variable (bound to
        ``row_arr_param[_sz_idx]`` at each iteration).
    diff_idx : sympy.Idx
        The fresh diff index — bound to ``col_arr_param[_sz_idx]``
        at each iteration.
    nnz : int
        Number of structurally non-zero positions in the block
        (length of the row / col arrays). Emitted as a literal in
        the ``range(nnz)`` loop.
    symbols_list : list of str
        Sorted Var/Param names that flow in as function arguments
        BEFORE the row / col arrays. The full signature is
        ``(<symbols>, <row_arr_param>, <col_arr_param>)``.
    var_map : dict
        IndexedBase name → Solverz Var/Param (passed through to
        ``_translate_loop_body_njit`` for the sparse-walker
        context).
    row_arr_param, col_arr_param : str
        Parameter names for the row / col index arrays inside the
        generated function. The caller can pick unique names to
        avoid collision in the module-level scope.

    Returns
    -------
    str
        Full Python source for the kernel function, including the
        ``def`` line and a trailing newline. Ready to ``exec`` (for
        the inline path) or to ``@njit(cache=True)``-decorate and
        paste into a module file (for the JIT path).
    """
    from Solverz.equation.eqn import _translate_loop_body_njit

    outer_name = _name_of(outer_idx)
    diff_name = _name_of(diff_idx)

    indent = '    '
    body_indent = indent * 2

    def _make_state():
        return {
            'acc_counter': 0,
            'prelude': [],
            'var_map': var_map,
            'outer_name': outer_name,
            'sparse_walker_ctx': None,
            'sparse_point_helpers': set(),
        }

    # Try to decompose the canonical Add into per-δ branches so the
    # generated kernel uses an if/elif chain instead of evaluating
    # every term. Each branch computes only the one matching stencil
    # derivative — critical for WENO/kt2 stencils where N terms
    # share the same loop but only 1 fires per (row, col) position.
    # Decompose the canonical Add into per-δ-condition branches.
    # Terms sharing the same KroneckerDelta condition are SUMMED so
    # an if/elif chain correctly accumulates all contributions at
    # each stencil position.
    from collections import OrderedDict
    branch_map: OrderedDict = OrderedDict()  # cond_key → [value_codes]
    fallback_terms: list = []
    has_sum = canonical.has(sp.Sum)
    if isinstance(canonical, sp.Add) and not has_sum:
        for term in canonical.args:
            if isinstance(term, sp.Mul):
                factors = list(term.args)
            else:
                factors = [term]
            deltas = [f for f in factors if isinstance(f, KroneckerDelta)]
            others = [f for f in factors if not isinstance(f, KroneckerDelta)]
            if deltas and others:
                st = _make_state()
                conds = []
                for d in deltas:
                    ac = _translate_loop_body_njit(d.args[0], st)
                    bc = _translate_loop_body_njit(d.args[1], st)
                    conds.append(f"{ac} == {bc}")
                cond_key = " and ".join(sorted(conds))
                val_expr = sp.Mul(*others) if len(others) > 1 else others[0]
                val_code = _translate_loop_body_njit(val_expr, st)
                branch_map.setdefault(cond_key, []).append(val_code)
            else:
                fallback_terms.append(term)
    branches: List[Tuple[str, str]] = [
        (cond, " + ".join(vals)) for cond, vals in branch_map.items()
    ]

    state = _make_state()
    use_branches = len(branches) >= 3 and not fallback_terms

    if not use_branches:
        body_expr = _translate_loop_body_njit(canonical, state)

    helper_sources: List[str] = []
    all_point_helpers = set()
    if use_branches:
        for st_key in [state] + [_make_state()]:
            all_point_helpers |= st_key.get('sparse_point_helpers', set())
    else:
        all_point_helpers = state.get('sparse_point_helpers', set())
    for walker_name in sorted(all_point_helpers):
        helper_sources.append(
            f"def _sz_csr_{walker_name}_point(row, col):\n"
            f"{indent}for _sz_pk in range("
            f"_sz_csr_{walker_name}_indptr[row], "
            f"_sz_csr_{walker_name}_indptr[row + 1]):\n"
            f"{indent * 2}if _sz_csr_{walker_name}_indices[_sz_pk] == col:\n"
            f"{indent * 3}return _sz_csr_{walker_name}_data[_sz_pk]\n"
            f"{indent}return 0.0\n"
        )

    arg_list = list(symbols_list) + [row_arr_param, col_arr_param]
    lines = [
        f"def {func_name}({', '.join(arg_list)}):",
        f"{indent}data = np.empty({nnz})",
        f"{indent}for _sz_idx in range({nnz}):",
        f"{body_indent}{outer_name} = {row_arr_param}[_sz_idx]",
        f"{body_indent}{diff_name} = {col_arr_param}[_sz_idx]",
    ]
    for stmt in state['prelude']:
        lines.append(f"{body_indent}{stmt}")

    if use_branches:
        for i, (cond, val) in enumerate(branches):
            kw = "if" if i == 0 else "elif"
            lines.append(f"{body_indent}{kw} {cond}:")
            lines.append(f"{body_indent}{indent}data[_sz_idx] = {val}")
        lines.append(f"{body_indent}else:")
        lines.append(f"{body_indent}{indent}data[_sz_idx] = 0.0")
    else:
        lines.append(f"{body_indent}data[_sz_idx] = {body_expr}")
    lines.append(f"{indent}return data")
    kernel_source = '\n'.join(lines) + '\n'

    return kernel_source, helper_sources


def _name_of(x) -> str:
    """Extract a comparable name from an Idx / Symbol / bare int.

    Used by the classifier to compare arguments of ``KroneckerDelta``
    against the LoopEqn's ``outer_index`` and the fresh diff index.
    Plain ``str(x)`` would work for ``Idx`` but fails for integer
    literals — we coerce everything to a string for robust set
    comparisons.
    """
    if hasattr(x, 'name'):
        return x.name
    return str(x)
