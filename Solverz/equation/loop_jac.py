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

from typing import Dict

import sympy as sp
from sympy.functions.special.tensor_functions import KroneckerDelta

from Solverz.sym_algebra.symbols import Para
from Solverz.utilities.type_checker import is_zero


def canonicalize_kronecker(expr: sp.Expr,
                            outer_idx: sp.Idx,
                            diff_idx: sp.Idx) -> sp.Expr:
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
    if isinstance(expr, sp.Add):
        return sp.Add(*(
            canonicalize_kronecker(a, outer_idx, diff_idx)
            for a in expr.args
        ))
    if isinstance(expr, sp.Mul):
        return sp.Mul(*(
            canonicalize_kronecker(a, outer_idx, diff_idx)
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
                                              outer_idx, diff_idx)
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
                              var_map: Dict[str, object]) -> sp.Expr:
    """Translate a *canonicalized* LoopEqn Jacobian expression back
    to a Solverz expression that ``FormJac`` and ``JacBlock`` can
    classify via ``is_constant_matrix_deri``.

    Phase J1 coverage
    -----------------
    Handles exactly the shapes the legacy ``_translate_loop_jac``
    was able to map:

    - ``KroneckerDelta(outer, diff)`` → ``_LoopJacEye(n_outer)``
    - ``Indexed(Base, outer, diff)`` with ``Base`` a 2-D ``Param``
      → ``Para(name, dim=2)``
    - ``Add`` / ``Mul`` thereof — reconstructed via ``sp.Add`` /
      ``sp.Mul`` in terms of the above

    Anything else raises ``NotImplementedError``: Phase J2 will
    route mutable shapes (``δ(outer) * Sum(...)``, row/col scale,
    bilinear entries, trig) through a new per-block kernel emitter
    that bypasses ``is_constant_matrix_deri`` entirely.
    """
    from Solverz.equation.eqn import _LoopJacEye
    from Solverz.equation.param import ParamBase

    outer_name = _name_of(outer_idx)
    diff_name = _name_of(diff_idx)

    def walk(e: sp.Expr) -> sp.Expr:
        if isinstance(e, (sp.Integer, sp.Float, sp.Rational, sp.Number)):
            return e

        if isinstance(e, KroneckerDelta):
            arg_names = {_name_of(a) for a in e.args}
            if arg_names == {outer_name, diff_name}:
                return _LoopJacEye(sp.Integer(n_outer))
            raise NotImplementedError(
                f"LoopEqn J translator (Phase J1): KroneckerDelta "
                f"with args {e.args} — only δ(outer, diff) is "
                f"supported at this phase"
            )

        if isinstance(e, sp.Indexed):
            base_name = e.base.name
            sol_obj = var_map.get(base_name)
            if sol_obj is None:
                raise NotImplementedError(
                    f"LoopEqn J translator: IndexedBase "
                    f"{base_name!r} has no var_map entry"
                )
            if isinstance(sol_obj, ParamBase) and sol_obj.dim == 2:
                if len(e.indices) != 2:
                    raise NotImplementedError(
                        f"LoopEqn J translator: expected 2 indices "
                        f"on {base_name!r}, got {e}"
                    )
                idx_names = {_name_of(ix) for ix in e.indices}
                if idx_names != {outer_name, diff_name}:
                    raise NotImplementedError(
                        f"LoopEqn J translator: 2-D Param "
                        f"{base_name!r} accessed with indices {e.indices} "
                        f"— Phase J1 only handles [outer, diff] "
                        f"(in either order)"
                    )
                return Para(sol_obj.name, dim=2)
            raise NotImplementedError(
                f"LoopEqn J translator: IndexedBase {base_name!r} of "
                f"{type(sol_obj).__name__} dim="
                f"{getattr(sol_obj, 'dim', None)} — reserved for "
                f"Phase J2"
            )

        if isinstance(e, sp.Add):
            return sp.Add(*(walk(a) for a in e.args))
        if isinstance(e, sp.Mul):
            return sp.Mul(*(walk(a) for a in e.args))

        if isinstance(e, sp.Sum):
            # A Sum that survived canonicalization is non-constant
            # by construction (no KroneckerDelta on its dummy) —
            # will be handled by the DiagTerm / BilinearEntry
            # paths in Phase J2.
            raise NotImplementedError(
                f"LoopEqn J translator: un-collapsed Sum {e} — "
                f"reserved for Phase J2"
            )

        if isinstance(e, sp.Idx):
            raise NotImplementedError(
                f"LoopEqn J translator: bare Idx {e!r} in "
                f"Jacobian expression"
            )
        if isinstance(e, sp.Symbol):
            raise NotImplementedError(
                f"LoopEqn J translator: unexpected bare Symbol {e!r}"
            )
        raise NotImplementedError(
            f"LoopEqn J translator: unsupported node "
            f"{type(e).__name__}: {e!r}"
        )

    return walk(expr)


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
