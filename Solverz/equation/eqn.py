from __future__ import annotations

from typing import Union, List, Dict, Callable

import numpy as np
import sympy as sp
from sympy import Symbol, Expr, latex, Derivative, sympify, Function
from sympy import lambdify as splambdify
from sympy.abc import t, x
from sympy.functions.special.tensor_functions import KroneckerDelta

from Solverz.sym_algebra.symbols import iVar, Para, IdxVar, idx, IdxPara, iAliasVar, IdxAliasVar
from Solverz.variable.ssymbol import Var
from Solverz.sym_algebra.functions import Mat_Mul, Slice
from Solverz.sym_algebra.matrix_calculus import MixedEquationDiff
from Solverz.num_api.module_parser import modules
from Solverz.variable.ssymbol import sSym2Sym
from Solverz.utilities.type_checker import is_zero


class Eqn:
    """
    The Equation object
    """

    def __init__(self,
                 name: str,
                 eqn):
        if not isinstance(name, str):
            raise ValueError("Equation name must be string!")
        self.name: str = name
        self.LHS = 0
        self.RHS = sympify(sSym2Sym(eqn))
        self.SYMBOLS: Dict[str, Symbol] = self.obtain_symbols()

        # if the eqn has Mat_Mul, then label it as mixed-matrix-vector equation
        if self.expr.has(Mat_Mul):
            self.mixed_matrix_vector = True
        else:
            self.mixed_matrix_vector = False

        self.NUM_EQN: Callable = self.lambdify()
        self.derivatives: Dict[str, EqnDiff] = dict()

    def obtain_symbols(self) -> Dict[str, Symbol]:
        temp_dict = dict()
        for symbol_ in list((self.LHS - self.RHS).free_symbols):
            if isinstance(symbol_, (iVar, Para, idx, iAliasVar)):
                temp_dict[symbol_.name] = symbol_
            elif isinstance(symbol_, (IdxVar, IdxPara, IdxAliasVar)):
                temp_dict[symbol_.name0] = symbol_.symbol0
                temp_dict.update(symbol_.SymInIndex)

        # to sort in lexicographic order
        sorted_dict = {key: temp_dict[key] for key in sorted(temp_dict)}
        return sorted_dict

    def lambdify(self) -> Callable:
        return splambdify(self.SYMBOLS.values(), self.RHS, modules)

    def eval(self, *args: Union[np.ndarray]) -> np.ndarray:
        return self.NUM_EQN(*args)

    def derive_derivative(self):
        """"""
        for symbol_ in list(self.RHS.free_symbols):
            # differentiate only to variables
            if isinstance(symbol_, IdxVar):  # if the equation contains Indexed variables
                idx_ = symbol_.index
                if self.mixed_matrix_vector:
                    diff = MixedEquationDiff(self.RHS, symbol_)
                else:
                    diff = self.RHS.diff(symbol_)
                if not is_zero(diff):
                    self.derivatives[symbol_.name] = EqnDiff(name=f'Diff {self.name} w.r.t. {symbol_.name}',
                                                             eqn=diff,
                                                             diff_var=symbol_,
                                                             var_idx=idx_.name if isinstance(idx_, idx) else idx_)
            elif isinstance(symbol_, iVar):
                if self.mixed_matrix_vector:
                    diff = MixedEquationDiff(self.RHS, symbol_)
                else:
                    diff = self.RHS.diff(symbol_)
                if not is_zero(diff):
                    self.derivatives[symbol_.name] = EqnDiff(name=f'Diff {self.name} w.r.t. {symbol_.name}',
                                                             eqn=diff,
                                                             diff_var=symbol_)

    @property
    def expr(self):
        return self.LHS - self.RHS

    def subs(self, *args, **kwargs):
        return self.RHS.subs(*args, **kwargs)

    def __repr__(self):
        # sympy objects' printing prefers __str__() to __repr__()
        return self.LHS.__str__() + r"=" + self.RHS.__str__()

    def _repr_latex_(self):
        """
        So that jupyter notebook can display latex equation of Eqn object.
        :return:
        """
        return r"$\displaystyle %s$" % (latex(self.LHS) + r"=" + latex(self.RHS))


class EqnDiff(Eqn):
    """
    To store the derivatives of equations W.R.T. variables
    """

    def __init__(self, name: str, eqn: Expr, diff_var: Symbol, var_idx=None):
        super().__init__(name, eqn)
        self.diff_var = diff_var
        self.diff_var_name = diff_var.name0 if isinstance(diff_var, IdxVar) else diff_var.name
        self.var_idx = var_idx  # df/dPi[i] then var_idx=i
        self.var_idx_func = None
        if self.var_idx is not None:
            if isinstance(self.var_idx, slice):
                temp = []
                if var_idx.start is not None:
                    temp.append(var_idx.start)
                if var_idx.stop is not None:
                    temp.append(var_idx.stop)
                if var_idx.step is not None:
                    temp.append(var_idx.step)
                self.var_idx_func = Eqn('To evaluate var_idx of variable' + self.diff_var.name, Slice(*temp))
            elif isinstance(self.var_idx, Expr):
                self.var_idx_func = Eqn('To evaluate var_idx of variable' + self.diff_var.name, self.var_idx)
        self.LHS = Derivative(Function('F'), diff_var)
        self.dim = -1
        self.v_type = ''


def Idx(name: str, n: int = None):
    """Create a sympy :class:`sympy.Idx` for :class:`LoopEqn` body
    construction — a thin shortcut so users don't need to
    ``import sympy as sp`` just to grab one index symbol.

    Parameters
    ----------
    name : str
        The index label.
    n : int, optional
        If given, the index is bounded to ``[0, n - 1]`` (sympy stores
        this as ``.lower`` / ``.upper``). :func:`Sum` and
        :class:`LoopEqn` can then pick up the range automatically:

        >>> i = Idx('i', 3)
        >>> Sum(m.G[i, j] * m.ux[j], j)  # no explicit range
        >>> m.eqn = LoopEqn('eqn', outer_index=i, body=body, model=m)
        # n_outer auto-inferred from i.upper - i.lower + 1 = 3

        Without ``n`` the result is an unbounded ``sp.Idx`` and
        :func:`Sum` / :class:`LoopEqn` require explicit range / n_outer.
    """
    if n is None:
        return sp.Idx(name)
    return sp.Idx(name, int(n))


def Sum(expr, dummy, n: int = None):
    """Range-aware shortcut for :class:`sympy.Sum` used by
    :class:`LoopEqn` bodies. Three call forms:

    1. ``Sum(expr, j, n)`` → ``sp.Sum(expr, (j, 0, n - 1))`` — the
       most common case. ``n`` is an int.
    2. ``Sum(expr, j)`` where ``j = Idx('j', n)`` is a bounded
       ``sp.Idx`` → ``sp.Sum(expr, (j, j.lower, j.upper))``.
    3. ``Sum(expr, (j, lo, hi))`` → passthrough to ``sp.Sum`` (legacy
       sympy tuple form).

    Raises ``ValueError`` if neither ``n`` nor a bounded ``Idx`` is
    available.
    """
    if isinstance(dummy, tuple):
        return sp.Sum(expr, dummy)
    if n is not None:
        return sp.Sum(expr, (dummy, 0, int(n) - 1))
    # Bounded Idx case — sympy exposes .lower / .upper on Idx nodes
    # whose second __new__ argument was an int.
    lower = getattr(dummy, 'lower', None)
    upper = getattr(dummy, 'upper', None)
    if lower is not None and upper is not None:
        return sp.Sum(expr, (dummy, lower, upper))
    raise ValueError(
        f"Sum(expr, {dummy!r}): need an explicit ``n`` (for the "
        f"shortcut form ``Sum(expr, j, n)``) or a bounded Idx "
        f"(created via ``Idx('j', n)`` so the range is carried on "
        f"the index itself)."
    )


def _rewrite_solverz_body(body, model):
    """Rewrite Solverz ``IdxVar``/``IdxPara`` nodes inside a LoopEqn
    body to their ``sympy.IndexedBase`` equivalents, and return the
    rewritten body together with an auto-built ``var_map``.

    The user-facing motivation is that writing

        body = m.ix_pin[i] - sp.Sum(m.G[i, j] * m.ux[j], (j, 0, n-1))

    is much nicer than

        ux_sym = sp.IndexedBase('ux')
        G_sym  = sp.IndexedBase('G')
        ...
        body = ix_pin_sym[i] - sp.Sum(G_sym[i, j] * ux_sym[j], (j, 0, n-1))

    plus a ``var_map`` dict mapping each string back to a Solverz
    symbol. Both of the above should produce identical downstream code.
    The second form is what the LoopEqn translator actually understands
    (because Solverz ``IdxVar``/``IdxPara`` don't interoperate with
    ``sp.Sum.doit()`` / ``.subs()`` / ``.diff()``), so we convert the
    first form to the second at construction time.

    Parameters
    ----------
    body : sympy.Expr
        The user's body, possibly containing Solverz ``IdxVar``/
        ``IdxPara`` references created via ``m.ux[j]`` / ``m.G[i, j]``.
        Already-sympy ``IndexedBase`` accesses pass through unchanged.
    model : Model
        The containing :class:`Model`, used to look up each referenced
        name via ``getattr(model, name)``. The returned object must be
        a Solverz ``Var`` or ``ParamBase``.

    Returns
    -------
    new_body : sympy.Expr
        The body with every Solverz ``IdxVar``/``IdxPara`` replaced by
        ``sp.IndexedBase(name0)[index_or_indices]``.
    var_map : Dict[str, object]
        Maps each referenced ``name0`` → Solverz ``Var``/``Param``
        object pulled from ``model``.
    """
    from Solverz.sym_algebra.symbols import IdxVar, IdxPara, iVar, Para
    from Solverz.variable.ssymbol import Var
    from Solverz.equation.param import ParamBase

    var_map: Dict[str, object] = {}

    def _resolve(name: str):
        """Look up a name in the model and validate its type.

        Caches into ``var_map`` so a given name resolves once.
        """
        if name in var_map:
            return var_map[name]
        sol_obj = getattr(model, name, None)
        if sol_obj is None:
            raise ValueError(
                f"LoopEqn body references symbol {name!r} but "
                f"``model`` has no attribute by that name. Make "
                f"sure you set ``m.{name} = Var/Param(...)`` "
                f"before constructing the LoopEqn."
            )
        if not isinstance(sol_obj, (Var, ParamBase)):
            raise TypeError(
                f"LoopEqn body references {name!r} which resolves "
                f"to a {type(sol_obj).__name__} on the model — "
                f"expected a Solverz ``Var`` or ``Param``."
            )
        var_map[name] = sol_obj
        return sol_obj

    def walk(expr):
        if isinstance(expr, (IdxVar, IdxPara)):
            name = expr.name0
            _resolve(name)  # populate var_map and validate
            # ``IdxSymBasic.index`` is either a single sympy Expr / Idx
            # (1-D access) or a tuple (2-D access). Recurse so any
            # inner Solverz ``IdxVar`` / ``IdxPara`` used as part of
            # the index (e.g. ``m.ux[m.node_idx[i]]`` for a subset-of-
            # nodes LoopEqn) also gets rewritten to ``IndexedBase``.
            raw_idx = expr.index
            if isinstance(raw_idx, tuple):
                new_idx = tuple(walk(x) for x in raw_idx)
            else:
                new_idx = walk(raw_idx)
            return sp.IndexedBase(name)[new_idx]

        # Bare Solverz ``iVar`` / ``Para`` (non-indexed) — e.g. a
        # scalar parameter like ``m.Cp`` referenced without
        # ``[i]``. The underlying symbol is still a sympy ``Symbol``
        # subclass, so we leave it in the tree as-is, but we MUST
        # register it in ``var_map`` so it flows into ``SYMBOLS`` and
        # the generated function receives it as a positional arg.
        # Without this, the body would reference an undefined local.
        if isinstance(expr, (iVar, Para)):
            name = expr.name
            _resolve(name)
            return expr

        if expr.is_Atom:
            return expr

        if isinstance(expr, sp.Sum):
            # Recurse into the summand; leave the (dummy, lo, hi) tuple
            # alone. sympy's Sum constructor accepts the walked body
            # plus the original limits.
            walked_body = walk(expr.args[0])
            return sp.Sum(walked_body, *expr.args[1:])

        if isinstance(expr, sp.Indexed):
            # Already a native sympy IndexedBase access; the user may
            # have mixed native and Solverz styles. Register the base
            # name in var_map if the model has a matching attribute.
            name = expr.base.name
            if hasattr(model, name) and name not in var_map:
                _resolve(name)
            # Recurse into the indices too — the user might build a
            # native-sympy outer with a Solverz-indexed inner (weird
            # but legal) and we want the same uniform rewrite.
            new_indices = tuple(walk(i) for i in expr.indices)
            return expr.base[new_indices if len(new_indices) > 1
                             else new_indices[0]]

        # Generic tree walk for Add / Mul / Pow / Function / ...
        new_args = [walk(arg) for arg in expr.args]
        return expr.func(*new_args)

    new_body = walk(body)
    if not var_map:
        raise ValueError(
            "LoopEqn: could not find any Solverz IdxVar / IdxPara / "
            "sympy IndexedBase references in the body — at least one "
            "Var / Param reference is required. If you meant to use "
            "the legacy sympy IndexedBase style, pass ``var_map`` "
            "instead of ``model``."
        )
    return new_body, var_map


class LoopEqn(Eqn):
    r"""
    Equation declared as a parameterised scalar template that prints to
    a Python ``for``-loop in the generated ``inner_F`` / ``inner_J``.

    The motivation: SolMuseum's network modules (``eps_network``,
    ``heat_network``, ``gas_network``) write per-bus / per-node
    equations using Python for-loops that emit ``2 * n_bus`` (or more)
    scalar :class:`Eqn` objects. Each scalar Eqn becomes its own
    ``inner_F<N>`` sub-function, and on the IES benchmark the resulting
    Numba LLVM compile time dominates the warm-up phase. ``LoopEqn``
    expresses the same thing as a SINGLE Eqn whose body is a scalar
    template parameterised by an outer index ``i``; the printer then
    emits ONE ``inner_F<N>`` containing a ``for i in range(n_outer):``
    loop. One sub-function instead of ``n_outer``.

    Substrate
    ---------
    Internally, the body is rewritten as a sympy expression built from
    :class:`sympy.IndexedBase`, :class:`sympy.Idx`, :class:`sympy.Sum`.
    Solverz's own :class:`IdxVar` / :class:`IdxPara` are opaque to
    sympy's ``subs`` / ``Sum.doit()`` / ``diff``, so we can't use them
    verbatim — LoopEqn walks the body once at construction and rewrites
    any Solverz ``IdxVar`` / ``IdxPara`` into the corresponding
    :class:`sympy.IndexedBase` form. The user is free to write either:

    1. **Native Solverz syntax** (recommended): ``m.G[i, j]`` and
       ``m.ux[j]`` directly inside the body, and pass ``model=m`` so
       LoopEqn can look up each name in the Model to resolve sparsity
       / dim / value info.
    2. **Explicit sympy syntax** (legacy): declare
       ``G_sym = sp.IndexedBase('G')`` etc., build the body out of
       those, and pass a ``var_map={'G': m.G, ...}`` dict mapping each
       IndexedBase name to a Solverz ``Var``/``Param``.

    Both styles produce the same internal representation. Native
    syntax is terser and avoids the bookkeeping trap of forgetting to
    add an entry to ``var_map`` when adding a new reference.

    Parameters
    ----------
    name : str
        Equation name.
    outer_index : sympy.Idx
        The outer loop index. Each row of the residual is the body
        evaluated at one value of ``outer_index``.
    n_outer : int
        Range of the outer loop. The residual block has size
        ``n_outer``.
    body : sympy.Expr
        A scalar sympy expression built either from
        ``sympy.IndexedBase`` (+ ``var_map``) or from Solverz
        ``IdxVar`` / ``IdxPara`` (+ ``model``). May involve the
        ``outer_index``, inner ``sympy.Sum`` over auxiliary ``Idx``
        symbols, and arithmetic.
    model : Model, optional
        When the body uses Solverz ``IdxVar`` / ``IdxPara``, pass the
        containing :class:`Model` so LoopEqn can resolve each
        ``name0`` back to a real ``Var`` / ``Param`` (needed for
        sparsity detection and CSR pre-computation). Mutually
        exclusive with ``var_map``.
    var_map : Dict[str, sSymBasic], optional
        Legacy API. Maps each ``sympy.IndexedBase`` name in ``body``
        to the corresponding Solverz ``Var`` / ``Param`` instance.
        Mutually exclusive with ``model``.

    Examples
    --------
    Native Solverz syntax (preferred)::

        m.ux = Var('ux', np.ones(nb))
        m.uy = Var('uy', np.zeros(nb))
        m.G = Param('G', csc_array(G_dense), dim=2, sparse=True)
        m.B = Param('B', csc_array(B_dense), dim=2, sparse=True)
        m.ix_pin = Param('ix_pin', ix_pin)

        i, j = sp.Idx('i'), sp.Idx('j')
        body = (m.ix_pin[i]
                - sp.Sum(m.G[i, j] * m.ux[j], (j, 0, nb - 1))
                + sp.Sum(m.B[i, j] * m.uy[j], (j, 0, nb - 1)))

        m.ix_inj = LoopEqn(
            'ix_inj',
            outer_index=i, n_outer=nb,
            body=body,
            model=m,
        )

    Legacy ``var_map`` API (still supported)::

        ux_sym = sp.IndexedBase('ux')
        G_sym = sp.IndexedBase('G')
        body = sp.Sum(G_sym[i, j] * ux_sym[j], (j, 0, nb - 1))
        m.ix_inj = LoopEqn(
            'ix_inj',
            outer_index=i, n_outer=nb,
            body=body,
            var_map={'ux': m.ux, 'G': m.G},
        )

    The generated ``inner_F`` for either form is a single function
    containing one Python ``for i in range(nb):`` loop, replacing the
    ``2 * nb`` scalar sub-functions the original loop pattern would
    emit.
    """

    def __init__(self,
                 name: str,
                 outer_index,
                 body,
                 n_outer: int = None,
                 var_map: Dict[str, object] = None,
                 model=None):
        if not isinstance(name, str):
            raise ValueError("Equation name must be string!")
        if not isinstance(outer_index, sp.Idx):
            raise TypeError(
                f"outer_index must be a sympy.Idx, got {type(outer_index).__name__}"
            )
        # Auto-infer n_outer from a bounded outer_index
        # (created via ``Idx('i', n)`` — sympy stores n-1 as upper).
        if n_outer is None:
            upper = getattr(outer_index, 'upper', None)
            lower = getattr(outer_index, 'lower', None)
            if upper is not None and lower is not None:
                n_outer = int(upper) - int(lower) + 1
            else:
                raise ValueError(
                    "LoopEqn: ``n_outer`` is required unless "
                    "``outer_index`` is a bounded Idx (create via "
                    "``Idx('i', n)``) so the range can be auto-"
                    "inferred from ``.lower`` / ``.upper``."
                )
        if not isinstance(n_outer, int) or n_outer < 1:
            raise ValueError(
                f"n_outer must be a positive int, got {n_outer!r}"
            )
        if var_map is not None and model is not None:
            raise ValueError(
                "LoopEqn: pass either ``var_map`` (legacy sympy "
                "IndexedBase style) or ``model`` (native Solverz "
                "IdxVar/IdxPara style), not both."
            )
        if var_map is None and model is None:
            raise ValueError(
                "LoopEqn: one of ``var_map`` or ``model`` is required "
                "so we can resolve each IndexedBase / IdxVar back to "
                "a Solverz Var/Param object."
            )

        # If ``model`` is given, walk the body, rewrite Solverz
        # ``IdxVar``/``IdxPara`` → ``sp.IndexedBase(name0)[index]``, and
        # auto-build the var_map by looking up each referenced name in
        # the model's ``__dict__``. The rewritten body is a pure
        # sympy-IndexedBase expression the rest of LoopEqn already
        # understands.
        if model is not None:
            body, var_map = _rewrite_solverz_body(body, model)

        if not isinstance(var_map, dict) or not var_map:
            raise ValueError("var_map must be a non-empty dict")

        # Bypass the parent ``Eqn.__init__`` because:
        #   * ``sympify(sSym2Sym(body))`` would happily store the body
        #     as RHS, but
        #   * ``self.lambdify()`` calls ``sympy.lambdify(..., body)``
        #     which raises ``PrintMethodNotImplementedError`` on
        #     ``Sum`` over ``Idx`` (sympy's NumPyPrinter cannot print
        #     ``Idx``). We provide a custom ``NUM_EQN`` instead.
        self.name = name
        self.LHS = sp.S.Zero
        self.RHS = body  # store as-is; the printer walks it
        self.outer_index = outer_index
        self.n_outer = n_outer
        self.body = body
        self.var_map = dict(var_map)

        # Build SYMBOLS — what the model assembler uses to discover
        # the Var / Param dependencies of this equation. Maps the
        # underlying-symbol-name to the Solverz internal Symbol
        # (iVar / Para). Sorted for determinism (matches what the
        # parent ``obtain_symbols`` does).
        from Solverz.variable.ssymbol import sSymBasic
        from Solverz.equation.param import ParamBase
        symbols_dict = {}
        for indexed_base_name, sol_obj in self.var_map.items():
            if isinstance(sol_obj, sSymBasic):
                symbols_dict[sol_obj.name] = sol_obj.symbol
            elif isinstance(sol_obj, ParamBase):
                # ParamBase exposes its sympy symbol as ``.sym`` (or
                # equivalent). We look up by name from the model.
                symbols_dict[sol_obj.name] = Para(sol_obj.name,
                                                   dim=sol_obj.dim)
            else:
                raise TypeError(
                    f"var_map entry {indexed_base_name!r} must be a "
                    f"Solverz Var or Param, got {type(sol_obj).__name__}"
                )
        self.SYMBOLS = {key: symbols_dict[key]
                        for key in sorted(symbols_dict)}

        # No Mat_Mul in the body — LoopEqn expresses everything via
        # sympy IndexedBase + Sum, NOT via Solverz Mat_Mul.
        self.mixed_matrix_vector = False

        # Pre-scan the body for sparse 2-D ``Param``s used as
        # ``M[outer_index, dummy]`` walkers inside a ``Sum``. For each
        # detected walker, freeze its CSR decomposition — the
        # translator emits code that walks these arrays instead of
        # iterating the full dense column range (which would (a) force
        # the user to densify the sparse Param and (b) iterate zeros
        # in the inner loop).
        self._sparse_csr = self._collect_sparse_walkers()

        # Custom numerical evaluator (bypasses sympy.lambdify).
        self.NUM_EQN = self._build_num_eqn()

        self.derivatives: Dict[str, EqnDiff] = dict()

    def _collect_sparse_walkers(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Scan the LoopEqn body for sparse 2-D ``Param``s in
        ``M[outer_index, dummy]`` positions inside a ``Sum``, and
        pre-compute the CSR decomposition for each.

        Returns a dict ``{param_name: {'data', 'indices', 'indptr'}}``
        of contiguous numpy arrays suitable for both Python closure
        capture (inline path) and numba-compatible module globals
        (JIT path).

        Enforces the Phase 1 constraints:

        - A sparse 2-D ``Param`` referenced anywhere in the body must
          be accessed *only* as ``M[outer_index, dummy]`` inside a
          ``Sum``. Any other access pattern raises
          ``NotImplementedError`` (we'd need different CSR plumbing to
          support partial-row / transposed / out-of-Sum accesses).
        - A single ``Sum`` body may contain at most ONE sparse 2-D
          ``Param``. Multiple sparse walkers sharing the same
          skeleton ("Case B") is Phase 2 work. Split into separate
          ``Sum``s, one per sparse walker.
        - The ``Sum``'s ``(dummy, lo, hi)`` range must span the full
          column extent of the walker (``lo == 0`` and
          ``hi == n_cols - 1``). Partial inner ranges with sparse
          walkers are not yet supported because the rewritten loop
          iterates stored non-zeros, not the user's range.
        """
        from Solverz.equation.param import ParamBase

        sparse_names = {
            nm for nm, obj in self.var_map.items()
            if isinstance(obj, ParamBase)
            and getattr(obj, 'sparse', False)
            and obj.dim == 2
        }
        if not sparse_names:
            return {}

        outer_name = self.outer_index.name
        walkers: Dict[str, Dict[str, np.ndarray]] = {}

        # First pass: collect walker assignments from each Sum body.
        seen_in_sum: Dict[str, Dict] = {}
        for sum_node in self.body.atoms(sp.Sum):
            dummy, lo, hi = sum_node.args[1]
            dummy_name = dummy.name
            inner = sum_node.args[0]

            sum_sparse_refs = set()
            for idx_node in inner.atoms(sp.Indexed):
                base = idx_node.base.name
                if base not in sparse_names:
                    continue
                indices = idx_node.indices
                if len(indices) != 2:
                    raise NotImplementedError(
                        f"LoopEqn {self.name!r}: sparse 2-D Param "
                        f"{base!r} accessed with {len(indices)} "
                        f"indices — expected exactly 2"
                    )
                if (str(indices[0]) != outer_name
                        or str(indices[1]) != dummy_name):
                    raise NotImplementedError(
                        f"LoopEqn {self.name!r}: sparse 2-D Param "
                        f"{base!r} in the body of ``Sum`` over "
                        f"{dummy_name!r} must be accessed as "
                        f"[{outer_name}, {dummy_name}] to qualify as "
                        f"a CSR walker, got {idx_node}. Non-walker "
                        f"access to sparse 2-D Params is not yet "
                        f"supported."
                    )
                sum_sparse_refs.add(base)

            if not sum_sparse_refs:
                continue
            if len(sum_sparse_refs) > 1:
                raise NotImplementedError(
                    f"LoopEqn {self.name!r}: ``Sum`` body contains "
                    f"{sorted(sum_sparse_refs)} — multiple sparse "
                    f"2-D Params sharing the same inner skeleton "
                    f"(\"Case B\") is not yet supported. Split into "
                    f"separate ``Sum``s, one per sparse walker."
                )
            (walker_name,) = sum_sparse_refs
            param_obj = self.var_map[walker_name]
            n_cols = param_obj.v.shape[1]
            if int(lo) != 0 or int(hi) != n_cols - 1:
                raise NotImplementedError(
                    f"LoopEqn {self.name!r}: ``Sum`` over {dummy_name!r} "
                    f"with range ({int(lo)}, {int(hi)}) does not span "
                    f"the full column extent ({n_cols}) of sparse "
                    f"walker {walker_name!r}. Partial-column sparse "
                    f"walkers are not yet supported."
                )
            seen_in_sum[walker_name] = {'param': param_obj}

        # Second pass: every sparse 2-D Param referenced at all in the
        # body must appear only through the Sum walker pattern. Catch
        # bare/outer-Sum accesses here.
        referenced = set()
        for idx_node in self.body.atoms(sp.Indexed):
            if idx_node.base.name in sparse_names:
                referenced.add(idx_node.base.name)
        missing = referenced - set(seen_in_sum.keys())
        if missing:
            raise NotImplementedError(
                f"LoopEqn {self.name!r}: sparse 2-D Params "
                f"{sorted(missing)} are referenced outside a "
                f"``Sum(..., (dummy, 0, n-1))`` walker, or with a "
                f"non-matching access pattern. Every sparse 2-D "
                f"Param must appear only as the walker of a ``Sum`` "
                f"over its full column range."
            )

        # Materialise CSR arrays.
        for walker_name, info in seen_in_sum.items():
            param_obj = info['param']
            csc = param_obj.v
            if not hasattr(csc, 'tocsr'):
                raise TypeError(
                    f"LoopEqn {self.name!r}: sparse 2-D Param "
                    f"{walker_name!r} has sparse=True but .v is not a "
                    f"scipy sparse object (type={type(csc).__name__})"
                )
            csr = csc.tocsr()
            walkers[walker_name] = {
                'data': np.ascontiguousarray(csr.data, dtype=float),
                'indices': np.ascontiguousarray(csr.indices, dtype=np.int64),
                'indptr': np.ascontiguousarray(csr.indptr, dtype=np.int64),
            }
        return walkers

    def _build_num_eqn(self) -> Callable:
        """Generate a Python callable that evaluates the LoopEqn body.

        The callable accepts the Var/Param values as positional
        arguments in lex-sorted SYMBOLS order (matching what
        ``Eqn.NUM_EQN`` would expect from ``sympy.lambdify``) and
        returns a 1D numpy array of length ``self.n_outer``.

        Shares the same translator as :meth:`print_njit_source` (the
        explicit-accumulator prelude approach), so sparse CSR walking
        kicks in automatically when the body references a sparse 2-D
        ``Param`` in an inner-``Sum`` ``[outer, dummy]`` position. The
        inline path makes the per-walker CSR arrays available as the
        generated function's module-level globals via the ``exec``
        namespace — no explicit args needed.
        """
        arg_names = sorted(self.SYMBOLS.keys())
        outer_idx_name = self.outer_index.name

        state = {
            'acc_counter': 0,
            'prelude': [],
            'var_map': self.var_map,
            'outer_name': outer_idx_name,
            'sparse_walker_ctx': None,
        }
        body_expr = _translate_loop_body_njit(self.body, state)

        indent = '    '
        inner_indent = indent * 2
        lines = [
            f"def _loop_eqn_func({', '.join(arg_names)}):",
            f"{indent}out = np.empty({self.n_outer})",
            f"{indent}for {outer_idx_name} in range({self.n_outer}):",
        ]
        for stmt in state['prelude']:
            lines.append(f"{inner_indent}{stmt}")
        lines.append(f"{inner_indent}out[{outer_idx_name}] = {body_expr}")
        lines.append(f"{indent}return out")
        source = '\n'.join(lines) + '\n'

        # ``SolCF`` is the Solverz custom-function namespace
        # (``Solverz.num_api.custom_function``) used by Solverz's
        # ``_numpycode`` methods for non-numpy primitives, e.g.
        # ``SolCF.Heaviside`` — the numba-friendly heaviside helper.
        # The inline path here must expose it by the same name so a
        # LoopEqn body that references ``heaviside(...)`` resolves
        # at eval time.
        from Solverz.num_api import custom_function as _SolCF
        ns: Dict[str, object] = {'np': np, 'SolCF': _SolCF}
        # Inject pre-computed CSR arrays as module-level globals so the
        # generated ``for _sz_kk ... in range(_sz_csr_<M>_indptr[i], ...)``
        # loop resolves them at call time. No need to plumb them through
        # the function signature — the generated body looks them up via
        # the exec namespace's ``__globals__``.
        for walker_name, csr in self._sparse_csr.items():
            ns[f'_sz_csr_{walker_name}_data'] = csr['data']
            ns[f'_sz_csr_{walker_name}_indices'] = csr['indices']
            ns[f'_sz_csr_{walker_name}_indptr'] = csr['indptr']
        exec(source, ns)
        func = ns['_loop_eqn_func']
        # Stash the source for debugging
        func._loopeqn_source = source  # type: ignore[attr-defined]
        return func

    def njit_arg_names(self) -> List[str]:
        """Lex-sorted Var/Param names this LoopEqn's ``inner_F<N>``
        actually receives as arguments in the JIT module path.

        Sparse 2-D ``Param``s used as CSR walkers are EXCLUDED — their
        CSR arrays (``_sz_csr_<M>_data`` / ``_sz_csr_<M>_indices`` /
        ``_sz_csr_<M>_indptr``) are injected as module-level constants
        via ``mut_mat_mappings`` at render time, and the generated
        body references them by those fixed names rather than
        receiving them through the call signature.

        This matters because scipy ``csc_array`` objects are not
        understood by Numba — the wrapper ``F_`` would have to
        decompose each sparse Param anyway, and we prefer to bind the
        CSR view once at module load time rather than re-unpack on
        every Newton call.
        """
        return [nm for nm in sorted(self.SYMBOLS.keys())
                if nm not in self._sparse_csr]

    def print_njit_source(self, func_name: str) -> str:
        """Return Numba-compatible Python source for an ``inner_F<N>``
        sub-function that evaluates this LoopEqn.

        Used by the module printer (``module_printer(jit=True)`` path).
        Numba's ``@njit`` does not support generator expressions, so
        each ``sympy.Sum`` in the body is hoisted into an explicit
        accumulator variable that is initialised inside the outer loop
        and incremented by an explicit inner ``for`` loop.

        The generated function takes this LoopEqn's
        :meth:`njit_arg_names` (sparse 2-D Params excluded — see that
        method's docstring) and returns a 1-D ndarray of length
        ``n_outer``. The call site in ``print_eqn_assignment`` matches
        this arg list exactly.

        Sparse walker support
        ---------------------
        When ``self._sparse_csr`` is non-empty, the translator emits
        ``_sz_csr_<M>_data``/``_indices``/``_indptr`` references that
        the module printer must make available at module level (via
        ``mut_mat_mappings`` in :mod:`module_generator`). Those
        references resolve to pre-computed CSR arrays of the
        ``Param``'s CSC value, loaded once at module import time.

        Raises ``NotImplementedError`` for nested ``Sum``s — the inner
        Sum's accumulator would need to reset on each iteration of the
        outer Sum's dummy, which the linear prelude approach does not
        express. SolMuseum's network modules don't (currently) need
        nested Sums; if a real use case appears, the right answer is
        to emit explicit nested ``for`` loops.
        """
        arg_names = self.njit_arg_names()
        outer_name = self.outer_index.name
        n_outer = self.n_outer

        state = {
            'acc_counter': 0,
            'prelude': [],
            'var_map': self.var_map,
            'outer_name': outer_name,
            'sparse_walker_ctx': None,
        }
        body_expr = _translate_loop_body_njit(self.body, state)

        indent = '    '
        inner_indent = indent * 2
        lines = [
            f"def {func_name}({', '.join(arg_names)}):",
            f"{indent}out = np.empty({n_outer})",
            f"{indent}for {outer_name} in range({n_outer}):",
        ]
        for stmt in state['prelude']:
            lines.append(f"{inner_indent}{stmt}")
        lines.append(f"{inner_indent}out[{outer_name}] = {body_expr}")
        lines.append(f"{indent}return out")
        return '\n'.join(lines) + '\n'

    def derive_derivative(self):
        """Compute per-Var Jacobian blocks symbolically.

        For each ``var_map`` entry that is a Solverz :class:`Var`
        (Params are skipped — they're constants from the model's POV),
        we differentiate the body w.r.t. ``IndexedBase[k]`` for a fresh
        outer-loop-style index ``k``. Sympy turns the Sum-over-Idx into
        a ``KroneckerDelta`` sum, which ``.doit()`` collapses to a
        ``Piecewise`` whose first branch is the per-element coefficient
        — exactly the per-(outer_index, k) Jacobian entry. We strip the
        bounds wrapper (``k`` is always in range by construction),
        translate the resulting sympy expression back to a Solverz
        :class:`Expr` involving :class:`Para`, and wrap the result in
        a regular :class:`EqnDiff`. The standard ``JacBlock`` machinery
        then classifies it as a constant-matrix block (when ``DeriExpr``
        is ``Para`` or ``-Para``) and the existing
        ``print_J_block`` path inlines the data — no custom J-side
        printer needed for this simplest pattern.

        Currently handled patterns (Phase 0 minimum):

        - ``±IndexedBase('M')[outer_index, k]`` with ``M`` a 2-D
          ``Param``. Translates to ``±Para('M', dim=2)``.

        Anything else raises ``NotImplementedError`` from
        :func:`_translate_loop_jac`. Phase 0.4 will extend this to
        cover the patterns SolMuseum's network modules actually need.
        """
        # ``Param(ParamBase, sSymBasic)`` inherits from sSymBasic too,
        # so we can't filter Vars with ``isinstance(_, sSymBasic)`` —
        # use the concrete ``Var`` class.
        from Solverz.variable.ssymbol import Var

        # Fresh derivative index. Use a name unlikely to collide with
        # any Idx the user put in their body.
        k = sp.Idx('_sz_loop_dk')

        for indexed_base_name, sol_obj in self.var_map.items():
            if not isinstance(sol_obj, Var):
                continue  # skip Params

            var_iVar = sol_obj.symbol
            target_base = sp.IndexedBase(indexed_base_name)
            target_slot = target_base[k]

            # Symbolically differentiate, then collapse the resulting
            # KroneckerDelta-Sum, then push any leading sign into the
            # Piecewise branches so the walker sees a uniform shape.
            deriv = sp.diff(self.body, target_slot).doit()
            deriv = sp.piecewise_fold(deriv)

            if is_zero(deriv):
                continue

            # Translate the (sympy IndexedBase) result back to a
            # (Solverz Para) expression that the standard EqnDiff
            # machinery can lambdify and that JacBlock can classify
            # as a constant-matrix block.
            sz_expr = _translate_loop_jac(deriv,
                                          self.outer_index,
                                          k,
                                          self.n_outer,
                                          self.var_map)

            ed = EqnDiff(
                name=f'Diff {self.name} w.r.t. {var_iVar.name}',
                eqn=sz_expr,
                diff_var=var_iVar,
            )
            self.derivatives[var_iVar.name] = ed


class _LoopJacEye(Function):
    """Sympy ``Function`` node that prints to ``np.eye(n)``.

    Emitted by :func:`_translate_loop_jac` whenever a LoopEqn
    derivative simplifies to ``KroneckerDelta(outer_index, k)``, which
    happens when a ``Var`` appears as an outer-indexed term in the body
    (e.g. ``ix[i] - sum_j (G[i, j] * ux[j])`` from the SolMuseum
    ``eps_network.mdl(dyn=True)`` rectangular current balance — the
    block ``∂F[i]/∂ix[k] = δ_{i,k}`` is the identity matrix).

    Has no free symbols, so the standard ``Eqn.lambdify`` /
    ``Eqn.NUM_EQN`` machinery wraps it as ``lambda: np.eye(n)``.
    ``JacBlock`` then sees a 2-D matrix Value0; because ``DeriExpr``
    is a ``Function`` (not a ``Para``) it routes through the mutable-
    matrix path. The mutable-matrix kernel re-evaluates
    ``np.eye(n)`` on every Newton step, but the cost is negligible
    (n is small and the result is cached after the first call).
    """

    @classmethod
    def eval(cls, n):
        # Don't auto-simplify; we want a Function application that the
        # printer can intercept.
        return None

    def _numpycode(self, printer, **kwargs):
        n_arg = self.args[0]
        return f'np.eye({printer._print(n_arg)})'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)

    def _lambdacode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


def _translate_loop_jac(expr, outer_idx, k, n_outer, var_map):
    """Translate the result of ``sp.diff(loop_body, IndexedBase[k]).doit()``
    back to a Solverz :class:`Expr`.

    The input has one of two rough shapes:

    1. ``Piecewise((<expr involving IndexedBase[outer, k]>,
                    (0 ≤ k) & (k ≤ n - 1)),
                   (0, True))``

       — produced when the body has ``Sum_j (… IndexedBase[…, j] …)``
       and we differentiate w.r.t. ``IndexedBase[k]``. ``.doit()``
       collapses the Sum via ``KroneckerDelta``, leaving a Piecewise
       wrapping the per-(outer, k) coefficient. We strip the bounds
       wrapper (``k`` is always in range by construction).

    2. ``KroneckerDelta(outer, k)`` (or wrapped in a sign Mul)

       — produced when the body has a bare ``IndexedBase[outer]``
       residual term referring to a Var. ``ix[i].diff(ix[k])`` is
       just the Kronecker delta, with no Sum involved. We translate
       this to ``_LoopJacEye(n_outer)`` which prints to ``np.eye(n)``.

    Walking strategy
    ----------------
    - ``Piecewise``  → take the first non-zero branch (recursively).
    - ``KroneckerDelta(a, b)`` with ``{a, b} == {outer_idx, k}``
      → ``_LoopJacEye(n_outer)``.
    - ``Indexed(IndexedBase('M'), outer, k)`` with ``M`` a 2-D Param
      → ``Para('M', dim=2)``.
    - Numbers pass through; ``Add`` / ``Mul`` / generic ``Function``
      walk children and reconstruct.
    - Bare ``Idx`` or unknown ``Symbol`` raises ``NotImplementedError``
      so the next missing case surfaces loudly.

    Phase 0 only handles the patterns above. Mixed Var-Var products,
    1-D Var/Param accesses, non-rectangular Sums, etc., will be added
    incrementally as Phase 0.4 sub-tests need them.
    """
    from Solverz.variable.ssymbol import sSymBasic
    from Solverz.equation.param import ParamBase

    outer_name = outer_idx.name
    k_name = k.name

    def walk(e):
        if isinstance(e, sp.Piecewise):
            # Take the first non-zero branch — by construction we
            # know ``k`` is always in range, so the bounds branch is
            # the live one.
            for branch_expr, _cond in e.args:
                if not is_zero(branch_expr):
                    return walk(branch_expr)
            return sp.S.Zero
        if isinstance(e, KroneckerDelta):
            arg_names = {str(a) for a in e.args}
            if arg_names == {outer_name, k_name}:
                # δ_{i,k} over the outer-index x diff-index span is the
                # n_outer × n_outer identity matrix.
                return _LoopJacEye(sp.Integer(n_outer))
            raise NotImplementedError(
                f"LoopEqn jacobian translator: KroneckerDelta with "
                f"args {e.args} — only ``KroneckerDelta(outer_idx, "
                f"k)`` is currently supported"
            )
        if isinstance(e, sp.Indexed):
            base_name = e.base.name
            if base_name not in var_map:
                raise NotImplementedError(
                    f"LoopEqn jacobian translator: IndexedBase "
                    f"{base_name!r} has no var_map entry"
                )
            sol_obj = var_map[base_name]
            indices = e.indices
            if isinstance(sol_obj, ParamBase) and sol_obj.dim == 2:
                if len(indices) != 2:
                    raise NotImplementedError(
                        f"LoopEqn jacobian translator only handles "
                        f"2-D Param accessed with exactly two "
                        f"indices, got {e}"
                    )
                # Drop the (outer, k) indices — the resulting Solverz
                # Para is the full matrix and JacBlock's vector+matrix
                # path handles the per-(row, col) addressing.
                return Para(sol_obj.name, dim=2)
            raise NotImplementedError(
                f"LoopEqn jacobian translator does not yet handle "
                f"IndexedBase {base_name!r} of "
                f"{type(sol_obj).__name__} dim="
                f"{getattr(sol_obj, 'dim', None)}"
            )
        if isinstance(e, (sp.Integer, sp.Float, sp.Rational, sp.Number)):
            return e
        if isinstance(e, sp.Idx):
            # Bare Idx symbols should never reach the translator —
            # they're only valid inside an ``Indexed`` slot or as a
            # ``KroneckerDelta`` arg, both handled above.
            raise NotImplementedError(
                f"LoopEqn jacobian translator: bare Idx {e!r} "
                f"in derivative expression"
            )
        if hasattr(e, 'args') and len(e.args) > 0 and not e.is_Atom:
            new_args = [walk(a) for a in e.args]
            return e.func(*new_args)
        if isinstance(e, sp.Symbol):
            raise NotImplementedError(
                f"LoopEqn jacobian translator: unexpected bare "
                f"Symbol {e!r}"
            )
        raise NotImplementedError(
            f"LoopEqn jacobian translator does not yet handle "
            f"{type(e).__name__}: {e!r}"
        )

    return walk(expr)


def _translate_loop_body(expr) -> str:
    """Recursively translate a sympy expression to a Python expression
    string. Supports the subset needed for ``LoopEqn`` bodies:

    - ``sympy.Add``, ``sympy.Mul``, ``sympy.Pow`` — arithmetic
    - ``sympy.Sum(body, (j, lo, hi))`` — rewritten as
      ``sum((body) for j in range(lo, hi+1))``
    - ``sympy.Indexed(IndexedBase('G'), i, j)`` — left as ``G[i, j]``
      (the IndexedBase name resolves to a numpy-array argument at
      eval time)
    - ``sympy.Idx`` / ``sympy.Symbol`` — emit as the bare name
    - ``sympy.Integer`` / ``sympy.Float`` / ``sympy.Number`` — emit as
      the literal value

    Everything else raises ``NotImplementedError`` — the test driver
    catches this and surfaces the unsupported node type so we can
    extend the translator.
    """
    if isinstance(expr, sp.Sum):
        if len(expr.args) < 2 or len(expr.args[1]) != 3:
            raise NotImplementedError(
                f"LoopEqn body translator only supports single-index "
                f"Sum with explicit (dummy, lo, hi); got {expr}"
            )
        inner_body = expr.args[0]
        dummy, lo, hi = expr.args[1]
        body_code = _translate_loop_body(inner_body)
        return (
            f"sum(({body_code}) "
            f"for {dummy.name} in range({int(lo)}, {int(hi) + 1}))"
        )
    if isinstance(expr, sp.Add):
        parts = [_translate_loop_body(a) for a in expr.args]
        return "(" + " + ".join(parts) + ")"
    if isinstance(expr, sp.Mul):
        parts = [_translate_loop_body(a) for a in expr.args]
        return "(" + " * ".join(parts) + ")"
    if isinstance(expr, sp.Pow):
        base = _translate_loop_body(expr.args[0])
        exp = _translate_loop_body(expr.args[1])
        return f"({base} ** {exp})"
    if isinstance(expr, sp.Indexed):
        base_name = expr.base.name
        index_strs = [_translate_loop_body(idx) for idx in expr.indices]
        return f"{base_name}[{', '.join(index_strs)}]"
    if isinstance(expr, sp.Idx):
        return expr.name
    if isinstance(expr, sp.Symbol):
        return expr.name
    if isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
        return str(float(expr))
    if isinstance(expr, (int, float)):
        return str(expr)
    raise NotImplementedError(
        f"LoopEqn body translator does not yet handle "
        f"{type(expr).__name__}: {expr!r}"
    )


def _find_sum_sparse_walker(sum_node, var_map, outer_name):
    """Return ``(walker_name, walker_param)`` if the ``Sum`` body has
    exactly one sparse 2-D ``Param`` accessed as
    ``[outer_name, dummy]``; ``None`` otherwise.

    Mirrors the logic in :meth:`LoopEqn._collect_sparse_walkers` but
    operates on a single ``Sum`` node for the per-Sum per-call check
    the translator needs. The stricter invariants (no non-walker
    sparse access anywhere in the body, full-column range) are
    enforced at ``LoopEqn.__init__`` time, so this helper can trust
    that any sparse Param it finds is already a valid walker.
    """
    from Solverz.equation.param import ParamBase

    dummy, _lo, _hi = sum_node.args[1]
    dummy_name = dummy.name
    inner = sum_node.args[0]
    found = None
    for idx_node in inner.atoms(sp.Indexed):
        base = idx_node.base.name
        obj = var_map.get(base)
        if not isinstance(obj, ParamBase):
            continue
        if not (getattr(obj, 'sparse', False) and obj.dim == 2):
            continue
        indices = idx_node.indices
        if (len(indices) == 2
                and str(indices[0]) == outer_name
                and str(indices[1]) == dummy_name):
            if found is None:
                found = (base, obj)
            elif found[0] != base:
                # __init__ already rejects multi-sparse Sums, but be
                # defensive here so an unchecked call site gets a
                # clear error rather than silently picking one.
                return None
    return found


def _translate_loop_body_njit(expr, state) -> str:
    """Numba-compatible body translator shared by the inline
    ``_build_num_eqn`` path and the JIT ``print_njit_source`` path.

    Numba's ``@njit`` does not accept generator expressions, and the
    inline path also benefits from the same explicit-accumulator shape
    (consistent code, single translator to maintain). Each ``sympy.Sum``
    in the body is hoisted into an explicit accumulator variable that is
    initialised inside the outer loop and incremented by an explicit
    inner ``for`` loop; the for-loop statements are appended to
    ``state['prelude']`` and the accumulator's name is returned in place
    of the ``Sum`` expression.

    Parameters
    ----------
    expr : sympy.Expr
        Sub-expression to translate.
    state : dict
        Mutable walker state. Must contain:

        - ``acc_counter`` (int — increments per ``Sum`` encountered)
        - ``prelude`` (list[str] — accumulator + for-loop statements)
        - ``var_map`` (dict — IndexedBase-name → Var/Param, for sparse
          walker detection)
        - ``outer_name`` (str — the LoopEqn outer index name, used to
          recognise the walker access pattern ``M[outer, dummy]``)
        - ``sparse_walker_ctx`` (None or dict with ``base_name`` and
          ``kk_var``; when set, any ``Indexed`` node whose base matches
          ``base_name`` is rewritten to ``_sz_csr_<base>_data[kk_var]``)

    Sparse walker rewrite
    ---------------------
    When a ``Sum``'s body contains a sparse 2-D ``Param`` accessed as
    ``M[outer, dummy]``, the Sum is rewritten from::

        _sz_loop_acc_N = 0.0
        for j in range(0, n_cols):
            _sz_loop_acc_N += <body with M[i, j], ux[j], ...>

    to::

        _sz_loop_acc_N = 0.0
        for _sz_kk_N in range(_sz_csr_M_indptr[i], _sz_csr_M_indptr[i + 1]):
            j = _sz_csr_M_indices[_sz_kk_N]
            _sz_loop_acc_N += <body with M[i,j] → _sz_csr_M_data[_sz_kk_N],
                              ux[j] → ux[j] (j is still in scope)>

    i.e. the outer loop iterates CSR row ``i`` directly, visiting only
    stored non-zeros of ``M``. Co-dummy-accesses like ``ux[j]`` stay
    literal because ``j`` is still a live variable bound to the current
    column index.

    Notes
    -----
    Nested ``Sum``s (Sum inside the body of another Sum) are rejected:
    the inner Sum's accumulator would need to reset on each iteration
    of the outer Sum's dummy, which the linear ``prelude`` approach
    does not capture. SolMuseum's network modules don't currently use
    nested Sums; if a real use case appears, the right answer is to
    emit a properly nested for-loop block instead of the flat prelude.
    """
    if isinstance(expr, sp.Sum):
        if len(expr.args) < 2 or len(expr.args[1]) != 3:
            raise NotImplementedError(
                f"LoopEqn njit body translator only supports single-"
                f"index Sum with explicit (dummy, lo, hi); got {expr}"
            )
        inner_body = expr.args[0]
        if inner_body.has(sp.Sum):
            raise NotImplementedError(
                f"LoopEqn njit body translator: nested Sum (Sum "
                f"inside the body of another Sum) is not yet "
                f"supported; got {expr}"
            )
        dummy, lo, hi = expr.args[1]

        # Detect sparse walker for this Sum.
        walker = _find_sum_sparse_walker(expr, state['var_map'],
                                         state['outer_name'])

        acc_idx = state['acc_counter']
        state['acc_counter'] += 1
        acc_name = f"_sz_loop_acc_{acc_idx}"

        if walker is None:
            # Dense path — identical to the original behaviour.
            inner_expr_code = _translate_loop_body_njit(inner_body, state)
            state['prelude'].append(f"{acc_name} = 0.0")
            state['prelude'].append(
                f"for {dummy.name} in range({int(lo)}, {int(hi) + 1}):"
            )
            state['prelude'].append(f"    {acc_name} += {inner_expr_code}")
            return acc_name

        # Sparse-walker path.
        walker_name, _walker_param = walker
        kk_name = f"_sz_kk_{acc_idx}"

        # Push the sparse walker context so inner ``M[outer, dummy]``
        # references rewrite to ``_sz_csr_<M>_data[kk]``. Restore on
        # exit even though there's no chance of exception here — the
        # next sibling Sum must see a clean context.
        prev_ctx = state['sparse_walker_ctx']
        state['sparse_walker_ctx'] = {
            'base_name': walker_name,
            'kk_var': kk_name,
        }
        try:
            inner_expr_code = _translate_loop_body_njit(inner_body, state)
        finally:
            state['sparse_walker_ctx'] = prev_ctx

        state['prelude'].append(f"{acc_name} = 0.0")
        state['prelude'].append(
            f"for {kk_name} in range("
            f"_sz_csr_{walker_name}_indptr[{state['outer_name']}], "
            f"_sz_csr_{walker_name}_indptr[{state['outer_name']} + 1]"
            f"):"
        )
        state['prelude'].append(
            f"    {dummy.name} = _sz_csr_{walker_name}_indices[{kk_name}]"
        )
        state['prelude'].append(f"    {acc_name} += {inner_expr_code}")
        return acc_name

    if isinstance(expr, sp.Add):
        parts = [_translate_loop_body_njit(a, state) for a in expr.args]
        return "(" + " + ".join(parts) + ")"
    if isinstance(expr, sp.Mul):
        parts = [_translate_loop_body_njit(a, state) for a in expr.args]
        return "(" + " * ".join(parts) + ")"
    if isinstance(expr, sp.Pow):
        base = _translate_loop_body_njit(expr.args[0], state)
        exp = _translate_loop_body_njit(expr.args[1], state)
        return f"({base} ** {exp})"
    if isinstance(expr, sp.Indexed):
        base_name = expr.base.name
        ctx = state.get('sparse_walker_ctx')
        if ctx is not None and base_name == ctx['base_name']:
            # Rewrite sparse walker access to a CSR data load. Index
            # correctness (must be [outer, dummy]) is guaranteed by
            # ``_collect_sparse_walkers`` at __init__ time.
            return f"_sz_csr_{base_name}_data[{ctx['kk_var']}]"
        index_strs = [_translate_loop_body_njit(idx, state)
                      for idx in expr.indices]
        return f"{base_name}[{', '.join(index_strs)}]"
    if isinstance(expr, sp.Idx):
        return expr.name
    if isinstance(expr, sp.Symbol):
        return expr.name
    if isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
        return str(float(expr))
    if isinstance(expr, (int, float)):
        return str(expr)
    # Sympy / Solverz function nodes — trig, exp / log, abs, sign,
    # heaviside, arctan2, etc. Matches the numpy backends Solverz's
    # own ``_numpycode`` methods emit (see
    # ``Solverz/sym_algebra/functions.py``), so the LoopEqn-generated
    # code uses the same symbols numba already imports at module level.
    if isinstance(expr, sp.Function):
        fname = expr.func.__name__
        mapped = _FUNCTION_NUMPY_MAP.get(fname)
        if mapped is not None:
            args_code = [_translate_loop_body_njit(a, state)
                         for a in expr.args]
            return f"{mapped}({', '.join(args_code)})"
    raise NotImplementedError(
        f"LoopEqn njit body translator does not yet handle "
        f"{type(expr).__name__}: {expr!r}"
    )


# Map sympy / Solverz ``Function`` node ``.func.__name__`` → the
# numpy (or ``SolCF``) call the LoopEqn translator emits. Names are
# chosen to match what Solverz's existing ``_numpycode`` methods in
# ``Solverz/sym_algebra/functions.py`` already emit so generated
# modules / the inline exec namespace both have the callable in scope.
_FUNCTION_NUMPY_MAP = {
    # sympy standard
    'sin': 'np.sin',
    'cos': 'np.cos',
    'tan': 'np.tan',
    'asin': 'np.arcsin',
    'acos': 'np.arccos',
    'atan': 'np.arctan',
    'exp': 'np.exp',
    'log': 'np.log',
    'sqrt': 'np.sqrt',
    # sympy Abs; Solverz re-exports under the same class name.
    'Abs': 'np.abs',
    # Solverz-specific names (``Solverz.sym_algebra.functions``).
    'ln': 'np.log',
    'Sign': 'np.sign',
    'atan2': 'np.arctan2',
    # Heaviside needs Solverz's numba-friendly custom helper, not
    # ``np.heaviside`` (which takes an explicit second argument).
    'heaviside': 'SolCF.Heaviside',
    'Heaviside': 'SolCF.Heaviside',
}


class Ode(Eqn):
    r"""
    The class for ODE reading

    .. math::

         \frac{\mathrm{d}y}{\mathrm{d}t}=f(t,y)

    where $y$ is the state vector.

    """

    def __init__(self, name: str,
                 f,
                 diff_var: Union[iVar, IdxVar, Var]):
        super().__init__(name, f)
        diff_var = sSym2Sym(diff_var)
        self.diff_var = diff_var
        self.LHS = Derivative(diff_var, t)
