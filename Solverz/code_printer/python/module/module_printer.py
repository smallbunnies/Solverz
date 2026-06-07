from __future__ import annotations

import warnings
from typing import Any, Set, Tuple

import numpy as np
import sympy as sp
from sympy import Add, Function, IndexedBase, Mul, Number, S
from Solverz.sym_algebra.functions import transpose
from Solverz.code_printer.python.utilities import *
from Solverz.code_printer.python.module.mutable_mat_analyzer import (
    analyze_mutable_mat_expr,
    generate_block_function_code,
    MutableMatBlockMapping,
)
from Solverz.equation.source import format_source


def _with_docstring(func_src: str, doc: str) -> str:
    """Insert ``doc`` as a one-line docstring immediately after the
    ``def …:`` line of a rendered function-source string.

    ``func_src`` is the text of a single function whose first ``def``
    line ends with ``:``. ``doc`` is sanitised to one physical line
    (collapsed whitespace, no triple quotes). Returns ``func_src``
    unchanged when ``doc`` is empty. Works uniformly for the AST-rendered
    (pycode) F/J functions and the string-rendered LoopEqn / loop-jac
    kernels because all of them start with a ``def`` line.
    """
    if not doc:
        return func_src
    doc = ' '.join(doc.split()).replace('"""', "'''")
    lines = func_src.split('\n')
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith('def ') and ln.rstrip().endswith(':'):
            indent = ' ' * (len(ln) - len(stripped) + 4)
            lines.insert(i + 1, f'{indent}"""{doc}"""')
            break
    return '\n'.join(lines)


class MutableMatJacDataModule(Function):
    """Evaluate a mutable matrix Jacobian expression and extract data.

    The slower-but-always-correct fallback used when the symbolic
    analyser cannot decompose a block into the diag / row-scale /
    col-scale terms that the vectorised fast path understands.

    The input expression has ``Diag`` replaced by ``SpDiag`` so the
    printed code uses ``sps.diags`` (not ``np.diagflat``) wherever
    ``Diag`` appears — sparse matrix products stay sparse, avoiding
    dense blowup for the common case of sparse parameters.

    However, when the block mixes sparse and dense ``dim=2``
    parameters, the evaluated expression can collapse to a numpy
    ``ndarray`` which does NOT have a ``.tocsr()`` method. The
    generated code therefore dispatches on ``issparse(...)`` at
    runtime, using sparse fancy indexing for sparse results and dense
    advanced indexing for ndarray results.

    Generated code (conceptually)::

        _sz_fb = (sparse_expr)
        data[...] = (asarray(_sz_fb.tocsr()[[rows],[cols]]).ravel()
                     if sps.issparse(_sz_fb)
                     else asarray(_sz_fb)[rows, cols])
    """

    def __new__(cls, expr, coo_row, coo_col):
        # Substitute Diag → SpDiag so the printed code uses sps.diags
        # where the expression is naturally sparse. ``SpDiag`` still
        # collapses correctly when its argument is a dense ndarray.
        sparse_expr = expr.replace(Diag, SpDiag)
        obj = super().__new__(cls, sparse_expr)
        obj._coo_row = coo_row.tolist()
        obj._coo_col = coo_col.tolist()
        return obj

    def _numpycode(self, printer, **kwargs):
        inner = printer._print(self.args[0])
        rows = self._coo_row
        cols = self._coo_col
        # Delegate to a runtime helper that dispatches between sparse
        # and dense results. ``SolCF.mutable_mat_fallback_extract`` lives
        # in ``Solverz.num_api.custom_function`` and is imported via
        # ``module_parser`` into every generated module.
        return (
            f'SolCF.mutable_mat_fallback_extract('
            f'{inner}, {rows}, {cols})'
        )

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


# %%
def print_Hvp(eqs_type: str,
              eqn_size: int,
              var_addr: Address,
              PARAM: Dict[str, ParamBase],
              nstep: int = 0):
    if eqn_size != var_addr.total_size:
        raise ValueError(f"Hvp matrix, with size ({eqn_size}*{var_addr.total_size}), not square")
    fp = print_Hvp_prototype(eqs_type,
                             'Hvp_',
                             nstep=nstep)
    body = []
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(PARAM)
    body.extend(param_assignments)
    body.extend(print_trigger(PARAM))
    args = [symbols('_data_hvp', real=True), symbols('v_', real=True)] + var_list + param_list
    body.extend([Assignment(iVar('data_hvp', internal_use=True), FunctionCall('inner_Hvp', args))])
    body.extend([Return(coo_2_csc_hvp(eqn_size, var_addr.total_size))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_Hvp(var_addr: Address,
                    PARAM: Dict[str, ParamBase],
                    hvp: Hvp,
                    nstep: int = 0):
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    param_assignments, param_list = print_param(PARAM)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_Hvp',
                           [symbols('_data_hvp', real=True), symbols('v_', real=True)] + args)
    body = []

    code_sub_inner_Hvp_blocks = []
    count = 0
    addr_by_ele_0 = 0
    for eqn_name, jbs_row in hvp.blocks_sorted.items():
        for var, jb in jbs_row.items():
            rhs = jb.SpDeriExpr
            SymbolsInDeri_ = list(Eqn(f'temp' + eqn_name + var.name, rhs).SYMBOLS.values())
            # add real assumption
            SymbolsInDeri = [symbols(arg.name, real=True) for arg in SymbolsInDeri_]
            addr_by_ele = slice(addr_by_ele_0, addr_by_ele_0 + jb.SpEleSize)
            if not jb.IsDeriNumber:
                # _data_[0:1] = inner_Hvp0(v_, t1, x)
                body.append(Assignment(iVar('_data_hvp', internal_use=True)[addr_by_ele],
                                       FunctionCall(f'inner_Hvp{int(count)}', SymbolsInDeri)))

                # def inner_Hvp0(v_, t1, x):
                #     return -t1 * pi * cos(pi * x) + 1
                fp1 = FunctionPrototype(real, f'inner_Hvp{int(count)}', SymbolsInDeri)
                body1 = [Return(rhs)]
                fd1 = FunctionDefinition.from_FunctionPrototype(fp1, body1)
                code_sub_inner_Hvp_blocks.append(pycode(fd1, fully_qualified_modules=False))
                count += 1
            addr_by_ele_0 += jb.SpEleSize
    temp = iVar('_data_hvp', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return {'code_inner_Hvp': pycode(fd, fully_qualified_modules=False),
            'code_sub_inner_Hvp': code_sub_inner_Hvp_blocks}


def print_J(eqs_type: str,
            eqn_size: int,
            var_addr: Address,
            PARAM: Dict[str, ParamBase],
            shape: List[int],
            nstep: int = 0,
            mutable_matrix_blocks=None):
    if eqn_size != var_addr.total_size:
        raise ValueError(f"Jac matrix, with size ({eqn_size}*{var_addr.total_size}), not square")
    fp = print_F_J_prototype(eqs_type,
                             'J_',
                             nstep)
    body = []
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    body.extend(var_assignments)
    # Load sparse matrices for use in J_ wrapper (mutable matrix blocks),
    # but do NOT pass them to inner_J (which is @njit-able).
    param_assignments, param_list = print_param(PARAM,
                                                include_sparse_in_list=False)
    body.extend(param_assignments)
    body.extend(print_trigger(PARAM))
    body.extend([Assignment(iVar('data', internal_use=True),
                            FunctionCall('inner_J', [symbols('_data_', real=True)] + var_list + param_list))])
    # Mutable matrix Jacobian blocks. Two modes:
    #
    # 1. 'vectorized' — the block's expression is a sum of recognised term
    #    shapes (Diag, Diag@Matrix, Matrix@Diag). The block's data array is
    #    assembled by a dedicated generated block function that uses pure
    #    scatter-add loops over precomputed index mappings. This is the fast
    #    path — no scipy.sparse matrix construction per J call.
    #
    # 2. 'fallback' — some term couldn't be classified; fall back to the
    #    slower-but-correct scipy sparse + fancy indexing path via
    #    ``MutableMatJacDataModule``.
    if mutable_matrix_blocks:
        for mb in mutable_matrix_blocks:
            if mb.get('mode') == 'vectorized':
                mapping = mb['mapping']
                block_idx = mb['block_idx']
                # (a) Pre-compute every dense vector the kernel needs —
                # diag inner vectors AND row/col-scale scaling vectors —
                # here in the wrapper, where scipy.sparse is available.
                # The kernel itself then runs scatter-add loops only.
                # Helper placeholder names begin with the reserved
                # prefix ``_sz_mb_``; pass ``internal_use=True`` so the
                # SolSymBasic reserved-prefix guard lets them through.
                for arg_name, t in zip(mb['diag_arg_names'], mapping.diag_terms):
                    body.append(Assignment(
                        iVar(arg_name, internal_use=True), t['inner_expr']))
                for arg_name, t in zip(mb['rs_arg_names'], mapping.row_scale_terms):
                    body.append(Assignment(
                        iVar(arg_name, internal_use=True), t['var_expr']))
                for arg_name, t in zip(mb['cs_arg_names'], mapping.col_scale_terms):
                    body.append(Assignment(
                        iVar(arg_name, internal_use=True), t['var_expr']))
                for arg_name, t in zip(mb['bs_u_arg_names'], mapping.biscale_terms):
                    body.append(Assignment(
                        iVar(arg_name, internal_use=True), t['u_expr']))
                for arg_name, t in zip(mb['bs_v_arg_names'], mapping.biscale_terms):
                    body.append(Assignment(
                        iVar(arg_name, internal_use=True), t['v_expr']))
                # (b) Build the call argument list in the order the block
                # function expects.
                call_args = []
                for arg_name in mb['diag_arg_names']:
                    call_args.append(symbols(arg_name, real=True))
                for arg_name in mb['rs_arg_names']:
                    call_args.append(symbols(arg_name, real=True))
                for arg_name in mb['cs_arg_names']:
                    call_args.append(symbols(arg_name, real=True))
                for arg_name in mb['bs_u_arg_names']:
                    call_args.append(symbols(arg_name, real=True))
                for arg_name in mb['bs_v_arg_names']:
                    call_args.append(symbols(arg_name, real=True))
                # Mapping arrays (loaded at module-level from setting)
                for ti in range(len(mapping.diag_terms)):
                    call_args.append(symbols(f'_sz_mb_{block_idx}_diag_out_{ti}', real=True))
                    call_args.append(symbols(f'_sz_mb_{block_idx}_diag_src_{ti}', real=True))
                for ti in range(len(mapping.row_scale_terms)):
                    call_args.append(symbols(f'_sz_mb_{block_idx}_rs_out_{ti}', real=True))
                    call_args.append(symbols(f'_sz_mb_{block_idx}_rs_src_{ti}', real=True))
                    call_args.append(symbols(f'_sz_mb_{block_idx}_rs_dat_{ti}', real=True))
                for ti in range(len(mapping.col_scale_terms)):
                    call_args.append(symbols(f'_sz_mb_{block_idx}_cs_out_{ti}', real=True))
                    call_args.append(symbols(f'_sz_mb_{block_idx}_cs_src_{ti}', real=True))
                    call_args.append(symbols(f'_sz_mb_{block_idx}_cs_dat_{ti}', real=True))
                for ti in range(len(mapping.biscale_terms)):
                    call_args.append(symbols(f'_sz_mb_{block_idx}_bs_out_{ti}', real=True))
                    call_args.append(symbols(f'_sz_mb_{block_idx}_bs_row_{ti}', real=True))
                    call_args.append(symbols(f'_sz_mb_{block_idx}_bs_col_{ti}', real=True))
                    call_args.append(symbols(f'_sz_mb_{block_idx}_bs_dat_{ti}', real=True))
                body.append(Assignment(
                    iVar('data', internal_use=True)[mb['addr_slice']],
                    FunctionCall(mb['fn_name'], call_args)))
            elif mb.get('mode') == 'loop_eqn':
                # LoopEqn-native path (Phase J4). Kernel calls are
                # now emitted INSIDE ``inner_J`` itself by
                # ``print_inner_J`` so the whole assembly lives in
                # a single ``@njit`` compilation unit. Nothing to
                # do in the ``J_`` wrapper — the returned ``data``
                # from ``inner_J`` already holds the block values.
                pass
            else:
                # Fallback: scipy sparse fancy indexing
                body.append(Assignment(
                    iVar('data', internal_use=True)[mb['addr_slice']],
                    MutableMatJacDataModule(mb['expr'], mb['coo_row'], mb['coo_col'])))
    body.extend([Return(coo_2_csc(shape[0], shape[1]))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_J(var_addr: Address,
                  PARAM: Dict[str, ParamBase],
                  jac: Jac,
                  nstep: int = 0,
                  source_map=None):
    source_map = source_map or {}
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    # inner_J must not receive sparse matrices (they're handled in J_ wrapper).
    param_assignments, param_list = print_param(PARAM,
                                                include_sparse_in_list=False)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_J', [symbols('_data_', real=True)] + args)
    body = []

    code_sub_inner_J_blocks = []
    no_njit_indices = set()
    mutable_matrix_blocks = []
    mut_mat_block_funcs = []  # generated @njit block function sources
    mut_mat_mappings = {}     # arrays to store in eqn_parameter (setting)
    count = 0
    addr_by_ele_0 = 0
    eqn_size_by_name = {}
    for eqn_name, jbs_row in jac.blocks_sorted.items():
        for var, jb in jbs_row.items():
            rhs = jb.SpDeriExpr
            SymbolsInDeri_ = list(Eqn(f'temp' + eqn_name + var.name, rhs).SYMBOLS.values())
            # add real assumption
            SymbolsInDeri = [symbols(arg.name, real=True) for arg in SymbolsInDeri_]
            addr_by_ele = slice(addr_by_ele_0, addr_by_ele_0 + jb.SpEleSize)

            jac_constant = jb.IsDeriNumber

            if jb.DeriType == 'matrix':
                # LoopEqn-native J block path (Phase J3.3). The
                # JacBlock carries a reference to its source
                # LoopEqnDiff via ``_loop_eqn_diff``, which owns a
                # pre-generated dense kernel function source. We
                # emit the kernel as a top-level @njit function
                # (via ``mut_mat_block_funcs``) and record a block
                # descriptor for ``print_J`` to render the wrapper
                # call + fancy-index Assignment.
                if hasattr(jb, '_loop_eqn_diff'):
                    ed = jb._loop_eqn_diff
                    block_idx = len(mutable_matrix_blocks)
                    kernel_fn_name = f'_sz_loop_jac_kernel_{block_idx}'
                    # Emit any scalar CSR point-lookup helpers the
                    # kernel body references, each as its own
                    # ``mut_mat_block_funcs`` entry so module_generator
                    # prepends its own ``@njit(cache=True)``. Dedup
                    # helpers by name across LoopEqnDiff instances —
                    # the helper body is value-independent (keyed on
                    # Param name and module-global CSR arrays), so
                    # emitting it once per sparse Param is enough.
                    for helper_src in getattr(ed, 'helper_sources', []):
                        if helper_src in mut_mat_block_funcs:
                            continue
                        mut_mat_block_funcs.append(helper_src)
                    # Rename the kernel function (it was
                    # generated with the sanitized EqnDiff name at
                    # LoopEqnDiff construction time — that name
                    # would collide with the inline path's
                    # closure kernel in an odd edge case, and the
                    # module-local indexed name is cleaner).
                    block_source = ed.kernel_source.replace(
                        ed._kernel_func_name, kernel_fn_name
                    )
                    _doc = (f"d({eqn_name})/d({var.name})"
                            f"{format_source(source_map.get(eqn_name))}")
                    mut_mat_block_funcs.append(_with_docstring(block_source, _doc))

                    row_key = f'_sz_loop_jac_row_{block_idx}'
                    col_key = f'_sz_loop_jac_col_{block_idx}'
                    mut_mat_mappings[row_key] = jb.CooRow.astype(np.int64)
                    mut_mat_mappings[col_key] = jb.CooCol.astype(np.int64)

                    mutable_matrix_blocks.append({
                        'addr_slice': addr_by_ele,
                        'mode': 'loop_eqn',
                        'block_idx': block_idx,
                        'kernel_fn_name': kernel_fn_name,
                        'kernel_symbols': sorted(ed.SYMBOLS.keys()),
                        'row_key': row_key,
                        'col_key': col_key,
                    })
                    addr_by_ele_0 += jb.SpEleSize
                    continue

                if jb.is_mutable_matrix:
                    # Mutable matrix derivative: analyze the expression into
                    # typed terms (diag / row_scale / col_scale), generate a
                    # dedicated loop-based block function, and record the
                    # precomputed index mapping arrays so the runtime J_ can
                    # assemble this block's data with pure numpy/numba scatter
                    # loops — no scipy.sparse construction at every call.
                    block_idx = len(mutable_matrix_blocks)
                    eqn_size = jb.EqnAddr.stop - jb.EqnAddr.start
                    mapping = analyze_mutable_mat_expr(
                        jb.SpDeriExpr, jb.CooRow, jb.CooCol, PARAM, eqn_size)
                    if mapping.has_fallback:
                        _emit_l2_fallback_warnings(
                            eqn_name, var.name, mapping.fallback_pieces)
                    block_info = {
                        'addr_slice': addr_by_ele,
                        'expr': jb.SpDeriExpr,
                        'coo_row': jb.CooRow,
                        'coo_col': jb.CooCol,
                        'block_idx': block_idx,
                        'mapping': mapping,
                        'mode': 'vectorized' if not mapping.has_fallback else 'fallback',
                    }
                    if not mapping.has_fallback:
                        # Generate the dedicated block function source.
                        fn_name = f'_mut_block_{block_idx}'
                        diag_arg_names = [f'_sz_mb_{block_idx}_u{ti}'
                                          for ti in range(len(mapping.diag_terms))]
                        rs_arg_names = [f'_sz_mb_{block_idx}_rsv{ti}'
                                        for ti in range(len(mapping.row_scale_terms))]
                        cs_arg_names = [f'_sz_mb_{block_idx}_csv{ti}'
                                        for ti in range(len(mapping.col_scale_terms))]
                        bs_u_arg_names = [f'_sz_mb_{block_idx}_bsu{ti}'
                                          for ti in range(len(mapping.biscale_terms))]
                        bs_v_arg_names = [f'_sz_mb_{block_idx}_bsv{ti}'
                                          for ti in range(len(mapping.biscale_terms))]
                        block_info['fn_name'] = fn_name
                        block_info['diag_arg_names'] = diag_arg_names
                        block_info['rs_arg_names'] = rs_arg_names
                        block_info['cs_arg_names'] = cs_arg_names
                        block_info['bs_u_arg_names'] = bs_u_arg_names
                        block_info['bs_v_arg_names'] = bs_v_arg_names
                        block_code = generate_block_function_code(
                            fn_name, mapping,
                            diag_arg_names, rs_arg_names, cs_arg_names,
                            bs_u_arg_names, bs_v_arg_names)
                        mut_mat_block_funcs.append(block_code)
                        # Collect the mapping arrays for the eqn_parameter
                        for ti, t in enumerate(mapping.diag_terms):
                            mut_mat_mappings[f'_sz_mb_{block_idx}_diag_out_{ti}'] = t['out_pos']
                            mut_mat_mappings[f'_sz_mb_{block_idx}_diag_src_{ti}'] = t['src_idx']
                        for ti, t in enumerate(mapping.row_scale_terms):
                            mut_mat_mappings[f'_sz_mb_{block_idx}_rs_out_{ti}'] = t['out_pos']
                            mut_mat_mappings[f'_sz_mb_{block_idx}_rs_src_{ti}'] = t['src']
                            mut_mat_mappings[f'_sz_mb_{block_idx}_rs_dat_{ti}'] = t['mat_data']
                        for ti, t in enumerate(mapping.col_scale_terms):
                            mut_mat_mappings[f'_sz_mb_{block_idx}_cs_out_{ti}'] = t['out_pos']
                            mut_mat_mappings[f'_sz_mb_{block_idx}_cs_src_{ti}'] = t['src']
                            mut_mat_mappings[f'_sz_mb_{block_idx}_cs_dat_{ti}'] = t['mat_data']
                        for ti, t in enumerate(mapping.biscale_terms):
                            mut_mat_mappings[f'_sz_mb_{block_idx}_bs_out_{ti}'] = t['out_pos']
                            mut_mat_mappings[f'_sz_mb_{block_idx}_bs_row_{ti}'] = t['src_row']
                            mut_mat_mappings[f'_sz_mb_{block_idx}_bs_col_{ti}'] = t['src_col']
                            mut_mat_mappings[f'_sz_mb_{block_idx}_bs_dat_{ti}'] = t['mat_data']
                    mutable_matrix_blocks.append(block_info)
                    addr_by_ele_0 += jb.SpEleSize
                    continue
                jac_constant = True
                # if the matrix derivative is triggerable, then update it in the Jacobian function call
                if isinstance(jb.DeriExpr, Para):
                    if PARAM[jb.DeriExpr.name].triggerable:
                        jac_constant = False
                elif isinstance(-jb.DeriExpr, Para):
                    name = (-jb.DeriExpr).name
                    if PARAM[name].triggerable:
                        jac_constant = False

            if not jac_constant:
                # _data_[0:1] = inner_J0(t1, x)
                body.append(Assignment(iVar('_data_', internal_use=True)[addr_by_ele],
                                       FunctionCall(f'inner_J{int(count)}', SymbolsInDeri)))

                # def inner_J0(t1, x):
                #     return -t1 * pi * cos(pi * x) + 1
                fp1 = FunctionPrototype(real, f'inner_J{int(count)}', SymbolsInDeri)
                body1 = [Return(rhs)]
                fd1 = FunctionDefinition.from_FunctionPrototype(fp1, body1)
                _doc = (f"d({eqn_name})/d({var.name})"
                        f"{format_source(source_map.get(eqn_name))}")
                code_sub_inner_J_blocks.append(
                    _with_docstring(pycode(fd1, fully_qualified_modules=False), _doc))
                count += 1
            addr_by_ele_0 += jb.SpEleSize

    # Move ``loop_eqn`` block kernel calls INSIDE ``inner_J`` so the
    # whole assembly happens inside a single ``@njit`` compilation
    # unit. Each kernel is itself ``@njit``-decorated, so the call
    # is inlined by numba at JIT time — no Python/numba boundary
    # crossing per kernel call at runtime. Row / col arrays are
    # module-level numpy globals (``_sz_loop_jac_row_<N>`` /
    # ``_sz_loop_jac_col_<N>``), which numba accepts via global
    # capture when the function is compiled.
    for mb in mutable_matrix_blocks:
        if mb.get('mode') != 'loop_eqn':
            continue
        kernel_args = [symbols(nm, real=True)
                       for nm in mb['kernel_symbols']]
        row_sym = symbols(mb['row_key'], real=True)
        col_sym = symbols(mb['col_key'], real=True)
        body.append(Assignment(
            iVar('_data_', internal_use=True)[mb['addr_slice']],
            FunctionCall(
                mb['kernel_fn_name'],
                kernel_args + [row_sym, col_sym],
            ),
        ))

    temp = iVar('_data_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return {'code_inner_J': pycode(fd, fully_qualified_modules=False),
            'code_sub_inner_J': code_sub_inner_J_blocks,
            'no_njit_sub_inner_J': no_njit_indices,
            'mutable_matrix_blocks': mutable_matrix_blocks,
            'mut_mat_block_funcs': mut_mat_block_funcs,
            'mut_mat_mappings': mut_mat_mappings}


def _classify_matmul_placeholders(precompute_info, PARAM, emit_warnings=False):
    """Classify every extracted ``Mat_Mul`` placeholder into fast /
    fallback and compute which ``Para`` symbols the fallback code
    path still needs to have loaded in the ``F_`` wrapper.

    This replaces the earlier per-placeholder ``_is_csc_matvec_fast_path``
    check, which looked only at ``matrix_arg`` and did not account for
    three concrete regressions that broke generated code:

    1. **Nested fallbacks referencing fast placeholders** — e.g.
       ``Mat_Mul(-A, Mat_Mul(B, x))``. ``extract_matmuls`` visits the
       inner ``Mat_Mul(B, x)`` first and produces placeholder
       ``_sz_mm_0``; the outer's matrix is ``-A`` (fallback) and its
       operand is ``_sz_mm_0``. A naive classifier marks ``_sz_mm_0``
       as fast (matrix is a bare ``Para``) but the outer fallback
       still expects ``_sz_mm_0`` to be materialised in the wrapper
       before its scipy SpMV runs, so the wrapper emits a reference
       to an undefined name.

    2. **Unresolved matrix factors in the operand** — e.g.
       ``Mat_Mul(A, B, x)``. Solverz's ``Mat_Mul`` accepts any number
       of arguments; ``extract_matmuls`` folds 3+-arg nodes into
       ``matrix_arg = A`` / ``operand_arg = Mat_Mul(B, x)``, and the
       freshly-constructed inner ``Mat_Mul(B, x)`` is not re-walked.
       A classifier that accepts this as fast would feed
       ``Mat_Mul(B, x)`` into ``SolCF.csc_matvec`` as if it were a
       plain vector, which either crashes (``B`` isn't in ``inner_F``'s
       argument list) or silently produces wrong results.

    3. **Sparse matrix references in the operand** — e.g. an operand
       expression that still contains a sparse ``dim=2`` ``Para``
       symbol. These symbols are not in ``inner_F``'s argument list
       (only their CSC flat fields are), so inlining the operand
       inside ``inner_F`` would reference an undefined name.

    The algorithm is a two-pass fixed point:

    - **Shape filter** — a placeholder is a *fast candidate* iff its
      ``matrix_arg`` is a bare sparse ``dim=2`` ``Para`` **and** its
      ``operand_arg`` contains neither ``Mat_Mul`` atoms nor sparse
      ``dim=2`` ``Para`` references.
    - **Demotion** — any fast candidate whose name appears as a free
      symbol in another (non-fast) placeholder's ``matrix_arg`` or
      ``operand_arg`` is demoted to fallback. The scan repeats until
      no more demotions happen (needed because demotion can cascade
      in a chain of nested ``Mat_Mul``s, though this is rare).

    Returns
    -------
    fast_info : dict[str, tuple[str, sympy.Expr]]
        Maps a fast-path placeholder's name to ``(matrix_name,
        operand_arg)`` so callers can emit the
        ``SolCF.csc_matvec(<m>_data, ..., operand)`` call directly.
    fallback_names : set[str]
        Names of placeholders that must be computed in the ``F_``
        wrapper (via scipy) and passed through ``inner_F`` as extra
        arguments.
    fallback_symbols : set[str]
        Para names that appear anywhere in a fallback placeholder's
        ``matrix_arg`` or ``operand_arg``. The wrapper must load each
        of these (``A = p_["A"]``) or the scipy evaluation below
        fails.
    """
    fast_info: Dict[str, Tuple[str, Any]] = {}
    fallback_names: Set[str] = set()
    fallback_symbols: Set[str] = set()

    if not precompute_info:
        return fast_info, fallback_names, fallback_symbols

    # Gather (name, matrix_arg, operand_arg) triples, deduped by name.
    all_triples: List[Tuple[str, Any, Any]] = []
    seen: Set[str] = set()
    for eqn_info in precompute_info:
        for placeholder, matrix_arg, operand_arg in eqn_info['matmuls']:
            if placeholder.name in seen:
                continue
            seen.add(placeholder.name)
            all_triples.append((placeholder.name, matrix_arg, operand_arg))

    def _shape_is_fast(mat, op):
        # matrix_arg must be a bare sparse dim=2 Para
        if not isinstance(mat, Para):
            return False
        if mat.name not in PARAM:
            return False
        p = PARAM[mat.name]
        if not (getattr(p, 'dim', 0) == 2
                and getattr(p, 'sparse', False)):
            return False
        # operand_arg must not contain any Mat_Mul node — ``Mat_Mul``
        # inside an operand means extract_matmuls folded a multi-arg
        # Mat_Mul and left an inner product unresolved (see R3).
        if hasattr(op, 'has') and op.has(Mat_Mul):
            return False
        # operand_arg must not reference any sparse dim=2 Para — those
        # are not available by name inside ``inner_F`` (only their CSC
        # flat fields are), so a direct reference would NameError.
        if hasattr(op, 'free_symbols'):
            for s in op.free_symbols:
                if isinstance(s, Para) and s.name in PARAM:
                    p_op = PARAM[s.name]
                    if (getattr(p_op, 'dim', 0) == 2
                            and getattr(p_op, 'sparse', False)):
                        return False
        return True

    # Pass 1 — shape-based classification.
    fast_candidates: Set[str] = set()
    for name, mat, op in all_triples:
        if _shape_is_fast(mat, op):
            fast_candidates.add(name)

    # Pass 2 — demote fast candidates consumed by any non-fast
    # placeholder's matrix or operand. Repeat until fixed point so
    # cascading dependencies propagate. Track *which* upstream
    # placeholder caused each demotion for root-cause traceback.
    demotion_edges: Dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for name, mat, op in all_triples:
            if name in fast_candidates:
                continue  # not a consumer of interest — fast placeholders
                          # cannot create wrapper-side dependencies
            for expr in (mat, op):
                if not hasattr(expr, 'free_symbols'):
                    continue
                for s in expr.free_symbols:
                    s_name = getattr(s, 'name', None)
                    if s_name and s_name in fast_candidates:
                        fast_candidates.discard(s_name)
                        demotion_edges[s_name] = name  # demoted → consumer
                        changed = True

    # Build the output sets.
    for name, mat, op in all_triples:
        if name in fast_candidates:
            # ``mat`` is guaranteed to be a Para by _shape_is_fast.
            fast_info[name] = (mat.name, op)
        else:
            fallback_names.add(name)
            # Collect every Para referenced by the fallback expression
            # so the wrapper filter keeps the corresponding loads.
            for expr in (mat, op):
                if hasattr(expr, 'free_symbols'):
                    for s in expr.free_symbols:
                        if isinstance(s, Para):
                            fallback_symbols.add(s.name)

    if emit_warnings:
        _emit_l1_fallback_warnings(all_triples, fast_candidates, PARAM,
                                   demotion_edges)

    return fast_info, fallback_names, fallback_symbols


def _is_matrix_factor(arg, PARAM):
    """Return ``True`` if ``arg`` is a matrix-valued symbolic operand —
    a ``dim=2`` ``Para`` (sparse or dense) registered in ``PARAM``, or a
    ``transpose(...)`` of one. Used to split the factors of a ``Mul``
    into "scalar coefficients" and "matrix factors" without assuming the
    coefficients are numeric (a Solverz ``Param('c', 2.0)`` scalar is
    just as legitimate as a literal ``2``).
    """
    if isinstance(arg, Para):
        p = PARAM.get(arg.name)
        return p is not None and getattr(p, 'dim', 0) == 2
    if isinstance(arg, transpose):
        return any(_is_matrix_factor(a, PARAM) for a in arg.args)
    return False


def _classify_l1_fallback_reason(mat, op, PARAM):
    """Identify which Layer 1 fallback shape this ``(matrix_arg,
    operand_arg)`` pair hits and return ``(reason, expression_str,
    suggestion)`` for the user-facing diagnostic warning.

    Each return triple is consumed by ``_emit_l1_fallback_warnings``
    to format a multi-line ``UserWarning`` body. The classification is
    deliberately deterministic and shape-based — no symbolic
    rewriting — so users see the *exact* expression that broke the
    fast path next to the suggested rewrite. Branches, in order:

    1. ``Mat_Mul(-A, x)`` — bare negation of a sparse Param.
    2. ``Mat_Mul(transpose(A), x)`` — transposed matrix Param.
    3. ``Mat_Mul(c*A, x)`` — single matrix factor with any (numeric or
       symbolic) scalar coefficients, i.e. ``Mul`` with exactly one
       matrix-valued factor.
    4. ``Mat_Mul(A*B, x)`` — element-wise ``Mul`` of two or more
       matrix-valued factors (operator mix-up: should be ``Mat_Mul``).
    5. ``Mat_Mul(A, Mat_Mul(B, x))`` — operand contains an unresolved
       ``Mat_Mul`` (R3 multi-arg fold).
    6. ``Mat_Mul(A, f(B, x))`` — operand references a sparse ``dim=2``
       Param other than through a ``Mat_Mul`` placeholder.
    7. ``Mat_Mul(A+B, x)`` — matrix operand is a sum.
    8. Anything else — generic fallback message.
    """
    # 1) Negation: Mat_Mul(-A, x) where A is a sparse dim=2 Para.
    if (isinstance(mat, Mul)
            and len(mat.args) >= 2
            and mat.args[0] == S.NegativeOne):
        rest = mat.args[1:]
        if len(rest) == 1 and isinstance(rest[0], Para):
            inner = rest[0]
            p = PARAM.get(inner.name)
            if (p is not None
                    and getattr(p, 'dim', 0) == 2
                    and getattr(p, 'sparse', False)):
                return (
                    f"matrix operand is `-{inner.name}` (negation of a "
                    f"sparse Param), not a bare Param",
                    f"-{inner.name}",
                    f"-Mat_Mul({inner.name}, <operand>) — move the "
                    f"negation outside Mat_Mul",
                )

    # 2) Transpose: Mat_Mul(transpose(A), x). The CSC flat fields of A
    # don't transpose at runtime; the user must precompute A.T as its
    # own sparse Param.
    if isinstance(mat, transpose) and len(mat.args) == 1:
        inner = mat.args[0]
        if isinstance(inner, Para):
            p = PARAM.get(inner.name)
            if (p is not None
                    and getattr(p, 'dim', 0) == 2):
                return (
                    f"matrix operand is `transpose({inner.name})` — the "
                    f"@njit fast path can't transpose a sparse matrix at "
                    f"runtime",
                    f"transpose({inner.name})",
                    f"predeclare ``{inner.name}_T = Param("
                    f"'{inner.name}_T', value={inner.name}_value.T, "
                    f"dim=2, sparse=True)`` and write "
                    f"``Mat_Mul({inner.name}_T, <operand>)``",
                )

    # 3 + 4) Mat_Mul(c*A, x) and element-wise A*B mix-ups.
    # Generalised so that any non-matrix factor (numeric Number, scalar
    # Para, scalar iVar) counts as a "scalar coefficient" — the
    # user-facing fix is identical: factor scalars outside ``Mat_Mul``.
    if isinstance(mat, Mul):
        matrix_factors = [a for a in mat.args if _is_matrix_factor(a, PARAM)]
        scalar_factors = [a for a in mat.args if not _is_matrix_factor(a, PARAM)]
        # Skip the bare negation already handled above.
        already_negation = (
            len(scalar_factors) == 1
            and scalar_factors[0] == S.NegativeOne
            and len(matrix_factors) == 1
        )
        if matrix_factors and not already_negation:
            # 3) one matrix factor with any number of scalar factors.
            if len(matrix_factors) == 1:
                inner = matrix_factors[0]
                inner_str = (inner.name if isinstance(inner, Para)
                             else str(inner))
                coeff = (Mul(*scalar_factors) if scalar_factors else S.One)
                return (
                    f"matrix operand is `({coeff})*{inner_str}` — a "
                    f"scalar multiple of a matrix, not a bare Param",
                    str(mat),
                    f"({coeff}) * Mat_Mul({inner_str}, <operand>) — "
                    f"factor the scalar coefficient outside ``Mat_Mul``",
                )
            # 4) two or more matrix factors → element-wise product.
            names = ' * '.join(
                a.name if isinstance(a, Para) else str(a)
                for a in matrix_factors
            )
            return (
                f"matrix operand is an element-wise ``Mul`` of "
                f"{len(matrix_factors)} matrix factors ({names}) — "
                f"Python ``*`` between matrices is element-wise, not "
                f"matrix-product",
                str(mat),
                "rewrite the matrix product as nested ``Mat_Mul(A, "
                "Mat_Mul(B, <operand>))``, or distribute to a sum of "
                "single-matrix ``Mat_Mul`` calls",
            )

    # 5) R3 — operand contains unresolved Mat_Mul: Mat_Mul(A, B, x) where
    # ``extract_matmuls`` folded the inner product into the operand and
    # did not re-walk the fresh inner Mat_Mul(B, x).
    # Must fire BEFORE operand_has_sparse_para: an operand like
    # ``Mat_Mul(B, x)`` also contains sparse Para ``B`` in its
    # free_symbols, but the nesting suggestion is more specific.
    if hasattr(op, 'has') and op.has(Mat_Mul):
        return (
            "operand contains an unresolved ``Mat_Mul`` — "
            "``extract_matmuls`` folded a multi-argument ``Mat_Mul`` "
            "and left the inner product in the operand",
            str(op),
            "nest explicitly: ``Mat_Mul(A, Mat_Mul(B, x))`` so the "
            "inner ``Mat_Mul`` is walked to its own placeholder",
        )

    # 6) operand references a sparse dim=2 Para — not available by name
    # inside ``inner_F`` (only its CSC flat fields are).
    if hasattr(op, 'free_symbols'):
        sparse_paras = set()
        for s in op.free_symbols:
            if isinstance(s, Para) and s.name in PARAM:
                p_op = PARAM[s.name]
                if (getattr(p_op, 'dim', 0) == 2
                        and getattr(p_op, 'sparse', False)):
                    sparse_paras.add(s.name)
        if sparse_paras:
            names = ', '.join(sorted(sparse_paras))
            return (
                f"operand references sparse ``dim=2`` Param(s): {names} — "
                f"not available by name inside ``inner_F``",
                str(op),
                f"precompute the lookup as a ``dim=1`` vector Param "
                f"before passing into ``Mat_Mul``",
            )

    # 7) Sum: Mat_Mul(A+B, x).
    if isinstance(mat, Add):
        arg_strs = [
            arg.name if isinstance(arg, Para) else str(arg)
            for arg in mat.args
        ]
        distributed = ' + '.join(
            f'Mat_Mul({s}, <operand>)' for s in arg_strs
        )
        all_para = all(isinstance(a, Para) for a in mat.args)
        if all_para:
            return (
                f"matrix operand is a sum of {len(mat.args)} matrices "
                f"({' + '.join(arg_strs)}), not a bare Param",
                str(mat),
                f"distribute: {distributed}",
            )
        return (
            f"matrix operand is a sum of {len(mat.args)} terms, "
            f"not a bare sparse ``dim=2`` Param",
            str(mat),
            f"distribute Mat_Mul over the sum: {distributed}",
        )

    # 8) Generic fallback.
    return (
        "matrix operand is not a bare sparse `dim=2` Param",
        str(mat),
        "rewrite the matrix expression so the matrix argument is a "
        "bare sparse `dim=2` Param",
    )


def _format_fallback_warning(header, body):
    """Format a multi-line fallback warning body. ``header`` is the
    one-line summary (placeholder name / Jac-block context). ``body``
    is a list of ``(label, value)`` tuples — typically Reason /
    Expression-or-Term / Suggested rewrite.

    Centralising the formatting keeps L1 and L2 warnings visually
    aligned and gives the test suite a single thing to assert against
    if the warning template changes.
    """
    lines = [f"  {label}: {value}" for label, value in body]
    return header + "\n" + "\n".join(lines)


# stacklevel from ``warnings.warn`` to user code, traced through the
# production call chain ``render_modules → print_F →
# _classify_matmul_placeholders → _emit_l1_fallback_warnings → warn``
# (5 frames) and ``render_modules → print_inner_J →
# _emit_l2_fallback_warnings → warn`` (4 frames). When the emitters are
# called directly from a test, the stacklevel overshoots into pytest
# internals — that's fine because tests assert on message content, not
# on warning location.
_L1_STACKLEVEL = 5
_L2_STACKLEVEL = 4


def _emit_l1_fallback_warnings(all_triples, fast_candidates, PARAM,
                               demotion_edges=None):
    """Emit one ``UserWarning`` per Mat_Mul fallback placeholder.

    Skips placeholders whose matrix is a dense ``dim=2`` Param —
    those are already covered by ``_warn_dense_matmul_params`` in
    ``equations.py``, which fires at ``FormJac`` time and applies
    equally to inline mode (where Layer 1 / Layer 2 fallback
    distinctions don't exist).
    """
    if demotion_edges is None:
        demotion_edges = {}
    for name, mat, op in all_triples:
        if name in fast_candidates:
            continue
        # Skip dense dim=2 Param — already warned by _warn_dense_matmul_params.
        if isinstance(mat, Para):
            p = PARAM.get(mat.name)
            if (p is not None
                    and getattr(p, 'dim', 0) == 2
                    and not getattr(p, 'sparse', False)):
                continue
        # Demotion root-cause traceback: if this placeholder would have
        # been on the fast path by shape alone but was demoted because
        # another fallback placeholder consumes it, tell the user which
        # upstream placeholder to fix first.
        if name in demotion_edges:
            upstream = demotion_edges[name]
            warnings.warn(
                _format_fallback_warning(
                    f"Mat_Mul placeholder {name!r} was demoted from the "
                    f"fast path because {upstream!r} consumes it as an "
                    f"operand and is itself on the fallback path.",
                    [(f"Fix the upstream fallback ({upstream!r}) first",
                      "demoted placeholders recover automatically")],
                ),
                UserWarning, stacklevel=_L1_STACKLEVEL
            )
            continue
        reason, expression_str, suggestion = _classify_l1_fallback_reason(
            mat, op, PARAM)
        warnings.warn(
            _format_fallback_warning(
                f"Mat_Mul placeholder {name!r} falls back to scipy.sparse "
                f"SpMV (slower than the @njit csc_matvec fast path).",
                [
                    ("Reason", reason),
                    ("Expression", f"Mat_Mul({expression_str}, <operand>)"),
                    ("Suggested rewrite", suggestion),
                ],
            ),
            UserWarning, stacklevel=_L1_STACKLEVEL
        )


def _classify_l2_fallback_reason(piece):
    """Identify why a mutable Jacobian term landed in the analyzer's
    fallback bucket and return ``(reason, suggestion)``.

    Mirrors the structural checks in
    :func:`analyze_mutable_mat_expr.handle`. After stripping the leading
    sign, a term lands in ``fallback_pieces`` for one of these reasons:

    * It is a ``Mat_Mul`` with ``Diag`` at both ends but the middle
      factor isn't a recognised constant sparse matrix (biscale shape,
      bad ``M``).
    * It is a ``Mat_Mul`` with ``Diag`` at exactly one end but the other
      side isn't a recognised constant sparse matrix.
    * It is a ``Mat_Mul`` with no ``Diag`` at either end — neither
      row-scale, col-scale, nor biscale shape.
    * It is an element-wise ``Mul`` (Python ``*``), not a
      ``Mat_Mul`` — operator mix-up.
    * It is a bare ``Para`` matrix.
    * Anything else.
    """
    # Local import avoids a circular import: this module is imported by
    # the analyzer's owning package at module init time.
    from Solverz.code_printer.python.module.mutable_mat_analyzer import (
        _extract_sign_and_core,
    )
    _, core = _extract_sign_and_core(piece)

    if isinstance(core, Mat_Mul):
        args = list(core.args)
        left_diag = bool(args) and isinstance(args[0], Diag)
        right_diag = bool(args) and isinstance(args[-1], Diag)
        # Two-arg ``Mat_Mul(Diag(u), Diag(v))`` has Diag at both ends but
        # no middle factor. The biscale fast path requires len(args) >= 3
        # in the analyzer (cf. ``_classify_matmul_biscale``); diag-times-
        # diag is degenerate and the user almost certainly meant
        # ``Diag(u * v)`` (two diagonals compose element-wise on the
        # diagonal).
        if left_diag and right_diag and len(args) == 2:
            return (
                "two-argument ``Mat_Mul(Diag(u), Diag(v))`` — both ends "
                "are ``Diag`` but there's no middle matrix factor",
                "rewrite as ``Diag(u * v)`` — two diagonal matrices "
                "compose element-wise on their diagonals",
            )
        if left_diag and right_diag:
            return (
                "biscale shape ``Diag(u) @ M @ Diag(v)`` where ``M`` "
                "isn't a recognised constant sparse matrix (must be a "
                "sparse ``dim=2`` Param or a Mat_Mul chain of constants)",
                "ensure the middle factor materialises to a constant "
                "sparse matrix; if it depends on a variable, split the "
                "term into a sum of supported shapes",
            )
        if left_diag or right_diag:
            shape = "Diag(v) @ M" if left_diag else "M @ Diag(v)"
            return (
                f"single-Diag shape ``{shape}`` where ``M`` isn't a "
                f"recognised constant sparse matrix",
                "ensure the matrix factor materialises to a constant "
                "sparse ``dim=2`` Param or a Mat_Mul chain of constants",
            )
        return (
            "``Mat_Mul`` of matrix operands without a ``Diag`` wrapper "
            "at either end — neither row-scale, col-scale, nor biscale",
            "wrap one operand in ``Diag(...)`` to use a fast-path "
            "shape, or split the term into a sum of supported shapes",
        )

    if isinstance(core, Mul):
        diag_count = sum(1 for a in core.args if isinstance(a, Diag))
        if diag_count:
            return (
                f"term is an element-wise ``Mul`` with {diag_count} "
                f"``Diag`` factor(s), not a matrix-product (``Mat_Mul``)",
                "rewrite as ``Mat_Mul(Diag(u), M, Diag(v))`` (or the "
                "matching single-side shape) — Python ``*`` between "
                "matrices is element-wise and doesn't match the "
                "row/col/biscale fast paths",
            )
        return (
            "term is an element-wise ``Mul`` of matrix operands, not "
            "a matrix-product (``Mat_Mul``)",
            "rewrite as ``Mat_Mul(...)`` so the analyzer can match a "
            "row-scale / col-scale / biscale shape",
        )

    if isinstance(core, Para):
        return (
            "bare matrix ``Param`` not wrapped in ``Diag`` or "
            "``Mat_Mul`` — no fast-path shape applies",
            "wrap the matrix in ``Diag(...)`` if it represents a "
            "diagonal contribution, or compose with a row/col scale "
            "via ``Mat_Mul``",
        )

    return (
        f"term shape ``{type(core).__name__}`` doesn't match any "
        f"supported fast-path (Diag / row-scale / col-scale / biscale)",
        "rewrite as one of: ``Diag(inner)``, ``Mat_Mul(Diag(v), M)``, "
        "``Mat_Mul(M, Diag(v))``, or ``Mat_Mul(Diag(u), M, Diag(v))``",
    )


def _emit_l2_fallback_warnings(eqn_name, var_name, fallback_pieces):
    """Emit one ``UserWarning`` per mutable Jacobian fallback piece."""
    for piece in fallback_pieces:
        reason, suggestion = _classify_l2_fallback_reason(piece)
        warnings.warn(
            _format_fallback_warning(
                f"Mutable Jacobian block (eqn {eqn_name!r}, var "
                f"{var_name!r}) contains a term that doesn't match the "
                f"diag / row-scale / col-scale / biscale fast path.",
                [
                    ("Term", piece),
                    ("Reason", reason),
                    ("Suggested rewrite", suggestion),
                ],
            ),
            UserWarning, stacklevel=_L2_STACKLEVEL
        )


def print_F(eqs_type: str,
            var_addr: Address,
            PARAM: Dict[str, ParamBase],
            nstep: int = 0,
            precompute_info=None):
    """Print the F_ wrapper.

    ``Mat_Mul(A, x)`` precomputes are split into two paths:

    - **Fast path** — ``A`` is a plain sparse ``dim=2`` ``Para``. The
      matvec is emitted into ``inner_F`` via ``SolCF.csc_matvec`` and
      the wrapper does **not** need to do anything: the CSC fields
      (``A_data`` / ``A_indices`` / ``A_indptr`` / ``A_shape0``) are
      already in ``param_list`` via ``print_param`` and flow through
      to ``inner_F`` as normal numpy arrays.
    - **Fallback** — ``A`` is a non-trivial expression (negated
      matrix, nested ``Mat_Mul`` etc.). The old path is kept:
      ``_sz_mm_N = A @ operand`` is emitted as a scipy SpMV inside
      the wrapper and ``_sz_mm_N`` is passed to ``inner_F`` as an
      extra dense argument.
    """
    fp = print_F_J_prototype(eqs_type,
                             'F_',
                             nstep)
    body = []
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(PARAM,
                                                include_sparse_in_list=False)

    # Shared classification (matches what ``print_inner_F`` computes,
    # see :func:`_classify_matmul_placeholders`). ``fallback_symbols``
    # names every ``Para`` that the wrapper must still materialise so
    # the scipy SpMV in a fallback precompute can execute correctly.
    #
    # ``emit_warnings=True`` is set here (not in ``print_inner_F`` or
    # ``print_J``) so each fallback placeholder produces exactly one
    # ``UserWarning`` per render, not three. ``print_F`` is the first
    # function to call the classifier in ``render_modules``.
    _, fallback_names, fallback_symbols = _classify_matmul_placeholders(
        precompute_info, PARAM, emit_warnings=True)

    # Drop sparse dim=2 ``Para`` wrapper loads that are (a) not
    # referenced by any fallback scipy SpMV **and** (b) not in
    # ``param_list`` — i.e., they are loaded purely as a side-effect
    # of ``print_param`` but never used downstream. The
    # ``param_list`` guard is what keeps sparse ``TimeSeriesParam``
    # loads (``A = p_["A"].get_v_t(t)``) in place: those params are
    # *in* ``param_list`` because ``print_param``'s TimeSeriesParam
    # branch appends them, and dropping the wrapper load would leave
    # ``inner_F(..., A, ...)`` referencing an undefined local.
    param_list_names = set()
    for entry in param_list:
        n = getattr(entry, 'name', None)
        if n is not None:
            param_list_names.add(n)
    filtered_assignments = []
    for assign in param_assignments:
        lhs_name = getattr(assign.lhs, 'name', None)
        if lhs_name and lhs_name in PARAM:
            p_obj = PARAM[lhs_name]
            is_sparse_dim2 = (getattr(p_obj, 'dim', 0) == 2
                              and getattr(p_obj, 'sparse', False))
            if (is_sparse_dim2
                    and lhs_name not in fallback_symbols
                    and lhs_name not in param_list_names):
                continue  # nothing downstream needs this load
        filtered_assignments.append(assign)
    body.extend(filtered_assignments)
    body.extend(print_trigger(PARAM))

    # Generate precompute assignments for fallback placeholders only:
    # ``_sz_mm_N = matrix_arg @ operand_arg``. Fast-path placeholders
    # are handled inside ``inner_F`` via ``SolCF.csc_matvec`` and do
    # not need a wrapper-level precompute. Iteration order is
    # precompute_info order, which is ``extract_matmuls`` post-order
    # (inner Mat_Mul first), so any fallback that references a
    # previously-demoted fast placeholder sees it defined.
    inner_extra_args = []
    if precompute_info:
        seen_placeholders = set()
        for eqn_info in precompute_info:
            for placeholder, matrix_arg, operand_arg in eqn_info['matmuls']:
                if placeholder.name in seen_placeholders:
                    continue
                seen_placeholders.add(placeholder.name)
                if placeholder.name not in fallback_names:
                    continue  # handled inside inner_F
                # Fallback: _sz_mm_N = matrix_arg @ operand_arg
                body.append(Assignment(placeholder,
                                       Mat_Mul(matrix_arg, operand_arg)))
                inner_extra_args.append(symbols(placeholder.name, real=True))

    body.extend(
        [Return(FunctionCall('inner_F',
                             [symbols('_F_', real=True)] + var_list + param_list + inner_extra_args))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_F(EQNs: Dict[str, Eqn],
                  EqnAddr: Address,
                  var_addr: Address,
                  PARAM: Dict[str, ParamBase],
                  nstep: int = 0,
                  precompute_info=None):
    """Print the @njit ``inner_F`` dispatcher.

    For fast-path ``Mat_Mul(A, op)`` placeholders (where ``A`` is a
    plain sparse ``dim=2`` ``Para``), this function emits an explicit
    ``_sz_mm_N = SolCF.csc_matvec(A_data, A_indices, A_indptr,
    A_shape0, op)`` assignment at the top of the body and treats
    ``_sz_mm_N`` as a local variable inside the function. The CSC
    fields are already in ``param_list`` (emitted by ``print_param``
    as part of ``model.basic``'s sparse-matrix decomposition), so no
    additional arguments are needed.

    For fallback placeholders (non-``Para`` matrices), the old path
    is preserved: the ``F_`` wrapper computes ``_sz_mm_N`` via scipy
    SpMV and passes it as an extra argument; ``inner_F`` receives it
    the same way it receives any other dense vector.
    """
    var_assignments, var_list = print_var(var_addr,
                                          nstep)
    param_assignments, param_list = print_param(PARAM,
                                                include_sparse_in_list=False)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))

    # Shared classification with ``print_F``. Both functions must
    # agree on which placeholders are fast-path vs fallback —
    # divergence produces ``inner_F`` that either references an
    # undefined local or receives ``_sz_mm_N`` twice.
    fast_info, fallback_names, _ = _classify_matmul_placeholders(
        precompute_info, PARAM)

    # Collect ordered lists for code emission.
    fast_path_matvec = []   # [(placeholder_iVar, matrix_name, operand)]
    fallback_placeholders = []  # [iVar]
    if precompute_info:
        seen = set()
        for eqn_info in precompute_info:
            for placeholder, _, _ in eqn_info['matmuls']:
                if placeholder.name in seen:
                    continue
                seen.add(placeholder.name)
                if placeholder.name in fast_info:
                    mat_name, operand_arg = fast_info[placeholder.name]
                    fast_path_matvec.append(
                        (placeholder, mat_name, operand_arg))
                elif placeholder.name in fallback_names:
                    fallback_placeholders.append(placeholder)

    # Fallback placeholders come in as extra arguments (old path).
    for placeholder in fallback_placeholders:
        args.append(symbols(placeholder.name, real=True))

    fp = FunctionPrototype(real, 'inner_F', [symbols('_F_', real=True)] + args)
    body = []

    # Emit the fast-path matvec prelude before any equation assignments.
    # Each fast placeholder becomes:
    #   _sz_mm_N = SolCF.csc_matvec(<A>_data, <A>_indices,
    #                               <A>_indptr, <A>_shape0, operand)
    # where A_data / A_indices / A_indptr / A_shape0 are already in
    # ``param_list`` (hence in the function's local scope). Numba
    # sees SolCF.csc_matvec via the module-level import and can call
    # it without leaving @njit land.
    for placeholder, mat_name, operand_arg in fast_path_matvec:
        body.append(Assignment(
            iVar(placeholder.name, internal_use=True),
            FunctionCall(
                'SolCF.csc_matvec',
                [symbols(f'{mat_name}_data', real=True),
                 symbols(f'{mat_name}_indices', real=True),
                 symbols(f'{mat_name}_indptr', real=True),
                 symbols(f'{mat_name}_shape0', real=True),
                 operand_arg])))

    body.extend(print_eqn_assignment_with_precompute(EQNs,
                                                     EqnAddr,
                                                     precompute_info))
    temp = iVar('_F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_eqn_assignment_with_precompute(EQNs, EqnAddr, precompute_info):
    """Generate _F_[slice] = inner_F{i}(args...) assignments.

    When precompute_info is provided, the args for each sub-function come
    from the info dict (which has sparse matrices removed and placeholders
    appended for Mat_Mul equations). For non-Mat_Mul equations, args fall
    back to eqn.SYMBOLS.values() — exactly the original behavior.

    ``LoopEqn`` with sparse walkers is a second special case: its
    sparse 2-D Params are NOT in ``inner_F``'s local scope (they're
    loaded in the ``F_`` wrapper's scope but excluded from
    ``param_list`` by ``print_param(..., include_sparse_in_list=False)``),
    so the call site must omit them from the argument list. The
    ``inner_F<N>`` sub-function's signature (emitted by
    ``print_sub_inner_F``) matches this pruned list exactly by using
    :meth:`LoopEqn.njit_arg_names`.
    """
    eqn_declaration = []
    _F_ = iVar('_F_', internal_use=True)
    if precompute_info is None:
        return print_eqn_assignment(EQNs, EqnAddr, True)
    for count, (eqn_name, eqn_info) in enumerate(zip(EQNs.keys(), precompute_info)):
        eqn_address = EqnAddr[eqn_name]
        eqn = EQNs[eqn_name]
        if eqn.mixed_matrix_vector:
            sub_args = [symbols(a.name, real=True) for a in eqn_info['args']]
        elif isinstance(eqn, LoopEqn):
            # Exclude sparse walker Params (they're module-level CSR
            # constants, not call arguments).
            sub_args = [eqn.SYMBOLS[nm] for nm in eqn.njit_arg_names()]
        else:
            # Preserve original behavior for non-matrix equations
            sub_args = list(eqn.SYMBOLS.values())
        eqn_declaration.append(Assignment(_F_[eqn_address],
                                          FunctionCall(f'inner_F{int(count)}', sub_args)))
    return eqn_declaration


def print_sub_inner_F(EQNs: Dict[str, Eqn]):
    """Generate inner_F0, inner_F1, ... sub-functions, one per equation.

    For equations containing Mat_Mul, extract the matrix-vector products and
    replace them with placeholder variables (_sz_mm_0, _sz_mm_1, ...). The
    placeholders are computed in the F_ wrapper (scipy.sparse) and passed in
    as dense vectors. This allows the sub-functions to remain @njit-friendly
    even when the original equation uses sparse matrices.

    Returns
    -------
    code_blocks : list of str
        Generated code for each sub-function.
    precompute_info : list of dict
        Per-equation metadata. Each dict has:
        - 'eqn_name': name of the equation
        - 'new_rhs': RHS with Mat_Mul replaced by placeholders
        - 'matmuls': list of (placeholder_iVar, matrix_arg, operand_arg)
        - 'args': ordered list of argument symbols for inner_F{i}
        - 'matrix_symbols_removed': set of symbol names dropped from args
          (sparse matrix params that only appear inside Mat_Mul)
    """
    code_blocks = []
    precompute_info = []
    # Global Mat_Mul cache: (matrix_arg, operand_arg) -> global placeholder iVar.
    # Shared across all equations so that identical Mat_Mul patterns are
    # computed only once in the F_ wrapper. SymPy's structural equality
    # makes the dict lookup hit on matching expressions.
    mm_cache = {}
    global_mm_counter = [0]
    count = 0
    for eqn_name, eqn in EQNs.items():
        # LoopEqn path: pycode cannot translate ``Sum`` over ``Idx``,
        # so we emit a hand-built source string with explicit nested
        # ``for`` loops (Numba-friendly) instead of going through the
        # AST + pycode pipeline. Args are the same lex-sorted SYMBOLS
        # the dispatcher emits via ``print_eqn_assignment``.
        if isinstance(eqn, LoopEqn):
            # Use njit_arg_names so sparse walker Params are excluded
            # from the sub-function's signature. Their CSR arrays are
            # pulled from module-level constants injected via
            # ``mut_mat_mappings`` by ``render_modules``.
            arg_names = eqn.njit_arg_names()
            args = [symbols(v, real=True) for v in arg_names]
            _doc = f"{eqn_name}{format_source(getattr(eqn, 'source', None))}"
            code_blocks.append(_with_docstring(
                eqn.print_njit_source(f'inner_F{count}'), _doc))
            precompute_info.append({
                'eqn_name': eqn_name,
                'new_rhs': eqn.RHS,
                'matmuls': [],
                'args': args,
                'matrix_symbols_removed': set(eqn._sparse_csr.keys()),
            })
            count += 1
            continue

        # Fast path: non-Mat_Mul equation — original behavior, no changes.
        if not eqn.mixed_matrix_vector:
            args = [symbols(v, real=True) for v in eqn.SYMBOLS.keys()]
            fp = FunctionPrototype(real, f'inner_F{count}', args)
            body = [Return(eqn.RHS)]
            fd = FunctionDefinition.from_FunctionPrototype(fp, body)
            _doc = f"{eqn_name}{format_source(getattr(eqn, 'source', None))}"
            code_blocks.append(_with_docstring(
                pycode(fd, fully_qualified_modules=False), _doc))
            precompute_info.append({
                'eqn_name': eqn_name,
                'new_rhs': eqn.RHS,
                'matmuls': [],
                'args': args,
                'matrix_symbols_removed': set(),
            })
            count += 1
            continue

        # Mat_Mul path: extract matrix-vector products into placeholders.
        new_rhs, matmuls = extract_matmuls(eqn.RHS)
        # Deduplicate: map each local placeholder to a (possibly shared)
        # global placeholder using mm_cache.
        subs_map = {}
        eqn_matmuls = []  # (placeholder, mat_arg, op_arg) unique to this eqn
        for placeholder_local, mat_arg, op_arg in matmuls:
            # Apply in-eqn substitutions so nested Mat_Muls use their
            # global placeholder names before the cache lookup.
            if subs_map:
                if hasattr(mat_arg, 'xreplace'):
                    mat_arg = mat_arg.xreplace(subs_map)
                if hasattr(op_arg, 'xreplace'):
                    op_arg = op_arg.xreplace(subs_map)
            key = (mat_arg, op_arg)
            if key in mm_cache:
                global_placeholder = mm_cache[key]
            else:
                new_name = f'_sz_mm_{global_mm_counter[0]}'
                global_mm_counter[0] += 1
                global_placeholder = iVar(new_name, internal_use=True)
                mm_cache[key] = global_placeholder
            subs_map[placeholder_local] = global_placeholder
            eqn_matmuls.append((global_placeholder, mat_arg, op_arg))
        new_rhs = new_rhs.xreplace(subs_map)

        # Determine which original base symbol names remain in new_rhs.
        # For IdxVar (e.g. x[0:2]), the base name is stored in `name0`;
        # regular symbols use `name` directly. eqn.SYMBOLS keys are base
        # names, so we match against the base.
        remaining_names = set()
        for s in new_rhs.free_symbols:
            if hasattr(s, 'name0'):
                remaining_names.add(s.name0)
            else:
                remaining_names.add(getattr(s, 'name', str(s)))

        ordered_args = []
        matrix_symbols_removed = set()
        for name in eqn.SYMBOLS.keys():
            if name in remaining_names:
                ordered_args.append(symbols(name, real=True))
            else:
                matrix_symbols_removed.add(name)
        # Append placeholders needed by this eqn (skip duplicates)
        seen_in_eqn = set()
        for ph, _, _ in eqn_matmuls:
            if ph.name not in seen_in_eqn:
                seen_in_eqn.add(ph.name)
                ordered_args.append(symbols(ph.name, real=True))

        fp = FunctionPrototype(real, f'inner_F{count}', ordered_args)
        body = [Return(new_rhs)]
        fd = FunctionDefinition.from_FunctionPrototype(fp, body)
        _doc = f"{eqn_name}{format_source(getattr(eqn, 'source', None))}"
        code_blocks.append(_with_docstring(
            pycode(fd, fully_qualified_modules=False), _doc))

        precompute_info.append({
            'eqn_name': eqn_name,
            'new_rhs': new_rhs,
            'matmuls': eqn_matmuls,
            'args': ordered_args,
            'matrix_symbols_removed': matrix_symbols_removed,
        })
        count += 1
    return code_blocks, precompute_info
