from __future__ import annotations

import numpy as np
from sympy import Function
from Solverz.code_printer.python.utilities import *
from Solverz.code_printer.python.module.mutable_mat_analyzer import (
    analyze_mutable_mat_expr,
    generate_block_function_code,
    MutableMatBlockMapping,
)


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
                # (b) Build the call argument list in the order the block
                # function expects.
                call_args = []
                for arg_name in mb['diag_arg_names']:
                    call_args.append(symbols(arg_name, real=True))
                for arg_name in mb['rs_arg_names']:
                    call_args.append(symbols(arg_name, real=True))
                for arg_name in mb['cs_arg_names']:
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
                body.append(Assignment(
                    iVar('data', internal_use=True)[mb['addr_slice']],
                    FunctionCall(mb['fn_name'], call_args)))
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
                  nstep: int = 0):
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
                        block_info['fn_name'] = fn_name
                        block_info['diag_arg_names'] = diag_arg_names
                        block_info['rs_arg_names'] = rs_arg_names
                        block_info['cs_arg_names'] = cs_arg_names
                        block_code = generate_block_function_code(
                            fn_name, mapping,
                            diag_arg_names, rs_arg_names, cs_arg_names)
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
                code_sub_inner_J_blocks.append(pycode(fd1, fully_qualified_modules=False))
                count += 1
            addr_by_ele_0 += jb.SpEleSize
    temp = iVar('_data_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return {'code_inner_J': pycode(fd, fully_qualified_modules=False),
            'code_sub_inner_J': code_sub_inner_J_blocks,
            'no_njit_sub_inner_J': no_njit_indices,
            'mutable_matrix_blocks': mutable_matrix_blocks,
            'mut_mat_block_funcs': mut_mat_block_funcs,
            'mut_mat_mappings': mut_mat_mappings}


def _is_csc_matvec_fast_path(matrix_arg, PARAM):
    """Whether a ``Mat_Mul(matrix_arg, operand)`` placeholder can be
    computed by the Numba ``SolCF.csc_matvec`` helper inside
    ``inner_F`` instead of a scipy SpMV in the Python ``F_`` wrapper.

    The condition is strict: the matrix must be a bare ``Para`` (no
    ``Mul(-1, ...)``, no nested expression) backed by a
    ``Param(dim=2, sparse=True)`` in the equation's ``PARAM`` dict.
    Those matrices already have their CSC decomposition
    (``_data``/``_indices``/``_indptr``/``_shape0``) emitted into
    ``inner_F``'s argument list by ``print_param``, so the fast path
    just assembles the matvec call.

    Anything else (negated matrices, matrix-matrix products,
    operands that are themselves placeholder iVars, dense dim=2
    params) falls back to the existing scipy SpMV path in ``F_``.
    """
    if not isinstance(matrix_arg, Para):
        return False
    if matrix_arg.name not in PARAM:
        return False
    p = PARAM[matrix_arg.name]
    return getattr(p, 'dim', 0) == 2 and getattr(p, 'sparse', False)


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

    # Identify sparse dim=2 Paras that are *only* used as fast-path
    # ``Mat_Mul`` matrix operands. These don't need to be loaded in
    # the wrapper at all — their CSC decomposition
    # (``<name>_data``/``_indices``/``_indptr``/``_shape0``) already
    # flows through ``param_list`` to ``inner_F`` and the fast-path
    # matvec runs entirely inside ``inner_F``. Loading the whole
    # sparse matrix in the wrapper would be dead work.
    fallback_mat_symbols = set()
    if precompute_info:
        for eqn_info in precompute_info:
            for _, matrix_arg, _ in eqn_info['matmuls']:
                if _is_csc_matvec_fast_path(matrix_arg, PARAM):
                    continue
                # Fallback case: any Para shown inside the fallback
                # matrix_arg is still needed in the wrapper for the
                # scipy SpMV to execute correctly.
                if hasattr(matrix_arg, 'free_symbols'):
                    for s in matrix_arg.free_symbols:
                        if isinstance(s, Para):
                            fallback_mat_symbols.add(s.name)

    # Filter param_assignments: drop sparse dim=2 loads whose matrix
    # is never referenced outside fast-path Mat_Mul.
    filtered_assignments = []
    for assign in param_assignments:
        lhs_name = getattr(assign.lhs, 'name', None)
        if lhs_name and lhs_name in PARAM:
            p_obj = PARAM[lhs_name]
            if (getattr(p_obj, 'dim', 0) == 2
                    and getattr(p_obj, 'sparse', False)
                    and lhs_name not in fallback_mat_symbols):
                continue  # fast-path only, no need to load in wrapper
        filtered_assignments.append(assign)
    body.extend(filtered_assignments)
    body.extend(print_trigger(PARAM))

    # Generate precompute assignments for fallback placeholders only:
    # ``_sz_mm_N = matrix_arg @ operand_arg``. Fast-path placeholders
    # are handled inside ``inner_F`` via ``SolCF.csc_matvec`` and do
    # not need a wrapper-level precompute.
    inner_extra_args = []
    if precompute_info:
        seen_placeholders = set()
        for eqn_info in precompute_info:
            for placeholder, matrix_arg, operand_arg in eqn_info['matmuls']:
                if placeholder.name in seen_placeholders:
                    continue
                seen_placeholders.add(placeholder.name)
                if _is_csc_matvec_fast_path(matrix_arg, PARAM):
                    continue
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

    # Classify each placeholder as fast-path (handled inline via
    # SolCF.csc_matvec) vs fallback (passed as arg from the wrapper).
    fast_path_matvec = []  # list of (placeholder_iVar, matrix_name, operand_expr)
    fallback_placeholders = []  # list of iVar
    if precompute_info:
        seen = set()
        for eqn_info in precompute_info:
            for placeholder, matrix_arg, operand_arg in eqn_info['matmuls']:
                if placeholder.name in seen:
                    continue
                seen.add(placeholder.name)
                if _is_csc_matvec_fast_path(matrix_arg, PARAM):
                    fast_path_matvec.append(
                        (placeholder, matrix_arg.name, operand_arg))
                else:
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
        # Fast path: non-Mat_Mul equation — original behavior, no changes.
        if not eqn.mixed_matrix_vector:
            args = [symbols(v, real=True) for v in eqn.SYMBOLS.keys()]
            fp = FunctionPrototype(real, f'inner_F{count}', args)
            body = [Return(eqn.RHS)]
            fd = FunctionDefinition.from_FunctionPrototype(fp, body)
            code_blocks.append(pycode(fd, fully_qualified_modules=False))
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
        code_blocks.append(pycode(fd, fully_qualified_modules=False))

        precompute_info.append({
            'eqn_name': eqn_name,
            'new_rhs': new_rhs,
            'matmuls': eqn_matmuls,
            'args': ordered_args,
            'matrix_symbols_removed': matrix_symbols_removed,
        })
        count += 1
    return code_blocks, precompute_info
