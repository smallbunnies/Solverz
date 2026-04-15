from __future__ import annotations

import warnings
from typing import Union, List, Dict, Callable
import builtins
from datetime import datetime

import numpy as np
from sympy.codegen.ast import Assignment, AddAugmentedAssignment
from sympy import pycode, symbols, Function, Symbol, Expr, Number as SymNumber
from sympy.codegen.ast import real, FunctionPrototype, FunctionDefinition, Return, FunctionCall as SymFuncCall
from sympy.utilities.lambdify import _import, _module_present, _get_namespace
from scipy.sparse import sparray, csc_array
from numbers import Number

from Solverz.equation.eqn import Eqn, LoopEqn
from Solverz.equation.equations import Equations as SymEquations, FDAE as SymFDAE, DAE as SymDAE, AE as SymAE
from Solverz.equation.jac import JacBlock, Jac
from Solverz.equation.hvp import Hvp
from Solverz.equation.param import TimeSeriesParam, ParamBase
from Solverz.sym_algebra.symbols import *
from Solverz.sym_algebra.functions import *
from Solverz.utilities.address import Address
from Solverz.utilities.type_checker import is_number
from Solverz.num_api.num_eqn import nAE, nFDAE, nDAE


# %%

def FunctionCall(name, args):
    if not isinstance(args, list):
        raise TypeError("args should be a list.")
    for i in range(len(args)):
        if isinstance(args[i], str):
            raise ValueError("""
            The `args` parameter passed to sympy.codegen.ast.FunctionCall should not contain str, which may cause sympy parsing error. 
            For example, the sympy.codegen.ast.FunctionCall parses str E in args to math.e! Please use sympy.Symbol objects instead.
            """)
    return SymFuncCall(name, args)


def parse_p(PARAM: Dict[str, ParamBase]):
    p = dict()
    for param_name, param in PARAM.items():
        if isinstance(param, TimeSeriesParam):
            p.update({param_name: param})
        else:
            p.update({param_name: param.v})
    return p


def parse_trigger_func(PARAM: Dict[str, ParamBase]) -> Dict[str, Callable]:
    func = dict()
    for para_name, param in PARAM.items():
        if param.trigger_fun is not None:
            func.update({para_name + '_trigger_func': param.trigger_fun})

    return func


def _print_var_parser(var_name, suffix: str, var_addr_slice):
    y = iVar('y_' + suffix, internal_use=True)
    if suffix == '':
        return Assignment(iVar(var_name + suffix), y[var_addr_slice])
    else:
        return Assignment(iVar(var_name + '_tag_' + suffix), y[var_addr_slice])
    pass


def print_var(var_addr: Address, nstep):
    var_declaration = []
    suffix = [''] + [f'{i}' for i in range(nstep)]
    var_list = []
    for i in suffix:
        for var_name in var_addr.v.keys():
            var_addr_slice = var_addr[var_name]
            var_assign = _print_var_parser(var_name, i, var_addr_slice)
            var_declaration.append(var_assign)
            var_list.append(var_assign.lhs)
    return var_declaration, var_list


def print_param(PARAM: Dict[str, ParamBase],
                output_sparse=False,
                include_sparse_in_list=False):
    """
    output_sparse : whether to print the sparse parameter
    include_sparse_in_list : if True, also add sparse dim=2 params to param_list
        (needed when equations use Mat_Mul, so the sparse matrix is passed to inner functions)
    """
    param_declaration = []
    p = SolDict('p_')
    param_list = []
    for param_name, param in PARAM.items():
        if not param.is_alias:
            if isinstance(param, TimeSeriesParam):
                param_assign = Assignment(Para(param_name),
                                          FunctionCall(f'p_["{param_name}"].get_v_t', [Symbol('t')]))
                param_declaration.append(param_assign)
                param_list.append(param_assign.lhs)
            else:
                if not param.sparse or output_sparse or param.triggerable:
                    param_assign = Assignment(Para(param_name),
                                              p[param_name])
                    param_declaration.append(param_assign)
                    param_list.append(param_assign.lhs)
                elif param.sparse and param.dim == 2:
                    # Load sparse matrix param in wrapper for Mat_Mul (A @ x)
                    param_assign = Assignment(Para(param_name),
                                              p[param_name])
                    param_declaration.append(param_assign)
                    if include_sparse_in_list:
                        param_list.append(param_assign.lhs)

    return param_declaration, param_list


def _print_trigger_func(para_name, trigger_var: List[str]):
    return Assignment(Para(para_name),
                      FunctionCall(para_name + '_trigger_func', [symbols(var) for var in trigger_var]))


def print_trigger(PARAM: Dict[str, ParamBase]):
    trigger_declaration = []
    for name, param in PARAM.items():
        if param.triggerable:
            trigger_declaration.append(_print_trigger_func(name,
                                                           param.trigger_var))
    return trigger_declaration


def print_F_J_prototype(eqs_type: str, func_name: str, nstep=0):
    if func_name not in ['F_', 'J_']:
        raise ValueError(f"Func name {func_name} not supported!")
    t, y_, p_ = symbols('t y_ p_', real=True)
    if eqs_type == 'DAE' or eqs_type == 'FDAE':
        args = [t, y_, p_]
    elif eqs_type == 'AE':
        args = [y_, p_]
    else:
        raise TypeError(f"Unknown equation type {eqs_type}")
    xtra_args = []

    if eqs_type == 'FDAE' and nstep == 0:
        raise ValueError("nstep of FDAE should be greater than 0!")
    elif eqs_type == 'FDAE' and nstep > 0:
        xtra_args.extend(
            [symbols('y_' + f'{i}', real=True) for i in range(nstep)])
    elif eqs_type != 'FDAE' and nstep > 0:
        raise TypeError(
            f"Only FDAE supports the case where nstep > 0! Not {eqs_type}")
    else:  # eqs_type != 'FDAE' and nstep == 0
        pass

    fp = FunctionPrototype(real, func_name, args + xtra_args)
    return fp


def print_Hvp_prototype(eqs_type: str, func_name: str = 'Hvp_', nstep=0):
    t, y_, p_, v_ = symbols('t y_ p_ v_', real=True)
    if eqs_type == 'DAE' or eqs_type == 'FDAE':
        args = [t, y_, p_, v_]
    elif eqs_type == 'AE':
        args = [y_, p_, v_]
    else:
        raise TypeError(f"Unknown equation type {eqs_type}")
    xtra_args = []

    if eqs_type == 'FDAE' and nstep == 0:
        raise ValueError("nstep of FDAE should be greater than 0!")
    elif eqs_type == 'FDAE' and nstep > 0:
        xtra_args.extend(
            [symbols('y_' + f'{i}', real=True) for i in range(nstep)])
    elif eqs_type != 'FDAE' and nstep > 0:
        raise TypeError(
            f"Only FDAE supports the case where nstep > 0! Not {eqs_type}")
    else:  # eqs_type != 'FDAE' and nstep == 0
        pass

    fp = FunctionPrototype(real, func_name, args + xtra_args)
    return fp


def extract_matmuls(expr):
    """Extract Mat_Mul nodes from an expression and replace with placeholders.

    Walks the expression tree, finds all Mat_Mul nodes, and replaces them with
    iVar placeholders (_sz_mm_0, _sz_mm_1, ...). Handles nested Mat_Mul correctly —
    inner Mat_Mul is extracted first, then the outer one uses the placeholder.

    Short-circuit: if the expression contains no Mat_Mul, returns
    ``(expr, [])`` unchanged without tree reconstruction, so non-matrix
    equations are untouched.

    Parameters
    ----------
    expr : sympy.Expr
        Input expression that may contain Mat_Mul nodes.

    Returns
    -------
    new_expr : sympy.Expr
        Expression with Mat_Mul nodes replaced by iVar placeholders.
    matmul_list : list of tuple
        List of (placeholder_iVar, matrix_arg, operand_arg) triples. To
        precompute: ``placeholder = matrix_arg @ operand_arg``.
    """
    # Fast path: no Mat_Mul means no transformation at all — preserves
    # original expression and avoids reconstruction side effects.
    if not (hasattr(expr, 'has') and expr.has(Mat_Mul)):
        return expr, []

    matmul_list = []
    counter = [0]

    def walk(e):
        if isinstance(e, Mat_Mul):
            # Recurse into args first (handle nested Mat_Mul)
            new_args = [walk(a) for a in e.args]
            name = f'_sz_mm_{counter[0]}'
            counter[0] += 1
            placeholder = iVar(name, internal_use=True)
            if len(new_args) == 2:
                matrix_arg = new_args[0]
                operand_arg = new_args[1]
            else:
                # Mat_Mul(A, B, C, ...): fold into A @ (B @ C @ ...)
                matrix_arg = new_args[0]
                operand_arg = Mat_Mul(*new_args[1:])
            matmul_list.append((placeholder, matrix_arg, operand_arg))
            return placeholder
        if hasattr(e, 'args') and len(e.args) > 0 and not e.is_Atom:
            new_args = [walk(a) for a in e.args]
            try:
                return e.func(*new_args)
            except Exception:
                return e
        return e

    new_expr = walk(expr)
    return new_expr, matmul_list


def print_eqn_assignment(EQNs: Dict[str, Eqn],
                         EqnAddr: Address,
                         module_printer=False):
    _F_ = iVar('_F_', internal_use=True)
    eqn_declaration = []
    if not module_printer:
        eqn_declaration.append(Assignment(_F_, zeros(EqnAddr.total_size, )))
        # Whereas in module printer, the _F_ is predeclared for only once.
    count = 0
    for eqn_name in EQNs.keys():
        eqn_address = EqnAddr[eqn_name]
        eqn = EQNs[eqn_name]
        if module_printer:
            for var in list(eqn.expr.free_symbols):
                if isinstance(var, IdxSymBasic):
                    if isinstance(var.index, (idx, list)):
                        raise ValueError(
                            f"Numba printer does not accept idx or list index {var.index}")
            _F_ = iVar('_F_', internal_use=True)
            # ``LoopEqn`` with sparse walkers prints its ``inner_F<N>``
            # with the sparse Params EXCLUDED from the signature (their
            # CSR arrays ride in as module-level constants). The call
            # site must match the pruned signature exactly — otherwise
            # the inner function receives a positional csc_array that
            # numba cannot handle. For non-LoopEqn equations and dense
            # LoopEqns, this is a no-op (``njit_arg_names`` falls back
            # to all SYMBOLS).
            if isinstance(eqn, LoopEqn):
                call_args = [eqn.SYMBOLS[nm]
                             for nm in eqn.njit_arg_names()]
            else:
                call_args = list(eqn.SYMBOLS.values())
            eqn_declaration.append(Assignment(_F_[eqn_address],
                                              FunctionCall(f'inner_F{int(count)}', call_args)))

            count = count + 1
        else:
            if isinstance(eqn, LoopEqn):
                # ``LoopEqn`` bodies use sympy ``Sum`` over ``Idx`` which
                # the standard ``NumPyPrinter`` cannot translate (raises
                # ``PrintMethodNotImplementedError`` on ``Idx``). Route
                # through a registered custom callable instead — the
                # lambdify modules dict is augmented with
                # ``_sz_loop_<eqn_name>`` → ``eqn.NUM_EQN`` over in
                # ``made_numerical``.
                eqn_declaration.append(Assignment(
                    _F_[eqn_address],
                    FunctionCall(f'_sz_loop_{eqn_name}',
                                 list(eqn.SYMBOLS.values()))
                ))
            else:
                eqn_declaration.append(Assignment(_F_[eqn_address], eqn.RHS))
    return eqn_declaration




