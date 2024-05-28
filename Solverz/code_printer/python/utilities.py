from __future__ import annotations

import warnings
from typing import Union, List
import builtins
from datetime import datetime

import numpy as np
from sympy.codegen.ast import Assignment, AddAugmentedAssignment
from sympy import pycode, symbols, Function, Symbol, Expr, Number as SymNumber
from sympy.codegen.ast import real, FunctionPrototype, FunctionDefinition, Return, FunctionCall
from sympy.utilities.lambdify import _import, _module_present, _get_namespace
from scipy.sparse import sparray
from numbers import Number

from Solverz.equation.equations import Equations as SymEquations, FDAE as SymFDAE, DAE as SymDAE, AE as SymAE
from Solverz.equation.param import TimeSeriesParam
from Solverz.sym_algebra.symbols import iVar, SolDict, Para, idx, IdxSymBasic
from Solverz.sym_algebra.functions import zeros, Arange
from Solverz.utilities.address import Address
from Solverz.num_api.custom_function import numerical_interface
from Solverz.num_api.num_eqn import nAE, nFDAE, nDAE


# %%


def parse_p(ae: SymEquations):
    p = dict()
    for param_name, param in ae.PARAM.items():
        if isinstance(param, TimeSeriesParam):
            p.update({param_name: param})
        else:
            p.update({param_name: param.v})
    return p


def parse_trigger_fun(ae: SymEquations):
    func = dict()
    for para_name, param in ae.PARAM.items():
        if param.trigger_fun is not None:
            func.update({para_name + '_trigger_func': param.trigger_fun})

    return func


def _print_var_parser(var_name, suffix: str, var_address):
    y = iVar('y_' + suffix, internal_use=True)
    if suffix == '':
        return Assignment(iVar(var_name + suffix), y[var_address])
    else:
        return Assignment(iVar(var_name + '_tag_' + suffix), y[var_address])
    pass


def print_var(ae: SymEquations, numba_printer=False):
    var_declaration = []
    if hasattr(ae, 'nstep'):
        nstep = ae.nstep
    else:
        nstep = 0
    suffix = [''] + [f'{i}' for i in range(nstep)]
    var_list = []
    for i in suffix:
        for var_name in ae.var_address.v.keys():
            var_address = ae.var_address[var_name]
            var_assign = _print_var_parser(var_name, i, var_address)
            var_declaration.append(var_assign)
            var_list.append(var_assign.lhs)
    if not numba_printer:
        return var_declaration
    else:
        return var_declaration, var_list


def print_param(ae: SymEquations, numba_printer=False):
    param_declaration = []
    p = SolDict('p_')
    param_list = []
    for symbol_name, symbol in ae.SYMBOLS.items():
        if isinstance(symbol, (Para, idx)):
            if isinstance(ae.PARAM[symbol_name], TimeSeriesParam):
                param_assign = Assignment(Para(symbol_name), FunctionCall(f'p_["{symbol_name}"].get_v_t', 't'))
            else:
                param_assign = Assignment(Para(symbol_name), p[symbol_name])
            param_declaration.append(param_assign)
            param_list.append(param_assign.lhs)
    if not numba_printer:
        return param_declaration
    else:
        return param_declaration, param_list


def _print_trigger_func(para_name, trigger_var: List[str]):
    return Assignment(Para(para_name),
                      FunctionCall(para_name + '_trigger_func', tuple([symbols(var) for var in trigger_var])))


def print_trigger(ae: SymEquations):
    trigger_declaration = []
    for name, param in ae.PARAM.items():
        if param.triggerable:
            trigger_declaration.append(_print_trigger_func(name,
                                                           param.trigger_var))
    return trigger_declaration


def print_func_prototype(ae: SymEquations, func_name: str):
    t, y_, p_ = symbols('t y_ p_', real=True)
    if isinstance(ae, (SymDAE, SymFDAE)):
        args = [t, y_, p_]
    else:
        args = [y_, p_]
    xtra_args = []
    if hasattr(ae, 'nstep'):
        if ae.nstep > 0:
            xtra_args.extend([symbols('y_' + f'{i}', real=True) for i in range(ae.nstep)])

    fp = FunctionPrototype(real, func_name, args + xtra_args)
    return fp


def print_eqn_assignment(ae: SymEquations, numba_printer=False):
    _F_ = iVar('_F_', internal_use=True)
    eqn_declaration = []
    if not numba_printer:
        eqn_declaration.append(Assignment(_F_, zeros(ae.eqn_size, )))
    count = 0
    for eqn_name in ae.EQNs.keys():
        eqn_address = ae.a[eqn_name]
        eqn = ae.EQNs[eqn_name]
        if numba_printer:
            for var in list(eqn.expr.free_symbols):
                if isinstance(var, IdxSymBasic):
                    if isinstance(var.index, (idx, list)):
                        raise ValueError(f"Numba printer does not accept idx or list index {var.index}")
            _F_ = iVar('_F_', internal_use=True)
            eqn_declaration.append(Assignment(_F_[eqn_address],
                                              FunctionCall(f'inner_F{int(count)}', list(eqn.SYMBOLS.values()))))
            count = count + 1
        else:
            eqn_declaration.append(Assignment(_F_[eqn_address], eqn.RHS))
    return eqn_declaration


class coo_2_csc(Symbol):

    def __new__(cls, ae: SymEquations):
        obj = Symbol.__new__(cls, f'coo_2_csc: {ae.name}')
        obj.eqn_size = ae.eqn_size
        obj.vsize = ae.vsize
        return obj

    def _numpycode(self, printer, **kwargs):
        return f'coo_array((data, (row,col)), ({self.eqn_size}, {self.vsize})).tocsc()'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class coo_array(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            raise ValueError(f"Solverz' coo_array object accepts only one inputs.")

    def _numpycode(self, printer, **kwargs):
        return f'coo_array({printer._print(self.args[0])})'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class extend(Function):

    def _numpycode(self, printer, **kwargs):
        return f'{printer._print(self.args[0])}.extend({printer._print(self.args[1])})'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)



