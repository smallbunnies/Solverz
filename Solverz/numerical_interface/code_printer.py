from __future__ import annotations
from typing import Callable
import builtins
from datetime import datetime

import numpy as np
from sympy.codegen.ast import Assignment, AddAugmentedAssignment, Variable
from sympy import pycode, symbols
from sympy.codegen.ast import real, FunctionPrototype, FunctionDefinition, Return, FunctionCall
from sympy.utilities.lambdify import _import, _imp_namespace, _module_present, _get_namespace

from Solverz.equation.equations import Equations as SymEquations, AE as SymAE, tAE as SymtAE, DAE as SymDAE
from Solverz.equation.eqn import EqnDiff, Eqn
from Solverz.equation.param import TimeSeriesParam
from Solverz.symboli_algebra.symbols import Var, idx, SolDict, Para
from Solverz.symboli_algebra.functions import zeros, CSC_array, Arange
from Solverz.auxiliary_service.address import Address
from Solverz.variable.variables import Vars


# %%
def Solverzlambdify(funcstr, funcname, modules=None):
    """Convert a Solverz numerical f/g/F/J evaluation expression into a function that allows for fast
        numeric evaluation.
        """

    # If the user hasn't specified any modules, use what is available.
    if modules is None:
        try:
            _import("scipy")
        except ImportError:
            try:
                _import("numpy")
            except ImportError:
                # Use either numpy (if available) or python.math where possible.
                # XXX: This leads to different behaviour on different systems and
                #      might be the reason for irreproducible errors.
                modules = ["math", "mpmath", "sympy"]
            else:
                modules = ["numpy"]
        else:
            modules = ["numpy", "scipy"]

    # Get the needed namespaces.
    namespaces = []

    # Check for dict before iterating
    if isinstance(modules, (dict, str)) or not hasattr(modules, '__iter__'):
        namespaces.append(modules)
    else:
        # consistency check
        if _module_present('numexpr', modules) and len(modules) > 1:
            raise TypeError("numexpr must be the only item in 'modules'")
        namespaces += list(modules)
    # fill namespace with first having highest priority
    namespace = {}
    for m in namespaces[::-1]:
        buf = _get_namespace(m)
        namespace.update(buf)

    # Provide lambda expression with builtins, and compatible implementation of range
    namespace.update({'builtins': builtins, 'range': range})

    funclocals = {}
    current_time = datetime.now()
    filename = '<generated at-%s>' % current_time
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)

    func = funclocals[funcname]
    # Apply the docstring
    src_str = funcstr
    # TODO: should collect and show the module imports from the code printers instead of the namespace
    func.__doc__ = (
        "Created with Solverz at \n\n"
        "{sig}\n\n"
        "Source code:\n\n"
        "{src}\n\n"
    ).format(sig=current_time, src=src_str)
    return func


def print_var(ae: SymEquations):
    var_declaration = []
    y = Var('y_', internal_use=True)
    for var_name in ae.var_address.v.keys():
        var_address_range = ae.var_address.v[var_name]
        if len(var_address_range.tolist()) > 1:
            var_address = slice(var_address_range[0], var_address_range[-1])
        else:
            var_address = var_address_range.tolist()
        var_declaration.extend([Assignment(ae.SYMBOLS[var_name], y[var_address])])
    return var_declaration


def print_param(ae: SymEquations):
    param_declaration = []
    p = SolDict('p_')
    for param_name, v in ae.PARAM.items():
        if isinstance(ae.PARAM[param_name], TimeSeriesParam):
            param_declaration.append(Assignment(Para(param_name), FunctionCall(f'p_["{param_name}"].get_v_t', 't')))
        else:
            param_declaration.append(Assignment(Para(param_name), p[param_name]))
    return param_declaration


def print_eqn_assignment(ae: SymEquations):
    temp = Var('F_', internal_use=True)
    eqn_declaration = [Assignment(temp, zeros(ae.eqn_size, ))]
    for eqn_name in ae.EQNs.keys():
        eqn_address_range = ae.a.v[eqn_name]
        if len(eqn_address_range.tolist()) > 1:
            eqn_address = slice(eqn_address_range[0], eqn_address_range[-1])
        else:
            eqn_address = eqn_address_range.tolist()
        eqn_declaration.append(Assignment(temp[eqn_address], ae.EQNs[eqn_name].RHS))
    return eqn_declaration


def print_J_block(ae: SymEquations):
    temp = Var('J_', internal_use=True)
    eqn_declaration = []
    for eqn_name, eqn in ae.EQNs.items():
        eqn_address_range = ae.a.v[eqn_name]
        for var_name, eqndiff in eqn.derivatives.items():
            derivative_dim = eqndiff.dim
            if derivative_dim < 0:
                raise ValueError("Derivative dimension not assigned")
            if len(eqn_address_range.tolist()) > 1:
                if derivative_dim == 2:
                    eqn_address = slice(eqn_address_range[0], eqn_address_range[-1])
                elif derivative_dim == 1 or derivative_dim == 0:
                    eqn_address = Arange(eqn_address_range[0], eqn_address_range[-1] + 1)
                else:
                    raise ValueError("Unsupported derivative dimension!")
            else:
                eqn_address = eqn_address_range.tolist()
            var_address_range = ae.var_address.v[eqndiff.diff_var_name]
            var_idx = eqndiff.var_idx
            if isinstance(var_idx, str):
                var_idx = idx(var_idx)
            if var_idx is not None:
                var_address = Var(f"arange({var_address_range[0]}, {var_address_range[-1] + 1})")[var_idx]
            else:
                if len(var_address_range.tolist()) > 1:
                    if derivative_dim == 2:
                        var_address = slice(var_address_range[0], var_address_range[-1])
                    elif derivative_dim == 1 or derivative_dim == 0:
                        var_address = Arange(var_address_range[0], var_address_range[-1] + 1)
                    else:
                        raise ValueError("Unsupported derivative dimension!")
                else:
                    var_address = var_address_range.tolist()
            # derivative is matrix or vector?
            eqn_declaration.append(AddAugmentedAssignment(temp[eqn_address, var_address], eqndiff.RHS))
    return eqn_declaration


# def print_var_address(ae):
#     v_address_dec = []
#     for var_name, address in ae.var_address.v.items():
#         if len(address) > 1:
#             v_address_dec.append(Variable('a_' + var_name).as_Declaration(value=ae.var_address[var_name]))
#         else:
#             v_address_dec.append(Variable('a_' + var_name).as_Declaration(value=address.tolist()))
#     return v_address_dec


def print_trigger(ae: SymEquations):
    trigger_declaration = []
    for triggered_var, trigger_var in ae.triggerable_quantity.items():
        trigger_declaration.append(
            Assignment(Var(triggered_var), FunctionCall(triggered_var + '_trigger_func', trigger_var)))
    return trigger_declaration


def print_F(ae: SymEquations):
    t, y_, p_ = symbols('t y_ p_', real=True)
    fp = FunctionPrototype(real, 'F_', [t, y_, p_])
    body = []
    body.extend(print_var(ae))
    body.extend(print_param(ae))
    body.extend(print_trigger(ae))
    body.extend(print_eqn_assignment(ae))
    temp = Var('F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd)


def print_J(ae: SymEquations):
    t, y_, p_ = symbols('t y_ p_', real=True)
    fp = FunctionPrototype(real, 'J_', [t, y_, p_])
    # initialize temp
    temp = Var('J_', internal_use=True)
    body = [Assignment(temp, zeros(ae.eqn_size, ae.vsize))]
    body.extend(print_var(ae))
    body.extend(print_param(ae))
    body.extend(print_trigger(ae))
    body.extend(print_J_block(ae))
    body.append(Return(CSC_array(temp)))
    Jd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(Jd)


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
    for triggered_var, trigger_var in ae.triggerable_quantity.items():
        func.update({triggered_var + '_trigger_func': ae.PARAM[triggered_var].trigger_fun})

    return func
