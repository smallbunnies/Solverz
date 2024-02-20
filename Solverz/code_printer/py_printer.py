from __future__ import annotations
from typing import Callable, Union, List, Dict
import builtins
from datetime import datetime
import re
import os

import numpy as np
from sympy.codegen.ast import Assignment, AddAugmentedAssignment
from sympy import pycode, symbols, Function, Symbol, Number as SymNumber, Expr, Number as SymNumber
from sympy.codegen.ast import real, FunctionPrototype, FunctionDefinition, Return, FunctionCall
from sympy.utilities.lambdify import _import, _module_present, _get_namespace
from scipy.sparse import sparray
from numbers import Number

from Solverz.equation.equations import Equations as SymEquations, FDAE as SymFDAE, DAE as SymDAE, AE as SymAE
from Solverz.variable.variables import Vars, TimeVars
from Solverz.equation.param import TimeSeriesParam
from Solverz.sym_algebra.symbols import Var, SolDict, Para, idx, IdxSymBasic
from Solverz.sym_algebra.functions import zeros, CSC_array, Arange
from Solverz.utilities.address import Address
from Solverz.utilities.io import save
from Solverz.variable.variables import combine_Vars
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
        func.update({para_name + '_trigger_func': param.trigger_fun})

    return func


def made_numerical(eqn: SymEquations, *xys, sparse=False, output_code=False):
    """
    factory method of numerical equations
    """
    print(f"Printing numerical codes of {eqn.name}")
    eqn.assign_eqn_var_address(*xys)
    code_F = print_F(eqn)
    code_J = print_J(eqn, sparse)
    custom_func = dict()
    custom_func.update(numerical_interface)
    custom_func.update(parse_trigger_fun(eqn))
    F = Solverzlambdify(code_F, 'F_', modules=[custom_func, 'numpy'])
    J = Solverzlambdify(code_J, 'J_', modules=[custom_func, 'numpy'])
    p = parse_p(eqn)
    print('Complete!')
    if isinstance(eqn, SymAE) and not isinstance(eqn, SymFDAE):
        num_eqn = nAE(F, J, p)
    elif isinstance(eqn, SymFDAE):
        num_eqn = nFDAE(F, J, p, eqn.nstep)
    elif isinstance(eqn, SymDAE):
        num_eqn = nDAE(eqn.M, F, J, p)
    else:
        raise ValueError(f'Unknown equation type {type(eqn)}')
    if output_code:
        return num_eqn, {'F': code_F, 'J': code_J}
    else:
        return num_eqn


def render_modules(eqn: SymEquations, *xys, name, directory=None, numba=False):
    """
    factory method of numerical equations
    """
    print(f"Printing python codes of {eqn.name}...")
    eqn.assign_eqn_var_address(*xys)
    p = parse_p(eqn)
    code_F = print_F_numba(eqn)
    code_inner_F = print_inner_F(eqn)
    code_sub_inner_F = print_sub_inner_F(eqn)
    code_J = print_J_numba(eqn)
    codes = print_inner_J(eqn, *xys)
    code_inner_J = codes['code_inner_J']
    code_sub_inner_J = codes['code_sub_inner_J']
    custom_func = dict()
    custom_func.update(numerical_interface)
    custom_func.update(parse_trigger_fun(eqn))

    print('Complete!')

    eqn_parameter = {}
    if isinstance(eqn, SymAE) and not isinstance(eqn, SymFDAE):
        eqn_type = 'AE'
    elif isinstance(eqn, SymFDAE):
        eqn_type = 'FDAE'
        eqn_parameter.update({'nstep': eqn.nstep})
    elif isinstance(eqn, SymDAE):
        eqn_type = 'DAE'
        eqn_parameter.update({'M': eqn.M})
    else:
        raise ValueError(f'Unknown equation type {type(eqn)}')

    if len(xys) == 1:
        y = xys[0]
    else:
        y = xys[0]
        for arg in xys[1:]:
            y = combine_Vars(y, arg)

    code_dict = {'F': code_F,
                 'inner_F': code_inner_F,
                 'sub_inner_F': code_sub_inner_F,
                 'J': code_J,
                 'inner_J': code_inner_J,
                 'sub_inner_J': code_sub_inner_J}
    eqn_parameter.update({'row': codes['row'], 'col': codes['col'], 'data': codes['data']})
    print(f"Rendering python modules!")
    render_as_modules(name,
                      code_dict,
                      eqn_type,
                      p,
                      eqn_parameter,
                      y,
                      [custom_func, 'numpy'],
                      numba,
                      directory)
    print('Complete!')


def is_valid_python_module_name(module_name):
    pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    return bool(pattern.match(module_name))


def create_python_module(module_name, initiate_code, module_code, dependency_code, auxiliary, directory=None):
    location = module_name if directory is None else directory + '/' + module_name

    # Create the parent directory if it doesn't exist
    os.makedirs(location, exist_ok=True)

    # Create an empty __init__.py file
    init_path = os.path.join(location, "__init__.py")
    with open(init_path, "w") as file:
        file.write(initiate_code)

    # Create the file with the dependency code
    module_path = os.path.join(location, "dependency.py")
    with open(module_path, "w") as file:
        file.write(dependency_code)

    # Create the file with the module code
    module_path = os.path.join(location, "num_func.py")
    with open(module_path, "w") as file:
        file.write(module_code)

    save(auxiliary, f'{location}/param_and_setting.pkl')


def print_init_code(eqn_type: str, module_name, eqn_param):
    code = 'from .num_func import F_, J_\n'
    code += 'from .dependency import setting, p, y\n'
    code += 'import time\n'
    match eqn_type:
        case 'AE':
            code += 'from Solverz.num_api.num_eqn import nAE\n'
            code += 'mdl = nAE(F_, J_, p)\n'
            code_compile = 'mdl.F(y.array, p)\nmdl.J(y.array, p)\n'
        case 'FDAE':
            try:
                nstep = eqn_param['nstep']
            except KeyError as e:
                raise ValueError("Cannot parse nstep attribute for FDAE object printing!")
            code += 'from Solverz.num_api.num_eqn import nFDAE\n'
            code += 'mdl = nFDAE(F_, J_, p, setting["nstep"])\n'
            args_str = ', '.join(['y.array' for i in range(nstep)])
            code_compile = f'mdl.F(0, y.array, p, {args_str})\nmdl.J(0, y.array, p, {args_str})\n'
        case 'DAE':
            code += 'from Solverz.num_api.num_eqn import nDAE\n'
            code += 'mdl = nDAE(setting["M"], F_, J_, p)\n'
            code_compile = 'mdl.F(0, y.array, p)\nmdl.J(0, y.array, p)\n'
        case _:
            raise ValueError(f'Unknown equation type {eqn_type}')
    code += f'print("Compiling model {module_name}...")\n'
    code += f'start = time.perf_counter()\n'
    code += code_compile
    code += f'end = time.perf_counter()\n'
    code += "print(f'Compiling time elapsed: {end-start}s')\n"
    return code


def print_module_code(code_dict: Dict[str, str], numba=False):
    code = 'from .dependency import *\n'
    code += """_data_ = setting["data"]\n"""
    code += """_F_ = zeros_like(y, dtype=float64)"""
    code += '\n\r\n'
    code += code_dict['F']
    code += '\n\r\n'
    if numba:
        code += '@njit(cache=True)\n'
    code += code_dict['inner_F']
    code += '\n\r\n'
    sub_inner_F = code_dict['sub_inner_F']
    for sub_func in sub_inner_F:
        if numba:
            code += '@njit(cache=True)\n'
        code += sub_func
        code += '\n\r\n'
    code += code_dict['J']
    code += '\n\r\n'
    if numba:
        code += '@njit(cache=True)\n'
    code += code_dict['inner_J']
    code += '\n\r\n'
    sub_inner_J = code_dict['sub_inner_J']
    for sub_func in sub_inner_J:
        if numba:
            code += '@njit(cache=True)\n'
        code += sub_func
        code += '\n\r\n'
    return code


def print_dependency_code(modules):
    code = "import os\n"
    code += "current_module_dir = os.path.dirname(os.path.abspath(__file__))\n"
    code += 'from Solverz import load\n'
    code += 'auxiliary = load(f"{current_module_dir}\\\\param_and_setting.pkl")\n'
    code += 'from numpy import *\n'
    code += 'from scipy.sparse import *\n'
    code += 'from numba import njit\n'
    code += 'setting = auxiliary["eqn_param"]\n'
    code += 'row = setting["row"]\n'
    code += 'col = setting["col"]\n'
    code += 'data_ = setting["data"]\n'
    code += 'p = auxiliary["p"]\n'
    code += 'y = auxiliary["vars"]\n'
    return code


def render_as_modules(name, code_dict: Dict[str, str], eqn_type: str, p: dict, eqn_param: dict, variables: Vars,
                      modules=None, numba=False, directory=None):
    if is_valid_python_module_name(name):
        module_name = name
    else:
        raise ValueError("Invalid python module name!")

    init_code = print_init_code(eqn_type, module_name, eqn_param)
    module_code = print_module_code(code_dict, numba=numba)
    dependency_code = print_dependency_code(modules)

    create_python_module(module_name,
                         init_code,
                         module_code,
                         dependency_code,
                         {'p': p, 'eqn_param': eqn_param, 'vars': variables},
                         directory)


def parse_name_space(modules=None):
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
    return namespace


def Solverzlambdify(funcstr, funcname, modules=None):
    """Convert a Solverz numerical f/g/F/J evaluation expression into a function that allows for fast
        numeric evaluation.
        """
    namespace = parse_name_space(modules)

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


def _print_var_parser(var_name, suffix: str, var_address):
    y = Var('y_' + suffix, internal_use=True)
    if suffix == '':
        return Assignment(Var(var_name + suffix), y[var_address])
    else:
        return Assignment(Var(var_name + '_tag_' + suffix), y[var_address])
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


def _print_F_assignment(eqn_address, eqn, i):
    F_ = Var('_F_', internal_use=True)
    return Assignment(F_[eqn_address], FunctionCall(f'inner_F{int(i)}', [name for name in eqn.SYMBOLS.keys()]))


def print_eqn_assignment(ae: SymEquations, numba_printer=False):
    _F_ = Var('_F_', internal_use=True)
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
            _F_ = Var('_F_', internal_use=True)
            eqn_declaration.append(Assignment(_F_[eqn_address],
                                              FunctionCall(f'inner_F{int(count)}',
                                                           [name for name in eqn.SYMBOLS.keys()])))
            count = count + 1
        else:
            eqn_declaration.append(Assignment(_F_[eqn_address], eqn.RHS))
    return eqn_declaration


def _parse_jac_eqn_address(eqn_address: slice, derivative_dim, sparse):
    if eqn_address.stop - eqn_address.start == 1:
        if derivative_dim == 1 or derivative_dim == 0:
            eqn_address = eqn_address.start if not sparse else SolList(eqn_address.start)
        else:
            if sparse:
                return Var(f'arange({eqn_address.start}, {eqn_address.stop})')[idx('value_coo.row')]
    else:
        if derivative_dim == 1 or derivative_dim == 0:
            eqn_address = Arange(eqn_address.start, eqn_address.stop)
        else:
            if sparse:
                return Var(f'arange({eqn_address.start}, {eqn_address.stop})')[idx('value_coo.row')]
    return eqn_address


def _parse_jac_var_address(var_address_slice: slice, derivative_dim, var_idx, sparse):
    if var_idx is not None:
        try:  # try to simplify the variable address in cases such as [1:10][0,1,2,3]
            temp = np.arange(var_address_slice.start, var_address_slice.stop)[var_idx]
            if isinstance(temp, (Number, SymNumber)):  # number a
                if not sparse:
                    var_address = temp
                else:
                    if derivative_dim == 0 or derivative_dim == 1:
                        var_address = SolList(temp)
                    else:
                        var_address = Var('array([' + str(temp) + '])')[idx('value_coo.col')]
            else:  # np.ndarray
                if np.all(np.diff(temp) == 1):  # list such as [1,2,3,4] can be viewed as slice [1:5]
                    if derivative_dim == 2:
                        if sparse:
                            return Var(f'arange({temp[0]}, {temp[-1] + 1})')[idx('value_coo.col')]
                        else:
                            var_address = slice(temp[0], temp[-1] + 1)
                    elif derivative_dim == 1 or derivative_dim == 0:
                        var_address = Arange(temp[0], temp[-1] + 1)
                else:  # arbitrary list such as [1,3,4,5,6,9]
                    if not sparse or derivative_dim < 2:
                        var_address = SolList(*temp)
                    else:
                        var_address = Var('array([' + ','.join([str(ele) for ele in temp]) + '])')[idx('value_coo.col')]
        except (TypeError, IndexError):
            if isinstance(var_idx, str):
                var_idx = idx(var_idx)
            if not sparse or derivative_dim < 2:
                var_address = Var(f"arange({var_address_slice.start}, {var_address_slice.stop})")[var_idx]
            else:
                var_address = Var(f"arange({var_address_slice.start}, {var_address_slice.stop})")[
                    var_idx[idx('value_coo.col')]]
    else:
        if derivative_dim == 2:
            if sparse:
                return Var(f'arange({var_address_slice.start}, {var_address_slice.stop})')[idx('value_coo.col')]
            else:
                var_address = var_address_slice
        elif derivative_dim == 1 or derivative_dim == 0:
            var_address = Arange(var_address_slice.start, var_address_slice.stop)
    return var_address


def _parse_jac_data(data_length, derivative_dim: int, rhs: Union[Expr, Number, SymNumber], rhs_v_type='array'):
    if derivative_dim == 2:
        return Var('value_coo.data')
    elif derivative_dim == 1:
        return rhs
    elif derivative_dim == 0:
        if rhs_v_type == 'Number':  # if rhs is a number, then return length*[rhs]
            return data_length * SolList(rhs)
        elif rhs_v_type == 'array':  # if rhs produces np.ndarray then return length*rhs.tolist()
            return data_length * tolist(rhs)


def print_J_block(eqn_address_slice, var_address_slice, derivative_dim, var_idx, rhs, sparse,
                  rhs_v_dtpe='array') -> List:
    if sparse:
        eqn_address = _parse_jac_eqn_address(eqn_address_slice,
                                             derivative_dim,
                                             True)
        var_address = _parse_jac_var_address(var_address_slice,
                                             derivative_dim,
                                             var_idx,
                                             True)
        # assign elements to sparse matrix can not be easily broadcast, so we have to parse the data
        data = _parse_jac_data(eqn_address_slice.stop - eqn_address_slice.start,
                               derivative_dim,
                               rhs,
                               rhs_v_dtpe)
        if derivative_dim < 2:
            return [extend(Var('row', internal_use=True), eqn_address),
                    extend(Var('col', internal_use=True), var_address),
                    extend(Var('data', internal_use=True), data)]
        else:
            return [Assignment(Var('value_coo'), coo_array(rhs)),
                    extend(Var('row', internal_use=True), eqn_address),
                    extend(Var('col', internal_use=True), var_address),
                    extend(Var('data', internal_use=True), data)]
    else:
        eqn_address = _parse_jac_eqn_address(eqn_address_slice,
                                             derivative_dim,
                                             False)
        var_address = _parse_jac_var_address(var_address_slice,
                                             derivative_dim,
                                             var_idx,
                                             False)
        return [AddAugmentedAssignment(Var('J_', internal_use=True)[eqn_address, var_address], rhs)]


def print_J_dense(ae: SymEquations):
    eqn_declaration = []
    for eqn_name, eqn in ae.EQNs.items():
        eqn_address_slice = ae.a[eqn_name]
        for var_name, eqndiff in eqn.derivatives.items():
            derivative_dim = eqndiff.dim
            if derivative_dim < 0:
                raise ValueError("Derivative dimension not assigned")
            var_address_slice = ae.var_address[eqndiff.diff_var_name]
            var_idx = eqndiff.var_idx
            rhs = eqndiff.RHS
            eqn_declaration.extend(print_J_block(eqn_address_slice,
                                                 var_address_slice,
                                                 derivative_dim,
                                                 var_idx,
                                                 rhs,
                                                 False))
    return eqn_declaration


def print_J_sparse(ae: SymEquations):
    eqn_declaration = []
    for eqn_name, eqn in ae.EQNs.items():
        eqn_address_slice = ae.a[eqn_name]
        for var_name, eqndiff in eqn.derivatives.items():
            derivative_dim = eqndiff.dim
            if derivative_dim < 0:
                raise ValueError("Derivative dimension not assigned")
            var_address_slice = ae.var_address[eqndiff.diff_var_name]
            var_idx = eqndiff.var_idx
            rhs = eqndiff.RHS
            rhs_v_type = eqndiff.v_type
            eqn_declaration.extend(print_J_block(eqn_address_slice,
                                                 var_address_slice,
                                                 derivative_dim,
                                                 var_idx,
                                                 rhs,
                                                 True,
                                                 rhs_v_type))
    return eqn_declaration


def _print_trigger_func(para_name, trigger_var: List[str]):
    return Assignment(Para(para_name),
                      FunctionCall(para_name + '_trigger_func', tuple(trigger_var)))


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


def print_F(ae: SymEquations):
    fp = print_func_prototype(ae, 'F_')
    body = []
    body.extend(print_var(ae))
    body.extend(print_param(ae))
    body.extend(print_trigger(ae))
    body.extend(print_eqn_assignment(ae))
    temp = Var('_F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_J(ae: SymEquations, sparse=False):
    fp = print_func_prototype(ae, 'J_')
    # initialize temp
    temp = Var('J_', internal_use=True)
    body = list()
    body.extend(print_var(ae))
    body.extend(print_param(ae))
    body.extend(print_trigger(ae))
    if not sparse:
        body.append(Assignment(temp, zeros(ae.eqn_size, ae.vsize)))
        body.extend(print_J_dense(ae))
        body.append(Return(temp))
    else:
        body.extend([Assignment(Var('row', internal_use=True), SolList()),
                     Assignment(Var('col', internal_use=True), SolList()),
                     Assignment(Var('data', internal_use=True), SolList())])
        body.extend(print_J_sparse(ae))
        body.append(Return(coo_2_csc(ae)))
    Jd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(Jd, fully_qualified_modules=False)


def print_J_numba(ae: SymEquations):
    fp = print_func_prototype(ae, 'J_')
    body = []
    var_assignments, var_list = print_var(ae, numba_printer=True)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(ae, numba_printer=True)
    body.extend(param_assignments)
    body.extend(print_trigger(ae))
    body.extend([Assignment(Var('data', internal_use=True),
                            FunctionCall('inner_J', [symbols('_data_', real=True)] + [arg.name for arg in
                                                                                      var_list + param_list]))])
    body.extend([Return(coo_2_csc(ae))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_J(ae: SymEquations, *xys):
    var_assignments, var_list = print_var(ae, numba_printer=True)
    param_assignments, param_list = print_param(ae, numba_printer=True)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_J', [symbols('_data_', real=True)] + args)
    body = []
    row, col, jac_address = parse_jac_address(ae, *xys)
    data = np.zeros_like(row, dtype=np.float64)
    # temp = Var('data_', internal_use=True)
    # body.extend([Assignment(temp, zeros(jac_address.total_size, ))])
    code_sub_inner_J_blocks = []
    count = 0
    for jac_ in jac_address.object_list:
        eqn_name, var_name = jac_.split("@@@")
        derivative = ae.EQNs[eqn_name].derivatives[var_name]
        rhs = derivative.RHS
        if isinstance(rhs, (Number, SymNumber)):
            data[jac_address[jac_]] = rhs
        else:
            body.append(Assignment(Var('_data_', internal_use=True)[jac_address[jac_]],
                                   FunctionCall(f'inner_J{int(count)}', [name for name in derivative.SYMBOLS.keys()])))
            args1 = []
            for var in derivative.SYMBOLS.keys():
                args1.append(symbols(var, real=True))

            fp1 = FunctionPrototype(real, f'inner_J{int(count)}', args1)
            body1 = [Return(rhs)]
            fd1 = FunctionDefinition.from_FunctionPrototype(fp1, body1)
            code_sub_inner_J_blocks.append(pycode(fd1, fully_qualified_modules=False))
            count += 1
    temp = Var('_data_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return {'code_inner_J': pycode(fd, fully_qualified_modules=False),
            'code_sub_inner_J': code_sub_inner_J_blocks,
            'row': row,
            'col': col,
            'data': data}


def _print_inner_J_block_assignment(rhs, address: slice):
    data = Var('data_', internal_use=True)
    return Assignment(data[address], rhs)


def print_F_numba(ae: SymEquations):
    fp = print_func_prototype(ae, 'F_')
    body = []
    var_assignments, var_list = print_var(ae, numba_printer=True)
    body.extend(var_assignments)
    param_assignments, param_list = print_param(ae, numba_printer=True)
    body.extend(param_assignments)
    body.extend(print_trigger(ae))
    body.extend(
        [Return(FunctionCall('inner_F', [symbols('_F_', real=True)] + [arg.name for arg in var_list + param_list]))])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_inner_F(ae: SymEquations):
    var_assignments, var_list = print_var(ae, numba_printer=True)
    param_assignments, param_list = print_param(ae, numba_printer=True)
    args = []
    for var in var_list + param_list:
        args.append(symbols(var.name, real=True))
    fp = FunctionPrototype(real, 'inner_F', [symbols('_F_', real=True)] + args)
    body = []
    body.extend(print_eqn_assignment(ae, True))
    temp = Var('_F_', internal_use=True)
    body.extend([Return(temp)])
    fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    return pycode(fd, fully_qualified_modules=False)


def print_sub_inner_F(ae: SymEquations):
    code_blocks = []
    count = 0
    for eqn_name in ae.EQNs.keys():
        eqn = ae.EQNs[eqn_name]
        args = []
        for var in eqn.SYMBOLS.keys():
            args.append(symbols(var, real=True))
        fp = FunctionPrototype(real, f'inner_F{count}', args)
        body = [Return(eqn.RHS)]
        fd = FunctionDefinition.from_FunctionPrototype(fp, body)
        count = count + 1
        code_blocks.append(pycode(fd, fully_qualified_modules=False))
    return code_blocks


def parse_jac_address(eqns: SymEquations, *xys):
    if isinstance(eqns, SymAE):
        gy = eqns.g_y(*xys)
    elif isinstance(eqns, SymDAE):
        gy = eqns.f_xy(0, *xys)
        if len(eqns.g_list) > 0:
            gy.extend(eqns.g_xy(0, *xys))
    else:
        raise NotImplementedError(f"Unknown equation type {type(eqns)}")

    row = np.array([], dtype=int)
    col = np.array([], dtype=int)
    jac_block_address = Address()
    for gy_tuple in gy:
        eqn_name = gy_tuple[0]
        # eqn_diffs: Dict[str, EqnDiff] = eqns.EQNs[eqn_name].derivatives
        var_name = gy_tuple[1]
        eqndiff = gy_tuple[2]
        diff_var = eqndiff.diff_var
        diff_var_name = eqndiff.diff_var.name
        value = gy_tuple[3]
        eqn_address = eqns.a[eqn_name]
        var_address = eqns.var_address[var_name]
        if isinstance(value, (np.ndarray, sparray)):
            if value.ndim == 2:  # matrix
                raise TypeError("Two-dimensional array not applicable for numba printer!\n  Try rewrite the Equations!")
            elif value.ndim == 1 and value.shape[0] != 1:  # vector
                num_jac_element = value.shape[0]
                if num_jac_element != eqn_address.stop - eqn_address.start:
                    raise ValueError("Number of jac block elements not compatible with equation length!")
            elif value.ndim == 1 and value.shape[0] == 1:  # scalar in np.ndarray for example array([0.0])
                num_jac_element = eqn_address.stop - eqn_address.start
            else:
                raise ValueError("Unknown derivative value dimension type!")
        elif isinstance(value, (Number, SymNumber)):
            num_jac_element = eqn_address.stop - eqn_address.start
        else:
            raise ValueError(f"Unknown derivative data type {type(value)}!")
        eqn_address_range = np.arange(eqn_address.start, eqn_address.stop)
        row = np.append(row, eqn_address_range)
        if isinstance(diff_var, IdxSymBasic):
            index = diff_var.index
            if not isinstance(index, (int, slice, np.integer)):
                raise TypeError(
                    f"Index type {type(diff_var.index)} not applicable for numba printer!\n Try rewrite the Variable!")
            else:
                # reshape is to convert float/integer to 1-dim numpy.ndarray
                var_address_range = np.array(np.arange(var_address.start, var_address.stop)[index]).reshape((-1,))
        elif isinstance(diff_var, Var):
            var_address_range = np.arange(var_address.start, var_address.stop)
        if len(var_address_range) != len(eqn_address_range):
            raise ValueError('Equation address range is different from variable address range')
        col = np.append(col, var_address_range)
        jac_block_address.add(f'{eqn_name}@@@{diff_var_name}', num_jac_element)

    assert len(row) == len(col) == jac_block_address.total_size

    return row, col, jac_block_address


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


class SolList(Function):

    # @classmethod
    # def eval(cls, *args):
    #     if any([not isinstance(arg, Number) for arg in args]):
    #         raise ValueError(f"Solverz' list object accepts only number inputs.")

    def _numpycode(self, printer, **kwargs):
        return r'[' + ', '.join([printer._print(arg) for arg in self.args]) + r']'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class Array(Function):

    def _numpycode(self, printer, **kwargs):
        return r'array([' + ', '.join([printer._print(arg) for arg in self.args]) + r'])'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)


class tolist(Function):

    @classmethod
    def eval(cls, *args):
        if len(args) != 1:
            raise ValueError(f"Solverz' tolist function accepts only one input.")

    def _numpycode(self, printer, **kwargs):
        return r'((' + printer._print(self.args[0]) + r').tolist())'

    def _pythoncode(self, printer, **kwargs):
        return self._numpycode(printer, **kwargs)
