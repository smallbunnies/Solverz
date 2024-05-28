from __future__ import annotations

import inspect
import os
import re
from typing import Dict

from Solverz.equation.equations import AE as SymAE, FDAE as SymFDAE, DAE as SymDAE
from Solverz.utilities.io import save
from Solverz.code_printer.python.utilities import parse_p, parse_trigger_fun
from Solverz.code_printer.python.module.module_printer import print_F_numba, print_inner_F, print_sub_inner_F, \
    print_J_numba, print_inner_J
from Solverz.equation.equations import Equations as SymEquations
from Solverz.num_api.custom_function import numerical_interface
from Solverz.variable.variables import Vars, combine_Vars


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

    def print_trigger_func_code():
        code_tfuc = dict()
        trigger_func = parse_trigger_fun(eqn)
        # rename trigger_func
        for func_name, func in trigger_func.items():
            name0 = func.__name__
            func_code = inspect.getsource(func).replace(name0, func_name)
            func_code = func_code.replace('np.', '')
            code_tfuc[func_name] = func_code
        return code_tfuc

    code_tfunc_dict = print_trigger_func_code()

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
                 'sub_inner_J': code_sub_inner_J,
                 'tfunc_dict': code_tfunc_dict}
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
    for tfunc in code_dict['tfunc_dict'].values():
        if numba:
            code += '@njit(cache=True)\n'
        code += tfunc
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
    code += 'from numpy import abs\n'
    code += 'from Solverz.num_api.custom_function import *\n'  # import Solverz built-in func
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
