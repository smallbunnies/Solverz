from __future__ import annotations

import inspect
import os
import re
from typing import Dict

from Solverz.equation.equations import AE as SymAE, FDAE as SymFDAE, DAE as SymDAE
from Solverz.equation.jac import JacBlock, Jac
from Solverz.utilities.io import save
from Solverz.code_printer.python.utilities import parse_p, parse_trigger_func
from Solverz.code_printer.python.module.module_printer import print_F, print_inner_F, print_sub_inner_F, \
    print_J, print_inner_J, print_Hvp, print_inner_Hvp
from Solverz.equation.equations import Equations as SymEquations
from Solverz.num_api.module_parser import modules
from Solverz.variable.variables import Vars, combine_Vars
from Solverz.equation.hvp import Hvp


def render_sub_J_block_module(name,
                              eqs: SymEquations,
                              jac: Jac,
                              directory,
                              numba=False):

    code_dict = dict()
    code_dict["J"] = print_J(eqs.__class__.__name__,
                             eqs.eqn_size,
                             eqs.var_address,
                             eqs.PARAM,
                             jac.shape,
                             eqs.nstep)
    J = print_inner_J(eqs.var_address,
                      eqs.PARAM,
                      jac,
                      eqs.nstep)
    code_dict["inner_J"] = J['code_inner_J']
    code_dict["sub_inner_J"] = J['code_sub_inner_J']

    row, col, data = jac.parse_row_col_data()
    data_row_col = dict()
    data_row_col["data"] = data
    data_row_col["row"] = row - jac.coordinate0[0]
    data_row_col["col"] = col - jac.coordinate0[1]


    module_code = print_module_code(name, code_dict, numba=numba)

    location = os.path.join(directory, '')

    # Create the parent directory if it doesn't exist
    # os.makedirs(location, exist_ok=True)

    # # Create an empty __init__.py file
    # init_path = os.path.join(location, "__init__.py")
    # with open(init_path, "w") as file:
    #     file.write(initiate_code)
    #
    # # Create the file with the dependency code
    # module_path = os.path.join(location, "dependency.py")
    # with open(module_path, "w") as file:
    #     file.write(dependency_code)

    # Create the file with the module code
    module_path = os.path.join(location, f"{name}.py")
    with open(module_path, "w") as file:
        file.write(module_code)

    save(data_row_col, os.path.join(location, f"jac_{name}.pkl"))


def print_module_code(name,
                      code_dict: Dict[str, str],
                      numba=False):
    code = 'from .dependency import *\n'
    code += "import os\n"
    code += "current_module_dir = os.path.dirname(os.path.abspath(__file__))\n"
    code += 'from Solverz import load\n'
    code += f'data_row_col = load(os.path.join(current_module_dir, "jac_{name}.pkl"))\n'
    code += """_data_ = data_row_col["data"]\n"""
    code += """row = data_row_col["row"]\n"""
    code += """col = data_row_col["col"]\n"""
    code += '\n\r\n'

    code += code_dict['J']
    code += '\n\r\n'
    if numba:
        code += '@njit(cache=True)\n'
    code += code_dict['inner_J']
    code += '\n\r\n'
    for sub_func in code_dict['sub_inner_J']:
        if numba:
            code += '@njit(cache=True)\n'
        code += sub_func
        code += '\n\r\n'

    return code
