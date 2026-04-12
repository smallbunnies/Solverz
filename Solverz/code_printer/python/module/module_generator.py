from __future__ import annotations

import inspect
import os
import re
from typing import Dict

from Solverz.equation.equations import AE as SymAE, FDAE as SymFDAE, DAE as SymDAE
from Solverz.equation.param import TimeSeriesParam
from Solverz.utilities.io import save
from Solverz.code_printer.python.utilities import parse_p, parse_trigger_func
from Solverz.code_printer.python.module.module_printer import print_F, print_inner_F, print_sub_inner_F, \
    print_J, print_inner_J, print_Hvp, print_inner_Hvp
from Solverz.equation.equations import Equations as SymEquations
from Solverz.num_api.module_parser import modules
from Solverz.variable.variables import Vars, combine_Vars
from Solverz.equation.hvp import Hvp


def _has_sparse_in_param_list(PARAM) -> bool:
    """Return True iff ``print_param`` would emit any sparse parameter
    into the ``param_list`` (the list that's forwarded as arguments to
    ``inner_F`` / ``inner_J``) *in this module's code path*.

    With the Mat_Mul precompute architecture, ``print_F`` / ``print_J``
    call ``print_param`` with ``include_sparse_in_list=False``, so
    plain sparse ``dim=2`` params never enter ``param_list``. They
    still get loaded inside the ``F_`` / ``J_`` wrappers, where scipy
    is available, and only the precomputed dense vectors cross the
    boundary into the inner functions.

    That still leaves two branches in ``print_param`` that unconditionally
    append to ``param_list`` regardless of the flag — and if any
    parameter hits either branch the receiving ``inner_F`` / ``inner_J``
    cannot be decorated with ``@njit`` (Numba rejects scipy.sparse
    arguments at compile time):

    - ``TimeSeriesParam`` with ``sparse=True`` — the time-series branch
      always appends;
    - ``triggerable=True`` with ``sparse=True`` — the ``if not sparse or
      triggerable`` branch appends unconditionally.

    Pure element-wise models with no sparse params at all trivially
    return False and continue to use ``@njit`` on everything.
    """
    for name, param in PARAM.items():
        if getattr(param, 'is_alias', False):
            continue
        if not getattr(param, 'sparse', False):
            continue
        if isinstance(param, TimeSeriesParam):
            return True
        if getattr(param, 'triggerable', False):
            return True
    return False


def render_modules(eqs: SymEquations,
                   y,
                   name,
                   directory=None,
                   numba=False,
                   make_hvp=False):
    """
    factory method of numerical equations
    """
    print(f"Printing python codes of {eqs.name}...")
    eqs.FormJac(y)
    p = parse_p(eqs.PARAM)

    # The Mat_Mul precompute architecture unconditionally forces sparse
    # matrix parameters OUT of the ``inner_F`` / ``inner_J`` argument
    # lists (they are loaded in the F_ / J_ wrappers and only the
    # precomputed dense vectors cross the boundary). See the docstring
    # of ``_has_sparse_in_param_list`` below for the edge cases that
    # still land sparse params in the inner list via ``TimeSeriesParam``
    # or ``triggerable=True`` — those are what drive the @njit gating.

    code_dict = dict()
    # Extract Mat_Mul precompute info first; used by F, inner_F, and sub_inner_F.
    sub_inner_F_codes, precompute_info_F = print_sub_inner_F(eqs.EQNs)
    code_dict["sub_inner_F"] = sub_inner_F_codes
    # Precompute architecture: sub_inner_F no longer receives sparse matrices,
    # so all sub-functions can now be @njit (empty no_njit set).
    code_dict["no_njit_sub_inner_F"] = set()
    code_dict['F'] = print_F(eqs.__class__.__name__,
                             eqs.var_address,
                             eqs.PARAM,
                             eqs.nstep,
                             precompute_info=precompute_info_F)
    code_dict["inner_F"] = print_inner_F(eqs.EQNs,
                                         eqs.a,
                                         eqs.var_address,
                                         eqs.PARAM,
                                         eqs.nstep,
                                         precompute_info=precompute_info_F)

    J = print_inner_J(eqs.var_address,
                      eqs.PARAM,
                      eqs.jac,
                      eqs.nstep)
    code_dict["J"] = print_J(eqs.__class__.__name__,
                             eqs.eqn_size,
                             eqs.var_address,
                             eqs.PARAM,
                             eqs.jac.shape,
                             eqs.nstep,
                             mutable_matrix_blocks=J.get('mutable_matrix_blocks'))
    code_dict["inner_J"] = J['code_inner_J']
    code_dict["sub_inner_J"] = J['code_sub_inner_J']
    if J.get('no_njit_sub_inner_J'):
        code_dict["no_njit_sub_inner_J"] = J['no_njit_sub_inner_J']
    # @njit decision for inner_F / inner_J is based on what actually
    # lands in the function's argument list. Even though the Mat_Mul
    # precompute architecture forces plain sparse ``dim=2`` matrices
    # OUT of ``param_list``, sparse matrices can still enter via the
    # ``TimeSeriesParam`` or ``triggerable=True`` branches of
    # ``print_param``. If any such sparse parameter exists we must NOT
    # wrap the inner function with ``@njit`` because Numba cannot
    # digest scipy.sparse arguments.
    sparse_in_inner_list = _has_sparse_in_param_list(eqs.PARAM)
    code_dict["no_njit_inner_F"] = sparse_in_inner_list
    code_dict["no_njit_inner_J"] = sparse_in_inner_list
    if sparse_in_inner_list:
        # Sub-functions inherit whatever inner_F / inner_J forward to
        # them, so they are also unsafe to @njit in this case.
        no_njit_F_set = code_dict.get("no_njit_sub_inner_F") or set()
        code_dict["no_njit_sub_inner_F"] = no_njit_F_set | {
            i for i in range(len(code_dict["sub_inner_F"]))}
    # Mutable matrix block functions — dedicated @njit scatter-add loops.
    code_dict["mut_mat_block_funcs"] = J.get('mut_mat_block_funcs', [])
    mut_mat_mappings = J.get('mut_mat_mappings', {})
    code_dict["mut_mat_mapping_keys"] = sorted(mut_mat_mappings.keys())

    if make_hvp:
        eqs.hvp = Hvp(eqs.jac)
        code_dict["Hvp"] = print_Hvp(eqs.__class__.__name__,
                                     eqs.eqn_size,
                                     eqs.var_address,
                                     eqs.PARAM,
                                     eqs.nstep)
        inner_Hvp = print_inner_Hvp(eqs.var_address,
                                    eqs.PARAM,
                                    eqs.hvp,
                                    eqs.nstep)
        code_dict["inner_Hvp"] = inner_Hvp['code_inner_Hvp']
        code_dict["sub_inner_Hvp"] = inner_Hvp['code_sub_inner_Hvp']

    def print_trigger_func_code():
        code_tfuc = dict()
        trigger_func = parse_trigger_func(eqs.PARAM)
        # rename trigger_func
        for func_name, func in trigger_func.items():
            name0 = func.__name__
            func_code = inspect.getsource(func).replace(name0, func_name)
            func_code = func_code.replace('np.', '')
            code_tfuc[func_name] = func_code
        return code_tfuc

    code_dict["tfunc_dict"] = print_trigger_func_code()

    print('Complete!')

    # parse equation parameters, which is different the p dict
    eqn_parameter = {}
    if isinstance(eqs, SymAE) and not isinstance(eqs, SymFDAE):
        eqn_type = 'AE'
    elif isinstance(eqs, SymFDAE):
        eqn_type = 'FDAE'
        eqn_parameter.update({'nstep': eqs.nstep})
    elif isinstance(eqs, SymDAE):
        eqn_type = 'DAE'
        eqn_parameter.update({'M': eqs.M})
    else:
        raise ValueError(f'Unknown equation type {type(eqs)}')

    row, col, data = eqs.jac.parse_row_col_data()
    eqn_parameter.update({'row': row, 'col': col, 'data': data})
    # Store mutable matrix block mapping arrays (loaded from setting at
    # runtime, passed to the block @njit functions).
    if mut_mat_mappings:
        eqn_parameter.update(mut_mat_mappings)
    if make_hvp:
        row_hvp, col_hvp, data_hvp = eqs.hvp.jac1.parse_row_col_data()
    else:
        row_hvp = 0
        col_hvp = 0
        data_hvp = 0
    eqn_parameter.update({'row_hvp': row_hvp,
                          'col_hvp': col_hvp,
                          'data_hvp': data_hvp})

    print(f"Rendering python modules!")
    render_as_modules(name,
                      code_dict,
                      eqn_type,
                      p,
                      eqn_parameter,
                      y,
                      modules,
                      numba,
                      directory)
    print('Complete!')


def is_valid_python_module_name(module_name):
    pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    return bool(pattern.match(module_name))


def create_python_module(module_name,
                         initiate_code,
                         module_code,
                         dependency_code,
                         auxiliary,
                         directory=None):
    location = module_name if directory is None else os.path.join(directory, module_name)

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

    save(auxiliary, os.path.join(location, "param_and_setting.pkl"))


try_hvp = """
try:
    from .num_func import Hvp_
    mdl.HVP = Hvp_
    has_hvp = True
except ImportError:
    has_hvp = False
"""

code_compile = """
mdl.F({alpha})
mdl.J({alpha})
if has_hvp:
    from numpy import ones_like
    v = ones_like(y)
    mdl.HVP({beta})
"""


def print_init_code(eqn_type: str, module_name, eqn_param):
    code = '"""\n'
    from ...._version import __version__
    code += f'Python module generated by Solverz {__version__}\n'
    code += '"""\n'
    code += 'from .num_func import F_, J_\n'
    code += 'from .dependency import setting, p__ as p, y__ as y\n'
    code += 'import time\n'
    match eqn_type:
        case 'AE':
            code += 'from Solverz.num_api.num_eqn import nAE\n'
            code += 'mdl = nAE(F_, J_, p)\n'
            code_compile_args_F_J = "y, p"
            code_compile_args_Hvp = "y, p, v"
        case 'FDAE':
            try:
                nstep = eqn_param['nstep']
            except KeyError as e:
                raise ValueError("Cannot parse nstep attribute for FDAE object printing!")
            code += 'from Solverz.num_api.num_eqn import nFDAE\n'
            code += 'mdl = nFDAE(F_, J_, p, setting["nstep"])\n'
            args_str = ', '.join(['y' for i in range(nstep)])
            code_compile_args_F_J = f"0, y, p, {args_str}"
            code_compile_args_Hvp = f"0, y, p, v, {args_str}"
        case 'DAE':
            code += 'from Solverz.num_api.num_eqn import nDAE\n'
            code += 'mdl = nDAE(setting["M"], F_, J_, p)\n'
            code_compile_args_F_J = "0, y, p"
            code_compile_args_Hvp = "0, y, p, v"
        case _:
            raise ValueError(f'Unknown equation type {eqn_type}')
    code += try_hvp
    code += f'print("Compiling model {module_name}...")\n'
    code += f'start = time.perf_counter()\n'
    code += code_compile.format(alpha=code_compile_args_F_J, beta=code_compile_args_Hvp)
    code += f'end = time.perf_counter()\n'
    code += "print(f'Compiling time elapsed: {end - start}s')\n"
    return code


def print_module_code(code_dict: Dict[str, str], numba=False):
    code = 'from .dependency import *\n'
    code += """_data_ = setting["data"]\n"""
    code += """_data_hvp = setting["data_hvp"]\n"""
    code += """_F_ = zeros_like(y__, dtype=float64)\n"""
    # Load mutable matrix block mapping arrays from setting at module-level
    # so the J_ wrapper can reference them directly (and numba sees typed
    # numpy arrays).
    for mapping_key in code_dict.get('mut_mat_mapping_keys', []):
        code += f'{mapping_key} = setting["{mapping_key}"]\n'
    code += '\r\n'
    for tfunc in code_dict['tfunc_dict'].values():
        if numba:
            code += '@njit(cache=True)\n'
        code += tfunc
        code += '\n\r\n'

    # --- Selective @njit logic ---
    # Equations using Mat_Mul receive scipy.sparse matrices that Numba cannot
    # handle.  We track which sub_inner functions need Mat_Mul (no_njit sets)
    # and skip @njit for those.  When ANY sub_inner_F uses Mat_Mul, the
    # dispatcher inner_F also cannot be @njit (it passes the sparse matrix).
    # Non-Mat_Mul sub_inner functions remain @njit for performance.
    no_njit_F = code_dict.get('no_njit_sub_inner_F', set())
    no_njit_J = code_dict.get('no_njit_sub_inner_J', set())
    has_mat_mul_F = len(no_njit_F) > 0
    has_mat_mul_J = len(no_njit_J) > 0
    no_njit_inner_F = code_dict.get('no_njit_inner_F', False)
    no_njit_inner_J = code_dict.get('no_njit_inner_J', False)

    code += code_dict['F']
    code += '\n\r\n'
    if numba and not has_mat_mul_F and not no_njit_inner_F:
        code += '@njit(cache=True)\n'
    code += code_dict['inner_F']
    code += '\n\r\n'
    for i, sub_func in enumerate(code_dict['sub_inner_F']):
        if numba and i not in no_njit_F:
            code += '@njit(cache=True)\n'
        code += sub_func
        code += '\n\r\n'

    # Mutable matrix block @njit functions must be defined BEFORE J_ wrapper
    # (which calls them). They're pure scatter-add loops, always @njit-able.
    for block_func in code_dict.get('mut_mat_block_funcs', []):
        if numba:
            code += '@njit(cache=True)\n'
        code += block_func
        code += '\n\r\n'

    code += code_dict['J']
    code += '\n\r\n'
    if numba and not has_mat_mul_J and not no_njit_inner_J:
        code += '@njit(cache=True)\n'
    code += code_dict['inner_J']
    code += '\n\r\n'
    for i, sub_func in enumerate(code_dict['sub_inner_J']):
        if numba and i not in no_njit_J:
            code += '@njit(cache=True)\n'
        code += sub_func
        code += '\n\r\n'

    try:
        code += code_dict['Hvp']
        code += '\n\r\n'
        if numba:
            code += '@njit(cache=True)\n'
        code += code_dict['inner_Hvp']
        code += '\n\r\n'
        for sub_func in code_dict['sub_inner_Hvp']:
            if numba:
                code += '@njit(cache=True)\n'
            code += sub_func
            code += '\n\r\n'
    except KeyError:
        pass

    return code


def print_dependency_code(modules):
    code = "import os\n"
    code += "current_module_dir = os.path.dirname(os.path.abspath(__file__))\n"
    code += 'from Solverz import load\n'
    code += 'auxiliary = load(os.path.join(current_module_dir, "param_and_setting.pkl"))\n'
    code += 'from numpy import *\n'
    code += 'from Solverz.num_api.module_parser import *\n'
    code += 'setting = auxiliary["eqn_param"]\n'
    code += 'row = setting["row"]\n'
    code += 'col = setting["col"]\n'
    code += 'row_hvp = setting["row_hvp"]\n'
    code += 'col_hvp = setting["col_hvp"]\n'
    code += 'data_hvp = setting["data_hvp"]\n'
    code += 'p__ = auxiliary["p"]\n'
    code += 'y__ = auxiliary["vars"]\n'
    return code


def render_as_modules(name,
                      code_dict: Dict[str, str],
                      eqn_type: str,
                      p: dict,
                      eqn_param: dict,
                      variables: Vars,
                      modules=None,
                      numba=False,
                      directory=None):
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
