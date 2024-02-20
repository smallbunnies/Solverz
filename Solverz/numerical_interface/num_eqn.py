from typing import Callable, Dict, Literal

import numpy as np

from Solverz.equation.equations import Equations as SymEquations, AE as SymAE, DAE as SymDAE, \
    FDAE as SymFDAE
from Solverz.equation.param import TimeSeriesParam
from Solverz.numerical_interface.code_printer import print_J, print_F, Solverzlambdify, print_F_numba, print_J_numba, \
    print_inner_F, print_inner_J, render_as_modules, print_sub_inner_F
from Solverz.numerical_interface.custom_function import numerical_interface
from Solverz.utilities.address import Address
from Solverz.variable.variables import Vars, TimeVars, combine_Vars


def parse_ae_v(y: np.ndarray, var_address: Address):
    return Vars(var_address, y)


def parse_dae_v(y: np.ndarray, var_address: Address):
    temp = Vars(var_address, y[0, :])
    temp = TimeVars(temp, y.shape[0])
    temp.array[:, :] = y
    return temp


class nAE:

    def __init__(self,
                 F: Callable,
                 J: Callable,
                 p: Dict):
        self.F = F
        self.J = J
        self.p = p


class nFDAE:

    def __init__(self,
                 F: callable,
                 J: callable,
                 p: dict,
                 nstep: int = 0):
        self.F = F
        self.J = J
        self.p = p
        self.nstep = nstep


class nDAE:

    def __init__(self,
                 M,
                 F: Callable,
                 J: Callable,
                 p: Dict):
        self.M = M
        self.F = F
        self.J = J
        self.p = p


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
