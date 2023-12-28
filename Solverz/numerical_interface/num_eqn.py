from typing import Callable, Dict

import numpy as np
from Solverz.auxiliary_service.address import Address
from Solverz.variable.variables import Vars, TimeVars
from Solverz.equation.equations import Equations as SymEquations, AE as SymAE, tAE as SymtAE, DAE as SymDAE, \
    FDAE as SymFDAE
from Solverz.numerical_interface.code_printer import print_J, print_F, Solverzlambdify
from Solverz.numerical_interface.custom_function import numerical_interface
from Solverz.equation.param import TimeSeriesParam


def parse_ae_v(y: np.ndarray, var_address: Address):
    return Vars(var_address, y)


def parse_dae_v(y: np.ndarray, var_address: Address):
    temp = Vars(var_address, y[0, :])
    temp = TimeVars(temp, y.shape[0])
    temp.array[:, :] = y
    return temp


class nAE:

    def __init__(self,
                 g: Callable,
                 J: Callable,
                 p: Dict):

        self.g = g
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
    for triggered_var, trigger_var in ae.triggerable_quantity.items():
        func.update({triggered_var + '_trigger_func': ae.PARAM[triggered_var].trigger_fun})

    return func


def made_numerical(eqn: SymEquations, *xys, output_code=False):
    """
    factory method of numerical equations
    """
    eqn.assign_eqn_var_address(*xys)
    code_F = print_F(eqn)
    code_J = print_J(eqn)
    custom_func = dict()
    custom_func.update(numerical_interface)
    custom_func.update(parse_trigger_fun(eqn))
    F = Solverzlambdify(code_F, 'F_', modules=[custom_func, 'numpy'])
    J = Solverzlambdify(code_J, 'J_', modules=[custom_func, 'numpy'])
    p = parse_p(eqn)
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
