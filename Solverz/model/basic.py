from typing import Dict

import numpy as np
from Solverz.equation.equations import AE, FDAE, DAE
from Solverz.equation.param import Param
from Solverz.equation.eqn import Eqn, Ode
from Solverz.utilities.address import Address
from Solverz.variable.variables import Vars
from Solverz.variable.ssymbol import sVar, sAliasVar


class Model:

    def __init__(self):
        self.eqn_dict = dict()
        self.var_dict: Dict[str, sVar] = dict()
        self.param_dict = dict()

    def init_var(self, name):
        init_func = Eqn(f'init_func_{name}', self.var_dict[name].init)
        args = []
        for arg in init_func.SYMBOLS.keys():
            if arg in self.param_dict:
                args += [self.param_dict[arg].value]
            elif arg in self.var_dict:
                if self.var_dict[arg].value is not None:
                    args += [self.var_dict[arg].value]
                else:
                    args += [self.init_var(arg)]
        self.var_dict[name].value = init_func.NUM_EQN(*args)

    def create_instance(self):
        attr_dict = vars(self)

        eqn_type = 'AE'
        for key, value in attr_dict.items():
            if isinstance(value, (Ode, Eqn)):
                self.eqn_dict[key] = value
            elif isinstance(value, sVar):
                self.var_dict[key] = value
            elif isinstance(value, Param):
                self.param_dict[key] = value

        if any([isinstance(arg, Ode) for arg in self.eqn_dict.values()]):
            eqn_type = 'DAE'
        elif any([isinstance(arg, sAliasVar) for arg in self.var_dict.values()]):
            eqn_type = 'FDAE'

        if eqn_type == 'AE':
            eqs = AE(list(self.eqn_dict.values()))
        elif eqn_type == 'DAE':
            eqs = DAE(list(self.eqn_dict.values()))
        elif eqn_type == 'FDAE':
            raise NotImplementedError('Type FDAE not implemented here. Please use FDAE() to create instance!')
        else:
            raise TypeError(f"Equation type {eqn_type} not implemented!")

        for name, param in self.param_dict.items():
            if isinstance(param, Param):
                eqs.param_initializer(name, param)

        a = Address()
        # initialize variables
        for name, var in self.var_dict.items():
            if var.value is None and var.init is None:
                raise ValueError(f'Variable {name} not initialized and init func not provided!')
            elif var.value is None and var.init is not None:
                self.init_var(name)
            a.add(name, self.var_dict[name].value.shape[0])

        array = np.zeros((a.total_size,))
        for var in a.object_list:
            array[a[var]] = self.var_dict[var].value
        y0 = Vars(a, array)
        eqs.assign_eqn_var_address(y0)
        return eqs, y0
