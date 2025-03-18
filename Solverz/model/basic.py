from typing import Dict
import warnings

import numpy as np
from Solverz.equation.equations import AE, FDAE, DAE
from Solverz.equation.param import ParamBase
from Solverz.equation.eqn import Eqn, Ode
from Solverz.utilities.address import Address
from Solverz.variable.variables import Vars
from Solverz.variable.ssymbol import Var, AliasVar
from Solverz.num_api.Array import Array


class Model:

    def __init__(self):
        self.eqn_dict = dict()
        self.var_dict: Dict[str, Var] = dict()
        self.param_dict = dict()
        self.alias_dict = dict()

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
        self.var_dict[name].value = Array(init_func.NUM_EQN(*args), dim=1)

    def create_instance(self):
        attr_dict = vars(self)

        eqn_type = 'AE'
        nstep = None
        for key, value in attr_dict.items():
            if isinstance(value, (Ode, Eqn)):
                self.eqn_dict[key] = value
            elif isinstance(value, Var):
                self.var_dict[key] = value
            elif isinstance(value, ParamBase):
                self.param_dict[key] = value
            elif isinstance(value, AliasVar):
                nstep = 1 if nstep is None else nstep
                self.alias_dict[key] = value
                nstep = np.max([nstep, value.step])

        if any([isinstance(arg, Ode) for arg in self.eqn_dict.values()]):
            if nstep is not None:
                raise ValueError("DAE object cannot have iAliasVar!")
            eqn_type = 'DAE'
        elif nstep is not None:
            eqn_type = 'FDAE'

        if eqn_type == 'AE':
            eqs = AE(list(self.eqn_dict.values()))
        elif eqn_type == 'DAE':
            eqs = DAE(list(self.eqn_dict.values()))
        elif eqn_type == 'FDAE':
            eqs = FDAE(list(self.eqn_dict.values()), nstep=nstep)
        else:
            raise TypeError(f"Equation type {eqn_type} not implemented!")

        for name, param in self.param_dict.items():
            if isinstance(param, ParamBase):
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
        if eqn_type == 'FDAE':
            for i in range(nstep):
                eqs.update_param(y0.derive_alias(f'_tag_{i}'))
        # eqs.FormJac(y0)
        eqs.assign_eqn_var_address(y0)

        if eqs.eqn_size != eqs.vsize:
            warnings.warn(f'Equation size {eqs.eqn_size} and variable size {eqs.vsize} not equal!')

        return eqs, y0

    def add(self, m):
        if isinstance(m, Model):
            self.__dict__.update(m.__dict__)
        elif isinstance(m, (Var, ParamBase, Eqn, Ode)):
            self.__dict__[m.name] = m
        elif isinstance(m, dict):
            self.__dict__.update(m)
        else:
            raise ValueError(f"Unknown element type {type(m)}")
