from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Union, List, Dict, Callable, Tuple

import numpy as np
from sympy import sympify, lambdify, symbols, Symbol
from numbers import Number

from .algebra import Sympify_Mapping
from .param import Param
from .solverz_array import SolverzArray, Lambdify_Mapping
from .var import Var
from .variables import Vars
from .eqn import Eqn, Ode


class Equations:

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 param: Union[List[Param], Param] = None
                 ):
        self.name = name

        self.EQNs: Dict[str, Eqn] = dict()
        self.__eqn_diffs: Dict[str, Dict[str, Eqn]] = dict()
        self.SYMBOLS: Dict[str, Symbol] = dict()
        self.__a: Dict[str, List[int]] = dict()  # equation address
        self.__eqn_size: Dict[str, int] = dict()  # equation size
        self.__var_size: int = 0  # potential equation size

        if isinstance(eqn, Eqn):
            eqn = [eqn]

        for eqn_ in eqn:
            self.EQNs[eqn_.name] = eqn_
            for symbol_ in self.EQNs[eqn_.name].SYMBOLS:
                # generate bare symbols without assumptions
                self.SYMBOLS[symbol_.name] = symbols(symbol_.name)
            self.__a[eqn_.name] = []

        # set parameters
        self.PARAM: Dict[str, Param] = dict()
        if param:
            if isinstance(param, Param):
                param = [param]

            for param_ in param:
                if not self.is_param_defined(param_.name):
                    raise ValueError(f'Parameter {param_.name} not defined in equations!')
                else:
                    self.PARAM[param_.name] = param_

    @property
    def eqn_diffs(self):
        return self.__eqn_diffs

    @property
    def a(self):
        return self.__a

    @property
    def size(self):
        return self.__eqn_size

    @property
    def eqn_size(self):
        return np.sum(np.array(list(self.size.values())))

    @property
    def var_size(self):
        return self.__var_size

    @var_size.setter
    def var_size(self, value: int):
        self.__var_size = value

    def is_param_defined(self, param: str) -> bool:

        if param in self.SYMBOLS:
            return True
        else:
            return False


class AE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 param: Union[List[Param], Param] = None):
        super().__init__(eqn, name, param)

        # Check if some equation in self.eqn is Eqn.
        # If not, raise error
        if any([isinstance(eqn_, Ode) for eqn_ in self.EQNs.values()]):
            raise ValueError(f'Ode found. This object should be DAE!')

        # generate derivatives of EQNs
        for eqn_name, eqn_ in self.EQNs.items():
            self.eqn_diffs[eqn_name] = dict()
            for symbol_ in eqn_.SYMBOLS:
                if symbol_.name not in self.PARAM:
                    eqn_diff = eqn_.EQN.diff(symbol_)
                    self.eqn_diffs[eqn_name][symbol_.name] = Eqn(name=f'Diff {eqn_name} w.r.t. {symbol_.name}',
                                                                 e_str=eqn_diff.__repr__(),
                                                                 commutative=eqn_.commutative)

    def eval(self, eqn_name: str, *args: Union[SolverzArray, np.ndarray]) -> SolverzArray:
        """
        Evaluate equations
        :param eqn_name:
        :param args:
        :return:
        """
        return SolverzArray(self.EQNs[eqn_name].NUM_EQN(*args))

    def eval_diffs(self, eqn_name: str, var_name: str, *args: Union[SolverzArray, np.ndarray]) -> SolverzArray:
        """
        Evaluate derivative of equations
        :param eqn_name:
        :param var_name:
        :param args:
        :return:
        """
        return SolverzArray(self.eqn_diffs[eqn_name][var_name].NUM_EQN(*args))

    def __obtain_eqn_args(self, y: Vars, eqn: Eqn) -> List[SolverzArray]:
        args = []
        for symbol in eqn.SYMBOLS:
            if symbol.name in self.PARAM:
                if self.PARAM[symbol.name].triggerable:
                    self.PARAM[symbol.name].v = self.PARAM[symbol.name].trigger_fun(
                        y[self.PARAM[symbol.name].trigger_var])
                args = [*args, self.PARAM[symbol.name].v]
            elif symbol.name in y.v:
                args = [*args, y[symbol.name]]
            else:
                raise ValueError(f'Cannot find the values of variable {symbol.name}')
        return args

    def assign_equation_address(self, y: Vars):
        """
        ASSIGN ADDRESSES TO EQUATIONS
        """
        temp = 0
        for eqn_name in self.EQNs.keys():
            self.a[eqn_name] = [temp, temp + self.g(y, eqn_name).row_size - 1]
            temp = temp + self.g(y, eqn_name).row_size
            self.size[eqn_name] = self.g(y, eqn_name).row_size
        self.var_size = y.total_size

    def g(self, y: Vars, eqn: str = None) -> SolverzArray:
        """

        :param y:
        :param eqn:
        :return:
        """

        if not eqn:
            temp = np.array([])
            for eqn_name, eqn_ in self.EQNs.items():
                args = self.__obtain_eqn_args(y, eqn_)
                temp = np.concatenate([temp, np.asarray(self.eval(eqn_name, *args)).reshape(-1, )])
            return SolverzArray(temp)
        else:
            args = self.__obtain_eqn_args(y, self.EQNs[eqn])
            return self.eval(eqn, *args)

    def g_y(self, y: Vars, eqn: List[str] = None, var: List[str] = None) -> List[Tuple[str, str, np.ndarray]]:
        """
        generate Jacobian matrices of Eqn object with respect to var object
        :param y:
        :param eqn:
        :param var:
        :return: List[Tuple[Equation_name, var_name, np.ndarray]]
        """
        if not eqn:
            eqn = list(self.EQNs.keys())
        if not var:
            var = list(y.v.keys())

        gy: List[Tuple[str, str, np.ndarray]] = []

        for eqn_name in eqn:
            for var_name in var:
                if var_name in self.eqn_diffs[eqn_name]:
                    args = self.__obtain_eqn_args(y, self.eqn_diffs[eqn_name][var_name])
                    temp = SolverzArray(self.eval_diffs(eqn_name, var_name, *args))
                    if temp.column_size > 1:
                        #  matrix or row vector
                        gy = [*gy, (eqn_name, var_name, np.asarray(temp))]
                    elif temp.column_size == 1 and temp.row_size > 1:
                        # column vector
                        gy = [*gy, (eqn_name, var_name, np.diag(temp))]
                    else:
                        # [number]
                        gy = [*gy, (eqn_name, var_name, np.asarray(temp) * np.identity(y.var_size[var_name]))]
        return gy

    def j(self, y: Vars) -> np.ndarray:
        if not self.eqn_size:
            self.assign_equation_address(y)
        gy = self.g_y(y)
        j = np.zeros((self.eqn_size, y.total_size))

        for gy_tuple in gy:
            j[self.a[gy_tuple[0]][0]:(self.a[gy_tuple[0]][-1] + 1), y.a[gy_tuple[1]][0]:(y.a[gy_tuple[1]][-1] + 1)] = \
                gy_tuple[2]

        return j

    def update_param(self, *args):

        if len(args) > 1:
            param: str = args[0]
            value: Union[SolverzArray, np.ndarray, list, Number] = args[1]
            self.PARAM[param].v = value
        elif isinstance(args[0], Vars):
            vars_: Vars = args[0]
            for param_name in self.PARAM.keys():
                if param_name in vars_.v.keys():
                    self.PARAM[param_name].v = vars_.v[param_name]

    def __setitem__(self, key, value):
        """
        hai mei xiang hao ke yi yong lai gan ma
        :param key:
        :param value:
        :return:
        """
        pass

    def __repr__(self):
        if not self.eqn_size:
            return f"Algebraic equation {self.name} with addresses uninitialized"
        else:
            return f"Algebraic equation {self.name} ({self.eqn_size}×{self.var_size})"


class DAE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 param: Union[List[Param], Param] = None
                 ):
        super().__init__(eqn, name, param)

        # Check if some equation in self.eqn is Ode.
        # If not, raise error
        if not any([isinstance(eqn_, Ode) for eqn_ in self.EQNs.values()]):
            raise ValueError(f'No ODE found. Should be AE!')

    def discretize(self, scheme) -> AE:
        eqns = []
        for eqn_ in self.EQNs.values():
            if isinstance(eqn_, Ode):
                self.PARAM, eqn_ = eqn_.discretize(scheme, self.PARAM)
                eqns = eqns + [eqn_]
            else:
                eqns = eqns + [eqn_]

        return AE(eqns, self.name, list(self.PARAM.values()))

    @property
    def is_autonomous(self):

        if 't' in self.SYMBOLS.keys():
            return True
        else:
            return False

    def __repr__(self):
        return f"DAE: {self.name}"
