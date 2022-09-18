from __future__ import annotations

from numbers import Number
from typing import Union, Optional, List, Dict, Callable, Set, Tuple
import numpy as np
from var import Var, Vars
from param import Param
from algebra import Sympify_Mapping
from solverz_array import SolverzArray, Lambdify_Mapping
from sympy import sympify, lambdify, symbols, Symbol, Expr
from copy import deepcopy
import warnings


class Eqn:
    """
    The Equation object
    """

    def __init__(self,
                 name: Union[str],
                 e_str: Union[str],
                 commutative: Union[bool]):

        self.name = name
        self.e_str = e_str
        self.commutative = commutative

        self.EQN = sympify(self.e_str, locals=Sympify_Mapping)

        temp_sympify_mapping = deepcopy(Sympify_Mapping)
        for symbol in self.EQN.free_symbols:
            # commutative=False and real=True are inconsistent assumptions
            if self.commutative:
                temp_sympify_mapping[symbol.name] = symbols(symbol.name, commutative=self.commutative, real=True)
            else:
                temp_sympify_mapping[symbol.name] = symbols(symbol.name, commutative=self.commutative)

        self.EQN = sympify(self.e_str, locals=temp_sympify_mapping)

        self.SYMBOLS: List[Symbol] = list(self.EQN.free_symbols)
        self.NUM_EQN: Callable = lambdify(self.SYMBOLS, self.EQN, [Lambdify_Mapping, 'numpy'])

        # self.check_if_var_and_param_defined()

    def check_if_var_and_param_defined(self):
        for symbol in list(self.free_symbols):
            if symbol.name not in self.__y and symbol.name not in self.PARAM:
                warnings.warn(f'{symbol.name} not defined')

    def eval(self, *args: Union[SolverzArray, np.ndarray]) -> np.ndarray:
        return np.asarray(self.NUM_EQN(*args))

    def __repr__(self):
        return f"Equation: {self.name}"


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
        if isinstance(eqn, Eqn):
            self.EQNs[eqn.name] = eqn
        else:
            for eqn_ in eqn:
                self.EQNs[eqn_.name] = eqn_
                for symbol_ in self.EQNs[eqn_.name].SYMBOLS:
                    # generate bare symbols without assumptions
                    self.SYMBOLS[symbol_.name] = symbols(symbol_.name)

        if param:
            if isinstance(param, Param):
                param = [param]

            self.PARAM: Dict[str, Param] = dict()
            for param_ in param:
                if not self.is_param_defined(param_.name):
                    raise ValueError(f'Parameter {param_.name} not defined in equations!')
                else:
                    self.PARAM[param_.name] = param_

            # generate derivatives of EQNs
            for eqn_name, eqn_ in self.EQNs.items():
                self.__eqn_diffs[eqn_name] = dict()
                for symbol_ in eqn_.SYMBOLS:
                    if symbol_.name not in self.PARAM:
                        temp = eqn_.EQN.diff(symbol_)
                        self.__eqn_diffs[eqn_name][symbol_.name] = Eqn(name=f'Diff {eqn_name} w.r.t. {symbol_.name}',
                                                                       e_str=temp.__repr__(),
                                                                       commutative=eqn_.commutative)

    @property
    def eqn_diffs(self):
        return self.__eqn_diffs

    def is_param_defined(self, param: Union[str, Param]) -> bool:
        if isinstance(param, str):
            pname = param
        else:
            pname = param.name
        if pname in self.SYMBOLS:
            return True
        else:
            return False

    def eval(self, eqn_name: str, *args: Union[SolverzArray, np.ndarray]) -> SolverzArray:
        return SolverzArray(self.EQNs[eqn_name].NUM_EQN(*args))

    def eval_diffs(self, eqn_name: str, var_name: str, *args: Union[SolverzArray, np.ndarray]) -> SolverzArray:
        return SolverzArray(self.eqn_diffs[eqn_name][var_name].NUM_EQN(*args))

    def __obtain_eqn_args(self, y: Vars, eqn: Eqn):
        args = []
        for symbol in eqn.SYMBOLS:
            if symbol.name in self.PARAM:
                args = [*args, self.PARAM[symbol.name].v]
            elif symbol.name in y.v:
                args = [*args, y[symbol.name]]
            else:
                raise ValueError(f'Cannot find the values of variable {symbol.name}')
        return args

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

    def g_y(self, y: Vars, eqn: List[str] = None, var: List[Var] = None) -> List[Tuple[str, str, np.ndarray]]:
        """
        generate Jacobian matrices of Eqn object with respect to var object
        :param y:
        :param eqn:
        :param var:
        :return:
        """
        if not eqn:
            eqn = list(self.EQNs.keys())
        if not var:
            var = y.VARS

        gy: List[Tuple[str, str, np.ndarray]] = []
        # size: Dict[str, Dict[str, Number]] = dict()
        for eqn_ in eqn:
            max_row_size = 0
            for var_ in var:
                args = []
                if var_.name in self.eqn_diffs[eqn_]:
                    args = self.__obtain_eqn_args(y, self.eqn_diffs[eqn_][var_.name])
                    temp1 = SolverzArray(self.eval_diffs(eqn_, var_.name, *args))
                    if temp1.column_size > 1 and temp1.row_size > 1:
                        #  matrix
                        gy = [*gy, (eqn_, var_.name, np.asarray(temp1))]
                    elif temp1.column_size == 1 and temp1.row_size > 1:
                        # vector
                        gy = [*gy, (eqn_, var_.name, np.diag(temp1))]
                    else:
                        # [number]
                        gy = [*gy, (eqn_, var_.name, np.asarray(temp1) * np.identity(y.size[var_.name]))]

        return gy

    def add(self,
            eqn: Union[list, str, Eqn] = None,
            variable: Var = None,
            parameter: Param = None):
        # TODO:  check vars/params with the same names
        self.is_param_defined()
        pass

    def __add__(self, other):
        # TODO OR NOT TODO. THIS IS A QUESTION.
        pass

    def __repr__(self):
        return f"{self.name} equations"
