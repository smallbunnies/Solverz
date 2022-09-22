from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Union, List, Dict, Callable, Tuple

import numpy as np
from sympy import sympify, lambdify, symbols, Symbol

from .algebra import Sympify_Mapping
from .param import Param
from .solverz_array import SolverzArray, Lambdify_Mapping
from .var import Var
from .variables import Vars


class Eqn:
    """
    The Equation object
    """

    def __init__(self,
                 name: Union[str],
                 e_str: Union[str],
                 commutative: Union[bool] = True):

        self.name = name
        self.e_str = e_str
        self.commutative = commutative

        self.EQN = sympify(self.e_str, locals=Sympify_Mapping)

        # commutative=False and real=True are inconsistent assumptions
        if self.commutative:
            temp_sympify_mapping = dict()
            for symbol in self.EQN.free_symbols:
                temp_sympify_mapping[symbol.name] = symbols(symbol.name, real=True)
        else:
            temp_sympify_mapping = deepcopy(Sympify_Mapping)
            for symbol in self.EQN.free_symbols:
                temp_sympify_mapping[symbol.name] = symbols(symbol.name, commutative=self.commutative)

        self.EQN = sympify(self.e_str, temp_sympify_mapping)
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
