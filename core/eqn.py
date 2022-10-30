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

    def eval(self, *args: Union[SolverzArray, np.ndarray]) -> np.ndarray:
        return np.asarray(self.NUM_EQN(*args))

    def diff(self, var: str):
        """"""
        pass

    def __repr__(self):
        return f"Equation: {self.name}"


class Ode(Eqn):
    """
    The class of ordinary differential equations
    """

    def __init__(self,
                 name: Union[str],
                 e_str: Union[str],
                 diff_var: str,
                 commutative: Union[bool] = True):
        super().__init__(name, e_str, commutative)
        self.diff_var = diff_var

    def discretize(self,
                   scheme: str,
                   param: Dict[str, Param] = None):
        """

        :param scheme:
        :param param: list of parameters in the Ode
        :return:
        """

        sym_scheme = sympify(scheme)

        sym_diff_var = None
        for symbol in self.SYMBOLS:
            if self.diff_var == symbol.name:
                sym_diff_var = symbol
        if not sym_diff_var:
            sym_diff_var = symbols(self.diff_var, commutative=self.commutative)

        fx0t0 = self.EQN
        for symbol in self.SYMBOLS:
            if param:
                if symbol.name not in param or symbol.name == 't':
                    # symbol.name == 't' in case of non-autonomous systems
                    fx0t0 = fx0t0.subs(symbol, symbols(symbol.name + '0', commutative=symbol.is_commutative))
            else:
                fx0t0 = fx0t0.subs(symbol, symbols(symbol.name + '0', commutative=symbol.is_commutative))

        discretized_ode = sym_scheme.subs([(sympify('f(x,t)'), self.EQN),
                                           (symbols('x'), sym_diff_var),
                                           (sympify('f(x0,t0)'), fx0t0),
                                           (symbols('x0'), symbols(sym_diff_var.name + '0',
                                                                   commutative=sym_diff_var.is_commutative))])

        # Check if the scheme introduces some new parameters like dt, etc.
        for symbol in discretized_ode.free_symbols:
            if param:
                if symbol not in self.EQN.free_symbols and symbol.name not in param and symbol != sym_diff_var:
                    param[symbol.name] = Param(symbol.name)
            else:
                if symbol not in self.EQN.free_symbols and symbol != sym_diff_var:
                    param = dict()
                    param[symbol.name] = Param(symbol.name)

        return param, Eqn('d_' + self.name, e_str=discretized_ode.__str__(), commutative=self.commutative)

    def __repr__(self):
        return f"Ode: {self.name}"


class Pde(Eqn):
    """
    The class of partial differential equations
    """
    pass
