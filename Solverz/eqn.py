from __future__ import annotations

from copy import deepcopy
from typing import Union, List, Dict, Callable

import numpy as np
from sympy import sympify, lambdify, Symbol, preorder_traversal, Basic

from Solverz.algebra.num_alg import Sympify_Mapping, F, X, StateVar, AliasVar, AlgebraVar, ComputeParam, new_symbols, traverse_for_mul
from .param import Param
from .solverz_array import Lambdify_Mapping


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
                temp_sympify_mapping[symbol.name] = new_symbols(symbol.name, commutative=self.commutative)
                self.EQN = sympify(self.e_str, temp_sympify_mapping)
        else:
            temp_sympify_mapping = deepcopy(Sympify_Mapping)
            for symbol in self.EQN.free_symbols:
                temp_sympify_mapping[symbol.name] = new_symbols(symbol.name, commutative=self.commutative)
            # traverse the Expr tree and replace '*' by Mat_Mul
            self.EQN = traverse_for_mul(sympify(self.e_str, temp_sympify_mapping))

        self.SYMBOLS: List[Symbol] = list(self.EQN.free_symbols)
        self.NUM_EQN: Callable = lambdify(self.SYMBOLS, self.EQN, [Lambdify_Mapping, 'numpy'])

    def eval(self, *args: Union[np.ndarray]) -> np.ndarray:
        return self.NUM_EQN(*args)

    def diff(self, var: str):
        """"""
        pass

    def subs(self, *args, **kwargs):
        return self.EQN.subs(*args, **kwargs)

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
                   scheme: Basic,
                   param: Dict[str, Param] = None,
                   extra_diff_var: List[str] = None):
        """

        :param extra_diff_var: diff_var from other Eqn
        :param scheme:
        :param param: list of parameters in the Ode
        :return:
        """

        if not extra_diff_var:
            extra_diff_var = []

        if not param:
            param = dict()

        funcs: Dict[F, Basic] = dict()  # function set
        alias: Dict[AliasVar, Symbol] = dict()
        scheme_elements = preorder_traversal(scheme)
        for arg in scheme_elements:
            if isinstance(arg, F):
                # generate subs dict of FunctionClass F
                # arg is a Function
                f_args = arg.args
                # args of Functions
                symbol_dict: Dict[Symbol, Basic] = dict()
                for symbol in self.SYMBOLS:
                    if symbol.name in [self.diff_var] + extra_diff_var:
                        # State Variable
                        symbol_dict[symbol] = self._subs_state_var_in_func_args(f_args[0], symbol)
                    elif symbol.name == 't':
                        # Time variable of non-autonomous equations
                        if 't' not in param.keys():
                            param['t'] = Param('t')
                        symbol_dict[symbol] = self._subs_t_in_func_args(f_args[2])
                    elif symbol.name not in param.keys():
                        # Algebra Variable
                        symbol_dict[symbol] = self._subs_algebra_var_in_func_args(f_args[1], symbol)
                funcs[arg] = self.subs(symbol_dict)
                scheme_elements.skip()
                # skip the args of function class
            elif isinstance(arg, AliasVar):
                if arg.alias_of == 'X':
                    alias[arg] = new_symbols(self.diff_var + arg.suffix, commutative=self.commutative)
                elif arg.alias_of == 'Y':
                    raise ValueError('Really? Schemes may be wrong.')

        scheme = scheme.subs([(key, value) for key, value in funcs.items()])
        scheme = scheme.subs([(key, value) for key, value in alias.items()])
        scheme = scheme.subs([(X, new_symbols(self.diff_var, commutative=self.commutative))])

        # Add new Param
        for symbol in list(scheme.free_symbols):
            if symbol not in self.SYMBOLS and symbol.name not in param and symbol.name != self.diff_var:
                param[symbol.name] = Param(symbol.name)

        return param, Eqn('d_' + self.name, e_str=scheme.__str__(), commutative=self.commutative)

    def _subs_state_var_in_func_args(self, expr: Basic, symbol: Symbol):
        subs_dict: Dict[Union[StateVar, AliasVar, ComputeParam], Symbol] = dict()
        for symbol_ in list(expr.free_symbols):
            if isinstance(symbol_, StateVar):
                subs_dict[symbol_] = symbol
            elif isinstance(symbol_, AliasVar):
                subs_dict[symbol_] = new_symbols(symbol.name + symbol_.suffix, commutative=self.commutative)
            elif isinstance(symbol_, ComputeParam):
                subs_dict[symbol_] = new_symbols(symbol_.name, commutative=self.commutative)
        return expr.subs(subs_dict)

    def _subs_t_in_func_args(self, expr: Basic):
        # symbol=t
        subs_dict: Dict[Union[ComputeParam], Symbol] = dict()
        for symbol_ in list(expr.free_symbols):
            if isinstance(symbol_, ComputeParam):
                subs_dict[symbol_] = new_symbols(symbol_.name, commutative=self.commutative)
        return expr.subs(subs_dict)

    def _subs_algebra_var_in_func_args(self, expr: Basic, symbol: Symbol):
        subs_dict: Dict[Union[AlgebraVar, AliasVar, ComputeParam], Symbol] = dict()
        for symbol_ in list(expr.free_symbols):
            if isinstance(symbol_, AlgebraVar):
                subs_dict[symbol_] = symbol
            elif isinstance(symbol_, AliasVar):
                subs_dict[symbol_] = new_symbols(symbol.name + symbol_.suffix, commutative=self.commutative)
            elif isinstance(symbol_, ComputeParam):
                subs_dict[symbol_] = new_symbols(symbol_.name, commutative=self.commutative)
        return expr.subs(subs_dict)

    def __repr__(self):
        return f"Ode: {self.name}"


class Pde(Eqn):
    """
    The class of partial differential equations
    """
    pass
