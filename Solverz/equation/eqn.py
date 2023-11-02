from __future__ import annotations

from typing import Union, List, Dict, Callable, Tuple

import numpy as np
import sympy
from sympy import Symbol, preorder_traversal, Basic, Expr, latex, Derivative
from sympy import lambdify as splambdify
from sympy.abc import t

from Solverz.num.num_alg import F, X, StateVar, AliasVar, AlgebraVar, ComputeParam, new_symbols, \
    pre_lambdify, Mat_Mul
from Solverz.num.num_alg import Param_, Var_, IdxVar, idx, IdxParam, Const_, IdxConst
from Solverz.num.num_interface import numerical_interface
from Solverz.num.matrix_calculus import MixedEquationDiff
from Solverz.param import Param


class Eqn:
    """
    The Equation object
    """

    def __init__(self,
                 name: str,
                 eqn: Expr):

        self.name = name
        self.LHS = 0
        self.RHS = eqn
        self.SYMBOLS: Dict[str, Symbol] = self.obtain_symbols()

        # if the eqn has Mat_Mul, then label it as mixed-matrix-vector equation
        if self.expr.has(Mat_Mul):
            self.mixed_matrix_vector = True
        else:
            self.mixed_matrix_vector = False

        self.NUM_EQN: Callable = self.lambdify()
        self.derivatives: Dict[str, EqnDiff] = dict()

    def obtain_symbols(self) -> Dict[str, Symbol]:
        temp_dict = dict()
        for symbol_ in list(self.RHS.free_symbols):
            if isinstance(symbol_, (Var_, Param_, Const_)):
                temp_dict[symbol_.name] = symbol_
            elif isinstance(symbol_, (IdxVar, IdxParam, IdxConst)):
                temp_dict[symbol_.symbol.name] = symbol_.symbol
                if isinstance(symbol_.index, idx):
                    temp_dict[symbol_.index.name] = symbol_.index
                elif isinstance(symbol_.index, tuple):
                    for idx_ in symbol_.index:
                        if isinstance(idx_, idx):
                            temp_dict[idx_.name] = idx_

        return temp_dict

    def lambdify(self) -> Callable:
        return splambdify(self.SYMBOLS.values(), pre_lambdify(self.RHS), [numerical_interface, 'numpy'])

    def eval(self, *args: Union[np.ndarray]) -> np.ndarray:
        return self.NUM_EQN(*args)

    def derive_derivative(self):
        """"""
        for symbol_ in list(self.RHS.free_symbols):
            # differentiate only to variables
            if isinstance(symbol_, IdxVar):  # if the equation contains Indexed variables
                idx_ = symbol_.index
                if self.mixed_matrix_vector:
                    diff = MixedEquationDiff(self.RHS, symbol_)
                else:
                    diff = self.RHS.diff(symbol_)
                self.derivatives[symbol_.name] = EqnDiff(name=f'Diff {self.name} w.r.t. {symbol_.name}',
                                                         eqn=diff,
                                                         diff_var=symbol_,
                                                         var_idx=idx_.name if isinstance(idx_, idx) else idx_)
            elif isinstance(symbol_, Var_):
                if self.mixed_matrix_vector:
                    diff = MixedEquationDiff(self.RHS, symbol_)
                else:
                    diff = self.RHS.diff(symbol_)
                self.derivatives[symbol_.name] = EqnDiff(name=f'Diff {self.name} w.r.t. {symbol_.name}',
                                                         eqn=diff,
                                                         diff_var=symbol_)

    @property
    def expr(self):
        return self.LHS - self.RHS

    def subs(self, *args, **kwargs):
        return self.RHS.subs(*args, **kwargs)

    def __repr__(self):
        # sympy objects' printing prefers __str__() to __repr__()
        return self.LHS.__str__() + r"=" + self.RHS.__str__()

    def _repr_latex_(self):
        """
        So that jupyter notebook can display latex equation of Eqn object.
        :return:
        """
        return r"$\displaystyle %s$" % (latex(self.LHS) + r"=" + latex(self.RHS))


class EqnDiff(Eqn):
    """
    To store the derivatives of equations W.R.T. variables
    """

    def __init__(self, name: str, eqn: Expr, diff_var: Symbol, var_idx=None):
        super().__init__(name, eqn)
        self.diff_var = diff_var
        self.diff_var_name = diff_var.symbol.name if isinstance(diff_var, IdxVar) else diff_var.name
        self.var_idx: str = var_idx  # df/dPi[i] then var_idx=i
        self.LHS = Derivative(sympy.Function('g'), diff_var)


class Ode(Eqn):
    """
    The class of ordinary differential equations
    """

    def __init__(self, name: str, eqn: Expr, diff_var: Var_):
        super().__init__(name, eqn)
        self.diff_var = diff_var
        self.LHS = Derivative(diff_var, t)

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

        return param, Eqn('d_' + self.name, eqn=scheme.__str__(), commutative=self.commutative)

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


class Pde(Eqn):
    """
    The class of partial differential equations
    """
    pass


class HyperbolicPde(Pde):
    pass
