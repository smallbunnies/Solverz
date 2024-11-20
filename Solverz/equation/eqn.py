from __future__ import annotations

from typing import Union, List, Dict, Callable

import numpy as np
from sympy import Symbol, Expr, latex, Derivative, sympify, Function
from sympy import lambdify as splambdify
from sympy.abc import t, x

from Solverz.sym_algebra.symbols import iVar, Para, IdxVar, idx, IdxPara, iAliasVar, IdxAliasVar
from Solverz.variable.ssymbol import Var
from Solverz.sym_algebra.functions import Mat_Mul, Slice
from Solverz.sym_algebra.matrix_calculus import MixedEquationDiff
from Solverz.num_api.module_parser import modules
from Solverz.variable.ssymbol import sSym2Sym
from Solverz.utilities.type_checker import is_zero


class Eqn:
    """
    The Equation object
    """

    def __init__(self,
                 name: str,
                 eqn):
        if not isinstance(name, str):
            raise ValueError("Equation name must be string!")
        self.name: str = name
        self.LHS = 0
        self.RHS = sympify(sSym2Sym(eqn))
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
        for symbol_ in list((self.LHS - self.RHS).free_symbols):
            if isinstance(symbol_, (iVar, Para, idx, iAliasVar)):
                temp_dict[symbol_.name] = symbol_
            elif isinstance(symbol_, (IdxVar, IdxPara, IdxAliasVar)):
                temp_dict[symbol_.name0] = symbol_.symbol0
                temp_dict.update(symbol_.SymInIndex)

        # to sort in lexicographic order
        sorted_dict = {key: temp_dict[key] for key in sorted(temp_dict)}
        return sorted_dict

    def lambdify(self) -> Callable:
        return splambdify(self.SYMBOLS.values(), self.RHS, modules)

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
                if not is_zero(diff):
                    self.derivatives[symbol_.name] = EqnDiff(name=f'Diff {self.name} w.r.t. {symbol_.name}',
                                                             eqn=diff,
                                                             diff_var=symbol_,
                                                             var_idx=idx_.name if isinstance(idx_, idx) else idx_)
            elif isinstance(symbol_, iVar):
                if self.mixed_matrix_vector:
                    diff = MixedEquationDiff(self.RHS, symbol_)
                else:
                    diff = self.RHS.diff(symbol_)
                if not is_zero(diff):
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
        self.diff_var_name = diff_var.name0 if isinstance(diff_var, IdxVar) else diff_var.name
        self.var_idx = var_idx  # df/dPi[i] then var_idx=i
        self.var_idx_func = None
        if self.var_idx is not None:
            if isinstance(self.var_idx, slice):
                temp = []
                if var_idx.start is not None:
                    temp.append(var_idx.start)
                if var_idx.stop is not None:
                    temp.append(var_idx.stop)
                if var_idx.step is not None:
                    temp.append(var_idx.step)
                self.var_idx_func = Eqn('To evaluate var_idx of variable' + self.diff_var.name, Slice(*temp))
            elif isinstance(self.var_idx, Expr):
                self.var_idx_func = Eqn('To evaluate var_idx of variable' + self.diff_var.name, self.var_idx)
        self.LHS = Derivative(Function('F'), diff_var)
        self.dim = -1
        self.v_type = ''


class Ode(Eqn):
    r"""
    The class for ODE reading

    .. math::

         \frac{\mathrm{d}y}{\mathrm{d}t}=f(t,y)

    where $y$ is the state vector.

    """

    def __init__(self, name: str,
                 f,
                 diff_var: Union[iVar, IdxVar, Var]):
        super().__init__(name, f)
        diff_var = sSym2Sym(diff_var)
        self.diff_var = diff_var
        self.LHS = Derivative(diff_var, t)
