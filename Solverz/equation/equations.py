from __future__ import annotations

import warnings
from numbers import Number
from typing import Union, List, Dict, Tuple
from copy import deepcopy

import numpy as np
from Solverz.equation.hvp import Hvp
from sympy import Symbol, Integer, Expr, Number as SymNumber
from scipy.sparse import csc_array, coo_array
# from cvxopt import spmatrix, matrix

from Solverz.equation.eqn import Eqn, Ode, EqnDiff
from Solverz.equation.param import ParamBase, Param, IdxParam
from Solverz.sym_algebra.symbols import iVar, idx, IdxVar, Para, iAliasVar
from Solverz.sym_algebra.functions import Slice
from Solverz.variable.variables import Vars
from Solverz.utilities.address import Address, combine_Address
from Solverz.utilities.type_checker import is_integer
from Solverz.num_api.Array import Array
from Solverz.equation.jac import Jac, JacBlock


class Equations:

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 matrix_container='scipy'):
        self.name = name

        self.EQNs: Dict[str, Eqn] = dict()
        self.SYMBOLS: Dict[str, Symbol] = dict()
        self.a = Address()  # equation address
        self.var_address = Address()  # variable address
        self.esize: Dict[str, int] = dict()  # size of each equation
        self.vsize: int = 0  # size of variables
        self.f_list = []
        self.g_list = []
        self.matrix_container = matrix_container
        self.PARAM: Dict[str, ParamBase] = dict()
        self.triggerable_quantity: Dict[str, str] = dict()
        self.jac_element_address = Address()
        self.jac: Jac = Jac()
        self.nstep = 0

        if isinstance(eqn, Eqn):
            eqn = [eqn]

        for eqn_ in eqn:
            self.add_eqn(eqn_)

    def add_eqn(self, eqn: Eqn):
        if eqn.name in self.EQNs.keys():
            raise ValueError(f"Equation {eqn.name} already defined!")
        self.EQNs.update({eqn.name: eqn})
        self.SYMBOLS.update(eqn.SYMBOLS)
        self.a.add(eqn.name)
        if isinstance(eqn, Eqn) and not isinstance(eqn, Ode):
            self.g_list = self.g_list + [eqn.name]
        elif isinstance(eqn, Ode):
            self.f_list = self.f_list + [eqn.name]

        for symbol_ in eqn.SYMBOLS.values():
            if isinstance(symbol_, Para):
                # this is not fully initialize of Parameters, please use param_initializer
                self.PARAM[symbol_.name] = Param(symbol_.name,
                                                 value=symbol_.value,
                                                 dim=symbol_.dim)
            elif isinstance(symbol_, iAliasVar):
                self.PARAM[symbol_.name] = Param(symbol_.name,
                                                 value=symbol_.value,
                                                 dim=symbol_.dim,
                                                 is_alias=True)
            elif isinstance(symbol_, idx):
                self.PARAM[symbol_.name] = IdxParam(symbol_.name, value=symbol_.value)

        self.EQNs[eqn.name].derive_derivative()

    def assign_eqn_var_address(self, *args):
        pass

    def Fy(self, y) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        pass

    def FormJac(self, y):
        self.assign_eqn_var_address(y)

        Fy_list = self.Fy(y)
        for fy in Fy_list:
            EqnName = fy[0]
            EqnAddr = self.a[EqnName]
            VarName = fy[1]
            VarAddr = self.var_address[VarName]
            DiffVar = fy[2].diff_var
            DeriExpr = fy[2].RHS

            DiffVarEqn = Eqn('DiffVarEqn' + DiffVar.name, DiffVar)
            args = self.obtain_eqn_args(DiffVarEqn, y, 0)
            DiffVarValue = Array(DiffVarEqn.NUM_EQN(*args), dim=1)

            # The value of deri can be either matrix, vector, or scalar(number). We cannot reshape it.
            Value0 = np.array(fy[3])

            jb = JacBlock(EqnName,
                          EqnAddr,
                          DiffVar,
                          DiffVarValue,
                          VarAddr,
                          DeriExpr,
                          Value0)
            self.jac.add_block(EqnName, DiffVar, jb)

    @property
    def eqn_size(self):
        # total size of all the equations
        return np.sum(np.array(list(self.esize.values())))

    def is_param_defined(self, param: str) -> bool:

        if param in self.SYMBOLS:
            return True
        else:
            return False

    with warnings.catch_warnings():
        warnings.simplefilter("once")

    def param_initializer(self, name, param: ParamBase):
        if not self.is_param_defined(name):
            warnings.warn(f'Parameter {name} not defined in equations!')
        if isinstance(param, ParamBase):
            self.PARAM[name] = param
            if param.triggerable:
                self.triggerable_quantity[param.name] = param.trigger_var
        else:
            raise TypeError(f"Unsupported parameter type {type(param)}")

    def update_param(self, *args):

        if isinstance(args[0], str):
            # Update specified params
            param: str = args[0]
            value: Union[np.ndarray, list, Number] = args[1]
            try:
                if param in self.PARAM:
                    self.PARAM[param].v = value
            except KeyError:
                warnings.warn(f'Equations have no parameter: {param}')
        elif isinstance(args[0], Vars):
            # Update params with Vars. For example, to update x0 in trapezoid rules.
            vars_: Vars = args[0]
            for param_name in self.PARAM.keys():
                if param_name in vars_.var_list:
                    self.PARAM[param_name].v = vars_[param_name]

    def eval(self, eqn_name: str, *args: Union[np.ndarray]) -> np.ndarray:
        """
        Evaluate equations
        :param eqn_name:
        :param args:
        :return:
        """
        return Array(self.EQNs[eqn_name].NUM_EQN(*args), dim=1)

    def trigger_param_updater(self, y):
        # update/initialize triggerable params
        for para_name, trigger_var in self.triggerable_quantity.items():
            trigger_func = self.PARAM[para_name].trigger_fun
            args = []
            for var in trigger_var:
                var_value = None
                if var in self.PARAM:
                    var_value = self.PARAM[var].v
                else:
                    if var in y.var_list:
                        var_value = y[var]
                if var_value is None:
                    raise ValueError(f'Para/iVar {var} not defined')
                else:
                    args.append(var_value)
            temp = trigger_func(*args)
            if self.PARAM[para_name].v is None:
                self.PARAM[para_name].v = temp
            else:
                if type(temp) is not type(self.PARAM[para_name].v):
                    raise TypeError(
                        f"The return types of trigger func for param {para_name} must be {type(self.PARAM[para_name].v)}")

    def obtain_eqn_args(self, eqn: Eqn, y: Vars, t=0) -> List[np.ndarray]:
        """
        Obtain the args of equations
        """

        self.trigger_param_updater(y)

        args = []
        for symbol in eqn.SYMBOLS.values():
            value_obtained = False
            if symbol.name in self.PARAM:
                temp = self.PARAM[symbol.name].get_v_t(t)
                if temp is None:
                    raise TypeError(f'Parameter {symbol.name} uninitialized')
                args.append(temp)
                value_obtained = True
            else:
                if symbol.name in y.var_list:
                    args.append(y[symbol.name])
                    value_obtained = True
            if not value_obtained:
                raise ValueError(f'Cannot find the values of variable {symbol.name}')
        return args

    def eval_diffs(self, eqn_name: str, var_name: str, *args: np.ndarray) -> np.ndarray:
        """
        Evaluate derivative of equations
        :param eqn_name:
        :param var_name:
        :param args:
        :return:
        """
        return self.EQNs[eqn_name].derivatives[var_name].NUM_EQN(*args)

    def evalf(self, *args) -> np.ndarray:
        pass


class AE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 matrix_container='scipy'):
        super().__init__(eqn, name, matrix_container)

        # Check if some equation in self.eqn is Eqn.
        # If not, raise error
        if len(self.f_list) > 0:
            raise ValueError(f'Ode found. This object should be DAE!')

    def assign_eqn_var_address(self, y: Vars):
        """
        ASSIGN ADDRESSES TO EQUATIONS
        """

        temp = 0
        for eqn_name in self.EQNs.keys():
            geval = self.g(y, eqn_name)
            if isinstance(geval, Number):
                eqn_size = 1
            else:
                eqn_size = geval.shape[0]
            self.a.update(eqn_name, eqn_size)
            temp = temp + eqn_size
            self.esize[eqn_name] = eqn_size
        self.var_address = y.a
        self.vsize = y.total_size

    def g(self, y: Vars, eqn: str = None) -> np.ndarray:
        """

        :param y:
        :param eqn:
        :return:
        """
        temp = []
        if not eqn:
            for eqn_name, eqn_ in self.EQNs.items():
                args = self.obtain_eqn_args(eqn_, y)
                g_eqny = self.eval(eqn_name, *args)
                g_eqny = g_eqny.toarray() if isinstance(g_eqny, csc_array) else g_eqny
                temp.append(g_eqny.reshape(-1, ))
            return np.hstack(temp)
        else:
            args = self.obtain_eqn_args(self.EQNs[eqn], y)
            return self.eval(eqn, *args)

    def gy(self, y: Vars, eqn: List[str] = None, var: List[str] = None) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
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
            var = list(y.var_list)

        gy: List[Tuple[str, str, EqnDiff, np.ndarray]] = []

        for eqn_name in eqn:
            eqn_diffs: Dict[str, EqnDiff] = self.EQNs[eqn_name].derivatives
            for var_name in var:
                for key, value in eqn_diffs.items():
                    if var_name == value.diff_var_name:  # f is viewed as f[k]
                        args = self.obtain_eqn_args(eqn_diffs[key], y)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        gy = [*gy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return gy

    def Fy(self, y):
        return self.gy(y)

    def evalf(self, expr: Expr, y: Vars) -> np.ndarray:
        eqn = Eqn('Solverz evalf temporal equation', expr)
        args = self.obtain_eqn_args(eqn, y)
        return eqn.NUM_EQN(*args)

    def __repr__(self):
        if not self.eqn_size:
            return f"Algebraic equation {self.name} with addresses uninitialized"
        else:
            return f"Algebraic equation {self.name} ({self.eqn_size}×{self.vsize})"


class FDAE(AE):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 nstep: int,
                 name: str = None,
                 matrix_container='scipy'):
        super().__init__(eqn, name, matrix_container)

        self.nstep = nstep

    def __repr__(self):
        if not self.eqn_size:
            return f"FDAE {self.name} with addresses uninitialized"
        else:
            return f"FDAE {self.name} ({self.eqn_size}×{self.vsize})"


class DAE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 matrix_container='scipy'
                 ):

        super().__init__(eqn, name, matrix_container)

        self.state_num: int = 0  # number of state variables
        self.algebra_num: int = 0  # number of algebraic variables

        # Check if some equation in self.eqn is Ode.
        # If not, raise error
        if len(self.f_list) == 0:
            raise ValueError(f'No ODE found. You should initialise AE instead!')

    def evalf(self, expr: Expr, t, y: Vars) -> np.ndarray:
        eqn = Eqn('Solverz evalf temporary equation', expr)
        args = self.obtain_eqn_args(eqn, y, t)
        return eqn.NUM_EQN(*args)

    def assign_eqn_var_address(self, y: Vars):
        """
        ASSIGN ADDRESSES TO EQUATIONS f and g
        """

        temp = 0
        for eqn_name in self.f_list:
            feval = self.f(None, y, eqn=eqn_name)
            lhs_eval = self.eval_lhs(None, y, eqn=eqn_name)
            if isinstance(feval, Number):
                rhs_size = 1
            else:
                rhs_size = feval.shape[0]
            if isinstance(lhs_eval, Number):
                lhs_size = 1
            else:
                lhs_size = lhs_eval.shape[0]
            eqn_size = np.max([rhs_size, lhs_size])
            self.a.update(eqn_name, eqn_size)
            temp = temp + eqn_size
            self.esize[eqn_name] = eqn_size

        self.state_num = temp

        for eqn_name in self.g_list:
            geval = self.g(None, y, eqn=eqn_name)
            if np.max(np.abs(geval)) > 1e-5:
                warnings.warn(
                    f'Inconsistent initial values for algebraic equation: {eqn_name}, with deviation {np.max(np.abs(geval))}!')
            if isinstance(geval, Number):
                eqn_size = 1
            else:
                eqn_size = geval.shape[0]
            self.a.update(eqn_name, eqn_size)
            temp = temp + eqn_size
            self.esize[eqn_name] = eqn_size

        self.algebra_num = temp - self.state_num

        self.var_address = y.a
        self.vsize = self.var_address.total_size

    def F(self, t, y) -> np.ndarray:
        """
        Return [f(t,x,y), g(t,x,y)]
        :param t: time
        :param y: Vars
        :return:
        """
        if len(self.g_list) > 0:
            return np.concatenate([self.f(t, y), self.g(t, y)])
        else:
            return self.f(t, y)

    def f(self, t, y, eqn=None) -> np.ndarray:

        temp = []
        if eqn:
            if eqn in self.f_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], y, t)
                temp.append(self.eval(eqn, *args).reshape(-1, ))
        else:
            for eqn in self.f_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], y, t)
                temp.append(self.eval(eqn, *args).reshape(-1, ))

        return np.hstack(temp)

    def eval_lhs(self, t, y, eqn=None) -> np.ndarray:

        temp = []
        if eqn:
            if eqn in self.f_list:
                ode = self.EQNs[eqn]
                if isinstance(ode, Ode):
                    lhs_eqn = Eqn('lhs_' + eqn, ode.diff_var)
                    args = self.obtain_eqn_args(lhs_eqn, y, t)
                    temp.append(Array(lhs_eqn.NUM_EQN(*args), dim=1))
                else:
                    raise TypeError(f"Equation {ode.name} in f_list is not Ode!")
        else:
            for eqn in self.f_list:
                ode = self.EQNs[eqn]
                if isinstance(ode, Ode):
                    lhs_eqn = Eqn('lhs_' + eqn, ode.diff_var)
                    args = self.obtain_eqn_args(lhs_eqn, y, t)
                    temp.append(Array(lhs_eqn.NUM_EQN(*args), dim=1))
                else:
                    raise TypeError(f"Equation {ode.name} in f_list is not Ode!")

        return np.hstack(temp)

    def g(self, t, y, eqn=None) -> np.ndarray:
        """

        `xys` is either:
          - two arguments, e.g. state vars x, and numerical equation y
          - one argument, e.g. state vars y.

        """

        if len(self.g_list) == 0:
            raise ValueError(f'No AE found in {self.name}!')

        temp = []
        if eqn:
            if eqn in self.g_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], y, t)
                temp.append(self.eval(eqn, *args).reshape(-1, ))
        else:
            for eqn in self.g_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], y, t)
                temp.append(self.eval(eqn, *args).reshape(-1, ))

        return np.hstack(temp)

    def fy(self, t, y: Vars) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        """
        generate partial derivatives of f w.r.t. y
        """

        fy: List[Tuple[str, str, EqnDiff, np.ndarray]] = []

        var: List[str] = y.var_list

        for eqn_name in self.f_list:
            eqn_diffs: Dict[str, EqnDiff] = self.EQNs[eqn_name].derivatives
            for var_name in var:
                for key, value in eqn_diffs.items():
                    if var_name == value.diff_var_name:
                        args = self.obtain_eqn_args(eqn_diffs[key], y, t)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        fy = [*fy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return fy

    def gy(self, t, y: Vars) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        """
        generate partial derivatives of g w.r.t. y
        """

        if len(self.g_list) == 0:
            raise ValueError(f'No AE found in {self.name}!')

        gy: List[Tuple[str, str, EqnDiff, np.ndarray]] = []

        var: List[str] = y.var_list

        for eqn_name in self.g_list:
            eqn_diffs: Dict[str, EqnDiff] = self.EQNs[eqn_name].derivatives
            for var_name in var:
                for key, value in eqn_diffs.items():
                    if var_name == value.diff_var_name:
                        args = self.obtain_eqn_args(eqn_diffs[key], y, t)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        gy = [*gy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return gy

    def Fy(self, y):
        fg_xy = self.fy(0, y)
        if len(self.g_list) > 0:
            fg_xy.extend(self.gy(0, y))
        return fg_xy

    @property
    def M(self):
        """
        return the singular mass matrix, M, of dae
        Row of 1 in M corresponds to the differential equation
        Col of 1 in M corresponds to the state variable
        """
        if self.state_num == 0:
            raise ValueError("DAE address uninitialized!")

        row = []
        col = []
        for eqn_name in self.f_list:
            eqn = self.EQNs[eqn_name]
            equation_address = self.a.v[eqn_name]
            if isinstance(eqn, Ode):
                diff_var = eqn.diff_var
                if isinstance(diff_var, iVar):
                    variable_address = self.var_address.v[diff_var.name]
                elif isinstance(diff_var, IdxVar):
                    var_idx = diff_var.index
                    var_name = diff_var.name0
                    if is_integer(var_idx):
                        variable_address = self.var_address.v[var_name][var_idx: var_idx + 1]
                    elif isinstance(var_idx, str):
                        variable_address = self.var_address.v[var_name][np.ix_(self.PARAM[var_idx].v.reshape((-1,)))]
                    elif isinstance(var_idx, slice):
                        variable_address = self.var_address.v[var_name][var_idx]
                    elif isinstance(var_idx, Expr):
                        raise TypeError(f"Index of {diff_var} cannot be sympy.Expr!")
                    elif isinstance(var_idx, list):
                        variable_address = self.var_address.v[var_name][var_idx]
                    else:
                        raise TypeError(f"Unsupported variable index {var_idx} in equation {eqn_name}")
                else:
                    raise NotImplementedError
                eqn_address_list = equation_address.tolist()
                var_address_list = variable_address.tolist()
                if len(eqn_address_list) != len(var_address_list):
                    raise ValueError(
                        f"Incompatible eqn address length {len(eqn_address_list)} and variable address length {len(var_address_list)}")
                row.extend(eqn_address_list)
                col.extend(var_address_list)
            else:
                raise ValueError("Equation in f_list is non-Ode.")

        if self.matrix_container == 'scipy':
            return csc_array((np.ones((self.state_num,)), (row, col)), (self.eqn_size, self.vsize))
        elif self.matrix_container == 'cvxopt':
            raise NotImplementedError("Not implemented!")
        else:
            raise ValueError(f"Unsupported matrix container {self.matrix_container}")

    @property
    def alg_eqn_addr(self):
        addr_list = []
        for eqn in self.g_list:
            addr_list.extend(self.a.v[eqn].tolist())

        return addr_list

    def __repr__(self):
        if not self.eqn_size:
            return f"DAE {self.name} with addresses uninitialized"
        else:
            return f"DAE {self.name} ({self.eqn_size}×{self.vsize})"
