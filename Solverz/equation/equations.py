from __future__ import annotations

import warnings
from numbers import Number
from typing import Union, List, Dict, Tuple
from copy import deepcopy

import numpy as np
from sympy import symbols, Symbol, Expr
from scipy.sparse import csc_array, coo_array
from cvxopt import spmatrix, matrix

from Solverz.equation.eqn import Eqn, Ode, EqnDiff
from Solverz.event import Event
from Solverz.param import Param
from Solverz.num.num_alg import Var, Param_, Const_, idx
from Solverz.variable.variables import Vars
from Solverz.auxiliary import Address, combine_Address


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

        if isinstance(eqn, Eqn):
            eqn = [eqn]

        for eqn_ in eqn:
            self.add_eqn(eqn_)

        self.PARAM: Dict[str, Param] = dict()
        self.CONST: Dict[str, Param] = dict()

        for symbol_ in self.SYMBOLS.values():
            if isinstance(symbol_, Param_):
                self.PARAM[symbol_.name] = Param(symbol_.name, value=symbol_.value, dim=symbol_.dim)
            elif isinstance(symbol_, idx):
                self.PARAM[symbol_.name] = Param(symbol_.name, value=symbol_.value, dtype=int)
            elif isinstance(symbol_, Const_):
                self.CONST[symbol_.name] = Param(symbol_.name, value=symbol_.value, dim=symbol_.dim)

        for eqn in self.EQNs.values():
            eqn.derive_derivative()

    def add_eqn(self, eqn: Eqn):
        self.EQNs.update({eqn.name: eqn})
        self.SYMBOLS.update(eqn.SYMBOLS)
        self.a.add(eqn.name)
        if isinstance(eqn, Eqn) and not isinstance(eqn, Ode):
            self.g_list = self.g_list + [eqn.name]
        elif isinstance(eqn, Ode):
            self.f_list = self.f_list + [eqn.name]

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

    def param_initializer(self, name, param: Param):
        if not self.is_param_defined(name):
            raise ValueError(f'Parameter {name} not defined in equations!')
        if isinstance(param, Param):
            if name in self.PARAM:
                self.PARAM[name] = param
            elif name in self.CONST:
                self.CONST[name] = param
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
                else:
                    self.CONST[param].v = value
            except KeyError:
                warnings.warn(f'Equations have no parameter/constant: {param}')
        elif isinstance(args[0], Vars):
            # Update params with Vars. For example, to update x0 in trapezoid rules.
            vars_: Vars = args[0]
            for param_name in self.PARAM.keys():
                if param_name in vars_.var_list:
                    self.PARAM[param_name].v = vars_[param_name]
        elif isinstance(args[0], Event):
            # Update params with Event object
            event = args[0]
            t = args[1]
            for param_name in event.var_value.keys():
                temp = event.index[param_name]
                if temp is not None:
                    self.PARAM[param_name].v[temp] = event.interpolate(param_name, t)
                else:
                    self.PARAM[param_name].v = event.interpolate(param_name, t)

    def eval(self, eqn_name: str, *args: Union[np.ndarray]) -> np.ndarray:
        """
        Evaluate equations
        :param eqn_name:
        :param args:
        :return:
        """
        return self.EQNs[eqn_name].NUM_EQN(*args)

    def obtain_eqn_args(self, eqn: Eqn, *xys: Vars) -> List[np.ndarray]:
        """
        Obtain the args of equations
        :param eqn:
        :param xys:
        :return:
        """
        args = []
        for symbol in eqn.SYMBOLS.values():
            value_obtained = False
            if symbol.name in self.PARAM:
                if self.PARAM[symbol.name].triggerable:
                    for y in xys:
                        if self.PARAM[symbol.name].trigger_var in y.var_list:
                            self.PARAM[symbol.name].v = self.PARAM[symbol.name].trigger_fun(
                                y[self.PARAM[symbol.name].trigger_var])
                temp = self.PARAM[symbol.name].v
                if temp is None:
                    raise TypeError(f'Parameter {symbol.name} uninitialized')
                args = [*args, temp]
                value_obtained = True
            elif symbol.name in self.CONST:
                temp = self.CONST[symbol.name].v
                if temp is None:
                    raise TypeError(f'Constant {symbol.name} uninitialized')
                args = [*args, temp]
                value_obtained = True
            else:
                for y in xys:
                    if symbol.name in y.var_list:
                        args = [*args, y[symbol.name]]
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

    def parse_address(self, eqn_name, var_name, eqndiff, *xys):
        var_idx = eqndiff.var_idx
        var_idx_func = eqndiff.var_idx_func

        equation_address = self.a.v[eqn_name]
        if var_idx is None:
            variable_address = self.var_address.v[var_name]
        elif isinstance(var_idx, (float, int)):
            variable_address = self.var_address.v[var_name][var_idx: var_idx + 1]
        elif isinstance(var_idx, str):
            variable_address = self.var_address.v[var_name][np.ix_(self.PARAM[var_idx].v.reshape((-1,)))]
        elif isinstance(var_idx, slice):
            temp = [None, None, None]
            for i in range(3):
                if isinstance(var_idx_func[i], Eqn):
                    temp[i] = int(
                        var_idx_func[i].NUM_EQN(*self.obtain_eqn_args(var_idx_func[i], *xys)))
                else:
                    temp[i] = var_idx_func[i]
            if temp[1] is not None:
                temp[1] = temp[1] + 1
            variable_address = self.var_address.v[var_name][slice(*temp)]
        elif isinstance(var_idx, Expr):
            args = self.obtain_eqn_args(var_idx_func, *xys)
            variable_address = self.var_address.v[var_name][var_idx_func.NUM_EQN(args)]
        elif isinstance(var_idx, list):
            variable_address = self.var_address.v[var_name][var_idx]
        else:
            raise TypeError(f"Unsupported variable index {var_idx} for equation {eqn_name}")

        return equation_address, variable_address

    def form_jac(self, gy, *xys):

        if self.matrix_container == 'cvxopt':

            jac = spmatrix([], [], [], (self.eqn_size, self.vsize))

            for gy_tuple in gy:
                eqn_name = gy_tuple[0]
                var_name = gy_tuple[1]
                eqndiff = gy_tuple[2]
                value = gy_tuple[3]
                equation_address, variable_address = self.parse_address(eqn_name,
                                                                        var_name,
                                                                        eqndiff,
                                                                        *xys)

                if isinstance(value, (np.ndarray, csc_array)):
                    # we use `+=` instead of `=` here because sometimes, Var `e` and IdxVar `e[0]` exists in the same equation
                    # in which case we have to add the jacobian element of Var `e` if it is not zero.

                    ## cvxopt.spmatrix
                    if value.ndim > 1:
                        if value.shape[1] > 1:  # np.ix_() creates a mesh for matrix
                            if not isinstance(value, np.ndarray):
                                value1 = value.tocoo()
                            else:
                                value1 = coo_array(value)
                            I = matrix(equation_address)
                            J = matrix(variable_address)
                            jac[I, J] = \
                                jac[I, J] + spmatrix(value1.data, value1.row, value1.col, value1.shape, 'd')
                        else:  # equation and variable lists constitute a address tuple for vector
                            I = (equation_address + variable_address * self.vsize).tolist()
                            jac[I] = jac[I] + matrix(value)
                    else:
                        I = (equation_address + variable_address * self.vsize).tolist()
                        jac[I] = jac[I] + matrix(value.tolist())
            return jac
        elif self.matrix_container == 'scipy':

            row = []
            col = []
            data = []

            for gy_tuple in gy:
                eqn_name = gy_tuple[0]
                var_name = gy_tuple[1]
                eqndiff = gy_tuple[2]
                value = gy_tuple[3]
                equation_address, variable_address = self.parse_address(eqn_name,
                                                                        var_name,
                                                                        eqndiff,
                                                                        *xys)

                if isinstance(value, (np.ndarray, csc_array)):
                    # we use `+=` instead of `=` here because sometimes, Var `e` and IdxVar `e[0]` exists in the same equation
                    # in which case we have to add the jacobian element of Var `e` if it is not zero.

                    if value.ndim > 1:
                        if value.shape[1] > 1:
                            # matrix
                            if isinstance(value, csc_array):
                                coo_ = value.tocoo()
                            else:
                                coo_ = coo_array(value)
                            data.extend(coo_.data.tolist())
                            # map local addresses of coo_ to equation_address and variable address
                            row.extend(equation_address[coo_.row].tolist())
                            col.extend(variable_address[coo_.col].tolist())
                        else:
                            # vector

                            if value.shape[0] < len(equation_address):
                                # two dimensional scalar as vector todo: this should be avoided
                                data.extend(value.reshape(-1).tolist() * len(equation_address))
                            else:  # vector
                                data.extend(value.reshape(-1).tolist())
                            row.extend(equation_address.tolist())
                            col.extend(variable_address.tolist())
                    else:
                        # scalar
                        data.extend([value.tolist()] * len(equation_address))
                        row.extend(equation_address.tolist())
                        col.extend(variable_address.tolist())
            return coo_array((data, (row, col)), (self.eqn_size, self.vsize)).tocsc()


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
            eqn_size = self.g(y, eqn_name).shape[0]
            self.a.update(eqn_name, temp, temp + eqn_size - 1)
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
        if not eqn:
            temp = np.array([])
            for eqn_name, eqn_ in self.EQNs.items():
                args = self.obtain_eqn_args(eqn_, y)
                g_eqny = self.eval(eqn_name, *args)
                g_eqny = g_eqny.toarray() if isinstance(g_eqny, csc_array) else g_eqny
                temp = np.concatenate([temp, g_eqny.reshape(-1, )])
            return temp
        else:
            args = self.obtain_eqn_args(self.EQNs[eqn], y)
            return self.eval(eqn, *args)

    def g_y(self, y: Vars, eqn: List[str] = None, var: List[str] = None) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
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
                        if not isinstance(temp, csc_array):
                            temp = np.array(temp)
                        gy = [*gy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return gy

    def j(self, y: Vars) -> spmatrix:
        if not self.eqn_size:
            self.assign_eqn_var_address(y)

        gy = self.g_y(y)

        return self.form_jac(gy, y)

    def __repr__(self):
        if not self.eqn_size:
            return f"Algebraic equation {self.name} with addresses uninitialized"
        else:
            return f"Algebraic equation {self.name} ({self.eqn_size}×{self.vsize})"


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

    def assign_eqn_var_address(self, *xys: Vars):
        """
        ASSIGN ADDRESSES TO EQUATIONS f and g
        """

        temp = 0
        for eqn_name in self.f_list:
            eqn_size = self.f(eqn_name, *xys).shape[0]
            self.a.update(eqn_name, temp, temp + eqn_size - 1)
            temp = temp + eqn_size
            self.esize[eqn_name] = eqn_size

        self.state_num = temp

        for eqn_name in self.g_list:
            eqn_size = self.g(eqn_name, *xys).shape[0]
            self.a.update(eqn_name, temp, temp + eqn_size - 1)
            temp = temp + eqn_size
            self.esize[eqn_name] = eqn_size

        self.algebra_num = temp - self.state_num

        if len(xys) == 1:
            self.var_address.v = xys[0].a.v
            self.vsize = self.var_address.total_size
        elif len(xys) == 2:
            x = xys[0]
            y = xys[1]
            if x.total_size != self.state_num:
                raise ValueError("Length of input state variable not compatible with state equations!")
            if y.total_size != self.algebra_num:
                raise ValueError("Length of input algebraic variable not compatible with algebraic equations!")
            self.var_address = combine_Address(x.a, y.a)

            for var_name in self.var_address.v.keys():
                if not var_name in self.SYMBOLS:
                    raise ValueError(f"DAE {self.name} has no variable {var_name}")

            self.vsize: int = self.var_address.total_size
        elif len(xys) > 2:
            raise ValueError("Accept at most two positional arguments!")

    def F(self, *xys) -> np.ndarray:
        """
        Return [f(x,y), g(x,y)]
        :param xys:
        :return:
        """
        if len(self.g_list) > 0:
            return np.concatenate([self.f(*xys), self.g(*xys)])
        else:
            return self.f(*xys)

    def f(self, *xys) -> np.ndarray:
        """

        `xys` is either:
          - two arguments, e.g. state vars x, and numerical equation y
          - one argument, e.g. state vars x.
        """
        eqn = None
        if isinstance(xys[0], str):
            eqn = xys[0]
            xys = xys[1:]

        temp = np.array([])
        if eqn:
            if eqn in self.f_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], *xys)
                temp = np.concatenate([temp, self.eval(eqn, *args).reshape(-1, )])
                return temp
        else:
            for eqn in self.f_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], *xys)
                temp = np.concatenate([temp, self.eval(eqn, *args).reshape(-1, )])
            return temp

    def g(self, *xys) -> np.ndarray:
        """

        `xys` is either:
          - two arguments, e.g. state vars x, and numerical equation y
          - one argument, e.g. state vars y.

        """

        if len(self.g_list) == 0:
            raise ValueError(f'No AE found in {self.name}!')

        eqn = None
        if isinstance(xys[0], str):
            eqn = xys[0]
            xys = xys[1:]

        temp = np.array([])
        if eqn:
            if eqn in self.g_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], *xys)
                temp = np.concatenate([temp, self.eval(eqn, *args).reshape(-1, )])
                return temp
        else:
            for eqn in self.g_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], *xys)
                temp = np.concatenate([temp, self.eval(eqn, *args).reshape(-1, )])
        return temp

    def f_xy(self, *xys: Vars) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        """
        generate partial derivatives of f w.r.t. vars in xys
        :return: List[Tuple[Equation_name, var_name, np.ndarray]]
        """

        fxy: List[Tuple[str, str, EqnDiff, np.ndarray]] = []

        var: List[str] = list()
        for xy in xys:
            var = var + xy.var_list

        for eqn_name in self.f_list:
            eqn_diffs: Dict[str, EqnDiff] = self.EQNs[eqn_name].derivatives
            for var_name in var:
                for key, value in eqn_diffs.items():
                    if var_name == value.diff_var_name:
                        args = self.obtain_eqn_args(eqn_diffs[key], *xys)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        if not isinstance(temp, csc_array):
                            temp = np.array(temp)
                        fxy = [*fxy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return fxy

    def g_xy(self, *xys: Vars) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
        """
        generate partial derivatives of f w.r.t. vars in xys
        :return: List[Tuple[Equation_name, var_name, np.ndarray]]
        """

        if len(self.g_list) == 0:
            raise ValueError(f'No AE found in {self.name}!')

        gxy: List[Tuple[str, str, EqnDiff, np.ndarray]] = []

        var: List[str] = list()
        for xy in xys:
            var = var + xy.var_list

        for eqn_name in self.g_list:
            eqn_diffs: Dict[str, EqnDiff] = self.EQNs[eqn_name].derivatives
            for var_name in var:
                for key, value in eqn_diffs.items():
                    if var_name == value.diff_var_name:
                        args = self.obtain_eqn_args(eqn_diffs[key], *xys)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        if not isinstance(temp, csc_array):
                            temp = np.array(temp)
                        gxy = [*gxy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return gxy

    def j(self, *xys: Vars, matrix_container='scipy'):
        """
        Derive Jacobian matrices of the RHS side
        """
        if not self.eqn_size:
            self.assign_eqn_var_address(*xys)

        fg_xy = self.f_xy(*xys)
        if len(self.g_list) > 0:
            fg_xy = fg_xy + self.g_xy(*xys)

        return self.form_jac(fg_xy, *xys, matrix_container)

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
            return False
        else:
            return True

    def __repr__(self):
        return f"DAE: {self.name}"
