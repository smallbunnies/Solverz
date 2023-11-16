from __future__ import annotations

import warnings
from numbers import Number
from typing import Union, List, Dict, Tuple

import numpy as np
from sympy import symbols, Symbol, Expr
from scipy.sparse import csc_array, coo_array
from cvxopt import spmatrix, matrix

from Solverz.equation.eqn import Eqn, Ode, EqnDiff
from Solverz.event import Event
from Solverz.param import Param
from Solverz.num.num_alg import Var, Param_, Const_, idx
from Solverz.variable.variables import Vars
from Solverz.auxiliary import Address


class Equations:

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None):
        self.name = name

        self.EQNs: Dict[str, Eqn] = dict()
        self.SYMBOLS: Dict[str, Symbol] = dict()
        self.a = Address()  # equation address
        self.esize: Dict[str, int] = dict()  # size of each equation
        self.vsize: int = 0  # size of variables

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

    def add_eqn(self, eqn: Eqn):
        self.EQNs.update({eqn.name: eqn})
        self.SYMBOLS.update(eqn.SYMBOLS)
        self.a.add(eqn.name)

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


class AE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None):
        super().__init__(eqn, name)
        for eqn in self.EQNs.values():
            eqn.derive_derivative()

        self.j_cache = csc_array((0, 0))

        # Check if some equation in self.eqn is Eqn.
        # If not, raise error
        if any([isinstance(eqn_, Ode) for eqn_ in self.EQNs.values()]):
            raise ValueError(f'Ode found. This object should be DAE!')

    def assign_equation_address(self, y: Vars):
        """
        ASSIGN ADDRESSES TO EQUATIONS
        """
        temp = 0
        for eqn_name in self.EQNs.keys():
            self.a.update(eqn_name, temp, temp + self.g(y, eqn_name).shape[0] - 1)
            temp = temp + self.g(y, eqn_name).shape[0]
            self.esize[eqn_name] = self.g(y, eqn_name).shape[0]
        self.vsize = y.total_size
        self.j_cache = csc_array((self.eqn_size, y.total_size))

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

    def j(self, y: Vars, ) -> spmatrix:
        if not self.eqn_size:
            self.assign_equation_address(y)

        gy = self.g_y(y)
        self.j_cache = spmatrix([], [], [], (self.eqn_size, self.vsize))

        for gy_tuple in gy:
            eqn_name = gy_tuple[0]
            var_name = gy_tuple[1]
            var_idx = gy_tuple[2].var_idx
            value = gy_tuple[3]

            equation_address = self.a.v[eqn_name]
            if var_idx is None:
                variable_address = y.a.v[var_name]
            elif isinstance(var_idx, (float, int)):
                variable_address = y.a.v[var_name][var_idx: var_idx + 1]
            elif isinstance(var_idx, str):
                variable_address = y.a.v[var_name][np.ix_(self.PARAM[var_idx].v.reshape((-1,)))]
            elif isinstance(var_idx, slice):
                temp = [None, None, None]
                for i in range(3):
                    if isinstance(gy_tuple[2].var_idx_func[i], Eqn):
                        temp[i] = int(
                            gy_tuple[2].var_idx_func[i].NUM_EQN(*self.obtain_eqn_args(gy_tuple[2].var_idx_func[i], y)))
                    else:
                        temp[i] = gy_tuple[2].var_idx_func[i]
                if temp[1] is not None:
                    temp[1] = temp[1] + 1
                variable_address = y.a.v[var_name][slice(*temp)]
            elif isinstance(var_idx, Expr):
                args = self.obtain_eqn_args(gy_tuple[2].var_idx_func, y)
                variable_address = y.a.v[var_name][gy_tuple[2].var_idx_func.NUM_EQN(args)]
            elif isinstance(var_idx, list):
                variable_address = y.a.v[var_name][var_idx]
            else:
                raise TypeError(f"Unsupported variable index {var_idx} for equation {eqn_name}")

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
                        self.j_cache[I, J] = \
                            self.j_cache[I, J] + spmatrix(value1.data, value1.row, value1.col, value1.shape, 'd')
                    else:  # equation and variable lists constitute a address tuple for vector
                        I = (equation_address + variable_address * self.vsize).tolist()
                        self.j_cache[I] = self.j_cache[I] + matrix(value)
                else:
                    I = (equation_address + variable_address * self.vsize).tolist()
                    self.j_cache[I] = self.j_cache[I] + matrix(value.tolist())

        return self.j_cache

    def __repr__(self):
        if not self.eqn_size:
            return f"Algebraic equation {self.name} with addresses uninitialized"
        else:
            return f"Algebraic equation {self.name} ({self.eqn_size}×{self.vsize})"


class DAE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None
                 ):

        self.f_dict: Dict[str, Ode] = dict()  # dict of state equations
        self.g_dict: Dict[str, Eqn] = dict()  # dict of algebraic equations
        self.var_address: Dict[str, List[int]] = dict()

        super().__init__(eqn, name)

        self.state_num: int = 0  # number of state variables
        self.algebra_num: int = 0  # number of algebraic variables

        # Check if some equation in self.eqn is Ode.
        # If not, raise error
        if not self.f_dict:
            raise ValueError(f'No ODE found. You should initialise AE instead!')

    def add_eqn(self, eqn: Union[Eqn, Ode]):
        self.EQNs[eqn.name] = eqn
        for symbol_ in self.EQNs[eqn.name].SYMBOLS.values():
            # generate bare symbols without assumptions
            self.SYMBOLS[symbol_.name] = symbols(symbol_.name)
        self.a[eqn.name] = []
        if isinstance(eqn, Ode):
            self.f_dict[eqn.name] = eqn
        elif isinstance(eqn, Eqn):
            self.g_dict[eqn.name] = eqn
        else:
            raise ValueError(f'Undefined equation: {eqn.__class__.__name__}')

    def assign_eqn_var_address(self, *xys: Vars):
        """
        ASSIGN ADDRESSES TO EQUATIONS f and g
        """

        temp = 0
        for eqn_name in self.f_dict.keys():
            eqn_size = self.f(eqn_name, *xys).shape[0]
            self.a[eqn_name] = [temp, temp + eqn_size - 1]
            temp = temp + eqn_size
            self.esize[eqn_name] = eqn_size

        self.state_num = temp

        for eqn_name in self.g_dict.keys():
            eqn_size = self.g(eqn_name, *xys).shape[0]
            self.a[eqn_name] = [temp, temp + eqn_size - 1]
            temp = temp + eqn_size
            self.esize[eqn_name] = eqn_size

        self.algebra_num = temp - self.state_num

        temp = 0
        for xy in xys:
            for var_name, a in xy.a.items():
                if var_name in self.SYMBOLS:
                    self.var_address[var_name] = [temp + xy.a[var_name][0], xy.a[var_name][-1] + temp]
                else:
                    raise ValueError(f"DAE {self.name} has no variable {var_name}")
            temp = temp + xy.total_size

        self.vsize: int = temp

    def f(self, *xys) -> np.ndarray:
        """

        `args` is either:
          - two arguments, e.g. state vars x, and numerical equation y
          - one argument, e.g. state vars x.
        """
        eqn = None
        if isinstance(xys[0], str):
            eqn = xys[0]
            xys = xys[1:]

        temp = np.array([])
        if eqn:
            if eqn in self.f_dict.keys():
                args = self.obtain_eqn_args(self.f_dict[eqn], *xys)
                temp = np.concatenate([temp, self.eval(eqn, *args).reshape(-1, )])
                return temp
        else:
            for eqn_name, eqn_ in self.f_dict.items():
                args = self.obtain_eqn_args(eqn_, *xys)
                temp = np.concatenate([temp, self.eval(eqn_name, *args).reshape(-1, )])
            return temp

    def g(self, *xys) -> np.ndarray:
        """

        `xys` is either:
          - two arguments, e.g. state vars x, and numerical equation y
          - one argument, e.g. state vars y.

        """

        if not self.g_dict:
            raise ValueError(f'No AE found in {self.name}!')

        eqn = None
        if isinstance(xys[0], str):
            eqn = xys[0]
            xys = xys[1:]

        temp = np.array([])
        if eqn:
            if eqn in self.g_dict.keys():
                args = self.obtain_eqn_args(self.g_dict[eqn], *xys)
                temp = np.concatenate([temp, self.eval(eqn, *args).reshape(-1, )])
                return temp
        else:
            for eqn_name, eqn_ in self.g_dict.items():
                args = self.obtain_eqn_args(eqn_, *xys)
                temp = np.concatenate([temp, self.eval(eqn_name, *args).reshape(-1, )])
        return temp

    def f_xy(self, *xys: Vars) -> List[Tuple[str, str, np.ndarray]]:
        """
        generate partial derivatives of f w.r.t. vars in xys
        :return: List[Tuple[Equation_name, var_name, np.ndarray]]
        """

        fx: List[Tuple[str, str, np.ndarray]] = []

        var: List[str] = list()
        for xy in xys:
            var = var + list(xy.v.keys())

        for eqn_name in self.f_dict:
            for var_name in var:
                for xy in xys:
                    if var_name in list(xy.v):
                        if var_name in self.eqn_diffs[eqn_name]:
                            args = self.obtain_eqn_args(self.eqn_diffs[eqn_name][var_name], *xys)
                            temp = np.array(self.eval_diffs(eqn_name, var_name, *args))
                            if temp.ndim > 1:
                                #  matrix or row vector
                                fx = [*fx, (eqn_name, var_name, temp)]
                            elif temp.ndim == 1 and temp.shape[0] > 1:
                                # column vector
                                fx = [*fx, (eqn_name, var_name, np.diag(temp))]
                            else:
                                # [number]
                                fx = [*fx, (eqn_name, var_name, temp * np.identity(xy.var_size[var_name]))]
        return fx

    def g_xy(self, *xys: Vars) -> List[Tuple[str, str, np.ndarray]]:
        """
        generate partial derivatives of f w.r.t. vars in xys
        :return: List[Tuple[Equation_name, var_name, np.ndarray]]
        """

        if not self.g_dict:
            raise ValueError(f'No AE found in {self.name}!')

        gy: List[Tuple[str, str, np.ndarray]] = []

        var: List[str] = list()
        for xy in xys:
            var = var + list(xy.v.keys())

        for eqn_name in self.g_dict:
            for var_name in var:
                for xy in xys:
                    if var_name in list(xy.v):
                        if var_name in self.eqn_diffs[eqn_name]:
                            args = self.obtain_eqn_args(self.eqn_diffs[eqn_name][var_name], *xys)
                            temp = np.array(self.eval_diffs(eqn_name, var_name, *args))
                            if temp.ndim > 1:
                                #  matrix or row vector
                                gy = [*gy, (eqn_name, var_name, temp)]
                            elif temp.ndim == 1 and temp.shape[0] > 1:
                                # column vector
                                gy = [*gy, (eqn_name, var_name, np.diag(temp))]
                            else:
                                # [number]
                                gy = [*gy, (eqn_name, var_name, temp * np.identity(xy.var_size[var_name]))]
        return gy

    def j(self, *xys: Vars) -> np.ndarray:
        """
        Derive full or partial Jacobian matrices
        :param xys:
        :return:
        """
        if not self.eqn_size:
            self.assign_eqn_var_address(*xys)

        j = np.zeros((self.eqn_size, self.var_size))

        fxy = self.f_xy(*xys)
        for fxy_tuple in fxy:
            j[self.a[fxy_tuple[0]][0]:(self.a[fxy_tuple[0]][-1] + 1),
            self.var_address[fxy_tuple[1]][0]:(self.var_address[fxy_tuple[1]][-1] + 1)] = fxy_tuple[2]

        if self.g_dict:
            gxy = self.g_xy(*xys)
            for gxy_tuple in gxy:
                j[self.a[gxy_tuple[0]][0]:(self.a[gxy_tuple[0]][-1] + 1),
                self.var_address[gxy_tuple[1]][0]:(self.var_address[gxy_tuple[1]][-1] + 1)] = gxy_tuple[2]

        return j

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
