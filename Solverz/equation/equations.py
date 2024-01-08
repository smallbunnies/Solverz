from __future__ import annotations

import warnings
from numbers import Number
from typing import Union, List, Dict, Tuple
from copy import deepcopy

import numpy as np
from sympy import Symbol, Expr
from scipy.sparse import csc_array, coo_array
# from cvxopt import spmatrix, matrix

from Solverz.equation.eqn import Eqn, Ode, EqnDiff
from Solverz.equation.param import Param, IdxParam
from Solverz.symboli_algebra.symbols import Var, idx, IdxVar, Para, AliasVar
from Solverz.symboli_algebra.functions import Slice
from Solverz.variable.variables import Vars
from Solverz.utilities.address import Address, combine_Address


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
        self.PARAM: Dict[str, Param] = dict()
        self.triggerable_quantity: Dict[str, str] = dict()
        self.trigger_quantity_cache: Dict[str, Dict[str, np.ndarray]] = dict()
        self.jac_element_address = Address()

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
            if isinstance(symbol_, (Para, AliasVar)):
                # this is not fully initialize of Parameters, please use param_initializer
                self.PARAM[symbol_.name] = Param(symbol_.name, value=symbol_.value, dim=symbol_.dim)
            elif isinstance(symbol_, idx):
                self.PARAM[symbol_.name] = IdxParam(symbol_.name, value=symbol_.value)

        self.EQNs[eqn.name].derive_derivative()

    def assign_eqn_var_address(self, *xys: Vars):
        pass

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
            warnings.warn(f'Parameter {name} not defined in equations!')
        if isinstance(param, Param):
            self.PARAM[name] = param
            if param.triggerable:
                self.triggerable_quantity[param.name] = param.trigger_var
                self.trigger_quantity_cache[param.name] = dict()
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
        return self.EQNs[eqn_name].NUM_EQN(*args)

    def trigger_param_updater(self, eqn: Eqn, *xys):
        # taq -- triggerable_quantity that is being triggered by parameters or variables
        # tq -- trigger_quantity that is triggering parameters
        for taq, tq in self.triggerable_quantity.items():
            if taq in eqn.SYMBOLS.keys():
                if tq in self.PARAM.keys():
                    if tq not in self.trigger_quantity_cache[taq]:
                        # tq value is None then initialize cache
                        self.PARAM[taq].v = self.PARAM[taq].trigger_fun(self.PARAM[tq].v)
                        self.trigger_quantity_cache[taq][tq] = np.copy(self.PARAM[tq].v)
                    else:
                        # If tq value is not None, then compare current value with cache.
                        if not np.allclose(self.trigger_quantity_cache[taq][tq],
                                           self.PARAM[tq].v,
                                           rtol=1e-8):
                            self.PARAM[taq].v = self.PARAM[taq].trigger_fun(self.PARAM[tq].v)
                            self.trigger_quantity_cache[taq][tq] = np.copy(self.PARAM[tq].v)
                else:
                    for y in xys:
                        if tq in y.var_list:
                            if tq not in self.trigger_quantity_cache[taq]:
                                # tq value is None then initialize cache
                                self.PARAM[taq].v = self.PARAM[taq].trigger_fun(y[tq])
                                self.trigger_quantity_cache[taq][tq] = np.copy(y[tq])
                            else:
                                # If tq value is not None, then compare current value with cache.
                                if not np.allclose(self.trigger_quantity_cache[taq][tq],
                                                   y[tq],
                                                   rtol=1e-8):
                                    self.PARAM[taq].v = self.PARAM[taq].trigger_fun(y[tq])
                                    self.trigger_quantity_cache[taq][tq] = np.copy(y[tq])

    def obtain_eqn_args(self, eqn: Eqn, t=None, *xys: Vars) -> List[np.ndarray]:
        """
        Obtain the args of equations
        """

        self.trigger_param_updater(eqn, *xys)

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
                for y in xys:
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

    def parse_address(self, eqn_name, var_name, eqndiff, t=None, *xys):
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
            variable_address = self.var_address.v[var_name][
                var_idx_func.NUM_EQN(*self.obtain_eqn_args(var_idx_func, t, *xys))]
        elif isinstance(var_idx, Expr):
            args = self.obtain_eqn_args(var_idx_func, *xys)
            variable_address = self.var_address.v[var_name][var_idx_func.NUM_EQN(*args).reshape(-1, )]
        elif isinstance(var_idx, list):
            variable_address = self.var_address.v[var_name][var_idx]
        else:
            raise TypeError(f"Unsupported variable index {var_idx} for equation {eqn_name}")

        return equation_address, variable_address

    def form_jac(self, gy, t=None, *xys):

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
                                                                        t,
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
                                                                        t,
                                                                        *xys)

                if isinstance(value, (np.ndarray, csc_array)):
                    # We use coo_array here because by default when converting to CSR or CSC format,
                    # duplicate (i,j) entries will be summed together.
                    # This facilitates efficient construction of finite element matrices and the like.

                    if value.ndim == 2:  # matrix
                        if isinstance(value, csc_array):
                            coo_ = value.tocoo()
                        else:
                            coo_ = coo_array(value)
                        data.extend(coo_.data.tolist())
                        # map local addresses of coo_ to equation_address and variable address
                        row.extend(equation_address[coo_.row].tolist())
                        col.extend(variable_address[coo_.col].tolist())
                    elif value.ndim == 1:  # vector
                        if value.shape[0] < len(equation_address):
                            # two dimensional scalar as vector todo: this should be avoided
                            data.extend(value.reshape(-1).tolist() * len(equation_address))
                        else:  # vector
                            data.extend(value.reshape(-1).tolist())
                        row.extend(equation_address.tolist())
                        col.extend(variable_address.tolist())
                    else:  # np.dim ==0  scalar in np.ndarray
                        value = np.squeeze(value)  # In inviscid burger test there is scalar like np.array([1])
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

    def assign_eqn_var_address(self, *xys: Vars):
        """
        ASSIGN ADDRESSES TO EQUATIONS
        """
        y = xys[0]

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

        # assign dim of derivatives, which is indispensable in Jac printer
        gy = self.g_y(y)
        for gy_tuple in gy:
            eqn_name = gy_tuple[0]
            var_name = gy_tuple[1]
            eqndiff = gy_tuple[2]
            value = gy_tuple[3]
            if isinstance(value, (np.ndarray, csc_array)):
                # We use coo_array here because by default when converting to CSR or CSC format,
                # duplicate (i,j) entries will be summed together.
                # This facilitates efficient construction of finite element matrices and the like.
                if value.ndim == 2:  # matrix
                    self.EQNs[eqn_name].derivatives[eqndiff.diff_var.name].dim = 2
                    warnings.warn('Address assignment of matrix derivative unimplemented!')
                elif value.ndim == 1 and value.shape[0] != 1:  # vector
                    self.EQNs[eqn_name].derivatives[eqndiff.diff_var.name].dim = 1
                    # self.jac_element_address.add((eqn_name, var_name, eqndiff), value.shape[0])
                else:  # np.dim ==0  scalar in np.ndarray
                    self.EQNs[eqn_name].derivatives[eqndiff.diff_var.name].dim = 0
                    # self.jac_element_address.add((eqn_name, var_name, eqndiff), self.a.size[eqn_name])

    def g(self, y: Vars, eqn: str = None) -> np.ndarray:
        """

        :param y:
        :param eqn:
        :return:
        """
        temp = []
        if not eqn:
            for eqn_name, eqn_ in self.EQNs.items():
                args = self.obtain_eqn_args(eqn_, None, y)
                g_eqny = self.eval(eqn_name, *args)
                g_eqny = g_eqny.toarray() if isinstance(g_eqny, csc_array) else g_eqny
                temp.append(g_eqny.reshape(-1, ))
            return np.hstack(temp)
        else:
            args = self.obtain_eqn_args(self.EQNs[eqn], None, y)
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
                        args = self.obtain_eqn_args(eqn_diffs[key], None, y)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        if not isinstance(temp, csc_array):
                            temp = np.array(temp)
                        gy = [*gy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return gy

    def j(self, y: Vars):
        if not self.eqn_size:
            self.assign_eqn_var_address(y)

        gy = self.g_y(y)

        return self.form_jac(gy, None, y)

    def __repr__(self):
        if not self.eqn_size:
            return f"Algebraic equation {self.name} with addresses uninitialized"
        else:
            return f"Algebraic equation {self.name} ({self.eqn_size}×{self.vsize})"


class tAE(AE):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 matrix_container='scipy'):
        super().__init__(eqn, name, matrix_container)

        dt = Para('dt')
        t0 = Para('t0')
        t = Var('t')
        self.add_eqn(Eqn('eqn of time', t - t0 - dt))

    def obtain_eqn_args(self, eqn: Eqn, t=None, *xys: Vars) -> List[np.ndarray]:
        for y in xys:
            if 't' in y.var_list:
                return super().obtain_eqn_args(eqn, y['t'], *xys)
        raise KeyError("Simulation time t undefined in AEs by discretizing DAE!")


class FDAE(AE):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 nstep: int,
                 name: str = None,
                 matrix_container='scipy'):
        super().__init__(eqn, name, matrix_container)

        self.nstep = nstep


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
            feval = self.f(None, *xys, eqn=eqn_name)
            if isinstance(feval, Number):
                eqn_size = 1
            else:
                eqn_size = feval.shape[0]
            self.a.update(eqn_name, eqn_size)
            temp = temp + eqn_size
            self.esize[eqn_name] = eqn_size

        self.state_num = temp

        for eqn_name in self.g_list:
            geval = self.g(None, *xys, eqn=eqn_name)
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

        if len(xys) == 1:
            self.var_address = xys[0].a
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
                if var_name not in self.SYMBOLS:
                    raise ValueError(f"DAE {self.name} has no variable {var_name}")

            self.vsize: int = self.var_address.total_size
        elif len(xys) > 2:
            raise ValueError("Accept at most two positional arguments!")

        # assign dim of derivatives, which is indispensable in Jac printer
        fg_xy = self.f_xy(0, *xys)
        if len(self.g_list) > 0:
            fg_xy.extend(self.g_xy(0, *xys))
        for gy_tuple in fg_xy:
            eqn_name = gy_tuple[0]
            var_name = gy_tuple[1]
            eqndiff = gy_tuple[2]
            value = gy_tuple[3]
            if isinstance(value, (np.ndarray, csc_array)):
                # We use coo_array here because by default when converting to CSR or CSC format,
                # duplicate (i,j) entries will be summed together.
                # This facilitates efficient construction of finite element matrices and the like.
                if value.ndim == 2:  # matrix
                    self.EQNs[eqn_name].derivatives[eqndiff.diff_var.name].dim = 2
                    warnings.warn('Address assignment of matrix derivative unimplemented!')
                elif value.ndim == 1 and value.shape[0] != 1:  # vector
                    self.EQNs[eqn_name].derivatives[eqndiff.diff_var.name].dim = 1
                    # self.jac_element_address.add((eqn_name, var_name, eqndiff), value.shape[0])
                else:  # np.dim ==0  scalar in np.ndarray
                    self.EQNs[eqn_name].derivatives[eqndiff.diff_var.name].dim = 0
                    # self.jac_element_address.add((eqn_name, var_name, eqndiff), self.a.size[eqn_name])

    def F(self, t, *xys) -> np.ndarray:
        """
        Return [f(t,x,y), g(t,x,y)]
        :param t: time
        :param xys: Vars
        :return:
        """
        if len(self.g_list) > 0:
            return np.concatenate([self.f(t, *xys), self.g(t, *xys)])
        else:
            return self.f(t, *xys)

    def f(self, t, *xys, eqn=None) -> np.ndarray:

        temp = []
        if eqn:
            if eqn in self.f_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], t, *xys)
                temp.append(self.eval(eqn, *args).reshape(-1, ))
        else:
            for eqn in self.f_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], t, *xys)
                temp.append(self.eval(eqn, *args).reshape(-1, ))

        return np.hstack(temp)

    def g(self, t, *xys, eqn=None) -> np.ndarray:
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
                args = self.obtain_eqn_args(self.EQNs[eqn], t, *xys)
                temp.append(self.eval(eqn, *args).reshape(-1, ))
        else:
            for eqn in self.g_list:
                args = self.obtain_eqn_args(self.EQNs[eqn], t, *xys)
                temp.append(self.eval(eqn, *args).reshape(-1, ))

        return np.hstack(temp)

    def f_xy(self, t, *xys: Vars) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
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
                        args = self.obtain_eqn_args(eqn_diffs[key], t, *xys)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        if not isinstance(temp, csc_array):
                            temp = np.array(temp)
                        fxy = [*fxy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return fxy

    def g_xy(self, t, *xys: Vars) -> List[Tuple[str, str, EqnDiff, np.ndarray]]:
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
                        args = self.obtain_eqn_args(eqn_diffs[key], t, *xys)
                        temp = self.eval_diffs(eqn_name, key, *args)
                        if not isinstance(temp, csc_array):
                            temp = np.array(temp)
                        gxy = [*gxy, (eqn_name, var_name, eqn_diffs[key], temp)]
        return gxy

    def j(self, t, *xys: Vars):
        """
        Derive Jacobian matrices of the RHS side
        """
        if not self.eqn_size:
            self.assign_eqn_var_address(*xys)

        fg_xy = self.f_xy(t, *xys)
        if len(self.g_list) > 0:
            fg_xy.extend(self.g_xy(t, *xys))

        return self.form_jac(fg_xy, t, *xys)

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
                if isinstance(diff_var, Var):
                    variable_address = self.var_address.v[diff_var.name]
                elif isinstance(diff_var, IdxVar):
                    var_idx = diff_var.index
                    var_name = diff_var.name0
                    if isinstance(var_idx, (float, int)):
                        variable_address = self.var_address.v[var_name][var_idx: var_idx + 1]
                    elif isinstance(var_idx, str):
                        variable_address = self.var_address.v[var_name][np.ix_(self.PARAM[var_idx].v.reshape((-1,)))]
                    elif isinstance(var_idx, slice):
                        temp = []
                        if var_idx.start is not None:
                            temp.append(var_idx.start)
                        if var_idx.stop is not None:
                            temp.append(var_idx.stop)
                        if var_idx.step is not None:
                            temp.append(var_idx.step)
                        temp_func = Eqn('To evaluate var_idx of variable' + diff_var.name, Slice(*temp))
                        variable_address = self.var_address.v[var_name][
                            temp_func.NUM_EQN(*self.obtain_eqn_args(temp_func))]
                    elif isinstance(var_idx, Expr):
                        temp_func = Eqn('To evaluate var_idx of variable' + diff_var.name, var_idx)
                        args = self.obtain_eqn_args(temp_func)
                        variable_address = self.var_address.v[var_name][temp_func.NUM_EQN(*args).reshape(-1, )]
                    elif isinstance(var_idx, list):
                        variable_address = self.var_address.v[var_name][var_idx]
                    else:
                        raise TypeError(f"Unsupported variable index {var_idx} for equation {eqn_name}")
                else:
                    raise NotImplementedError
                row.extend(equation_address.tolist())
                col.extend(variable_address.tolist())
            else:
                raise ValueError("Equation in f_list is non-Ode.")

        if self.matrix_container == 'scipy':
            return csc_array((np.ones((self.state_num,)), (row, col)), (self.eqn_size, self.vsize))
        elif self.matrix_container == 'cvxopt':
            raise NotImplementedError("Not implemented!")
        else:
            raise ValueError(f"Unsupported matrix container {self.matrix_container}")

    def discretize(self, scheme=1) -> AE:
        if scheme == 1:
            # trapezoidal method
            trapezoidal_ae = tAE(name='trapezoidal_ae', eqn=[self.EQNs[ae] for ae in self.g_list])

            dt = Para('dt')
            for ode in self.f_list:
                var_list = []
                for symbol_ in list(self.EQNs[ode].RHS.free_symbols):
                    if isinstance(symbol_, Var):
                        var_list.append((symbol_, Para(symbol_.name + '0')))
                    elif isinstance(symbol_, IdxVar):
                        var_list.append((symbol_, Para(symbol_.name0 + '0')[symbol_.index]))
                diff_var = self.EQNs[ode].diff_var
                ode_rhs1 = self.EQNs[ode].RHS
                ode_rhs2 = self.EQNs[ode].RHS.subs(var_list)
                trapezoidal_ae.add_eqn(Eqn(name=self.EQNs[ode].name,
                                           eqn=diff_var - Para(diff_var.name + '0') - 1 / 2 * dt * (ode_rhs1 + ode_rhs2)
                                           ))
            trapezoidal_ae.PARAM.update(deepcopy(self.PARAM))

        return trapezoidal_ae

    def __repr__(self):
        return f"DAE: {self.name}"


class PDAE(Equations):
    pass
