from __future__ import annotations

import warnings
from numbers import Number
from typing import Union, List, Dict, Tuple

import numpy as np
from sympy import symbols, Symbol

from Solverz.eqn import Eqn, Ode
from Solverz.event import Event
from Solverz.param import Param
from Solverz.variables import Vars


class Equations:

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 param: Union[List[Param], Param] = None,
                 const: Union[List[Param], Param] = None
                 ):
        self.name = name

        self.EQNs: Dict[str, Eqn] = dict()
        self.__eqn_diffs: Dict[str, Dict[str, Eqn]] = dict()
        self.SYMBOLS: Dict[str, Symbol] = dict()
        self.__a: Dict[str, List[int]] = dict()  # equation address
        self.__eqn_size: Dict[str, int] = dict()  # equation size
        self.__var_size: int = 0  # potential equation size

        if isinstance(eqn, Eqn):
            eqn = [eqn]

        for eqn_ in eqn:
            self.add_eqn(eqn_)

        # set parameters
        self.PARAM: Dict[str, Param] = dict()
        if param:
            if isinstance(param, Param):
                param = [param]

            for param_ in param:
                if not self.is_param_defined(param_.name):
                    raise ValueError(f'Parameter {param_.name} not defined in equations!')
                else:
                    self.PARAM[param_.name] = param_

        # set constants
        self.CONST: Dict[str, Param] = dict()
        if const:
            if isinstance(const, Param):
                const = [const]

            for const_ in const:
                if not self.is_param_defined(const_.name):
                    raise ValueError(f'Constant {const_.name} not defined in equations!')
                else:
                    self.CONST[const_.name] = const_

        # generate derivatives of EQNs
        for eqn in self.EQNs.values():
            self.gen_diff(eqn)

    def add_eqn(self, eqn: Eqn):
        self.EQNs[eqn.name] = eqn
        for symbol_ in self.EQNs[eqn.name].SYMBOLS:
            # generate bare symbols without assumptions
            self.SYMBOLS[symbol_.name] = symbols(symbol_.name)
        self.a[eqn.name] = []

    def gen_diff(self, eqn: Eqn):
        self.eqn_diffs[eqn.name] = dict()
        for symbol_ in eqn.SYMBOLS:
            if symbol_.name not in self.PARAM:
                eqn_diff = eqn.RHS.diff(symbol_)
                self.eqn_diffs[eqn.name][symbol_.name] = Eqn(name=f'Diff {eqn.name} w.r.t. {symbol_.name}',
                                                             eqn=eqn_diff.__repr__(), commutative=eqn.commutative)

    @property
    def eqn_diffs(self):
        return self.__eqn_diffs

    @property
    def a(self):
        return self.__a

    @property
    def size(self):
        return self.__eqn_size

    @property
    def eqn_size(self):
        return np.sum(np.array(list(self.size.values())))

    @property
    def var_size(self):
        return self.__var_size

    @var_size.setter
    def var_size(self, value: int):
        self.__var_size = value

    def is_param_defined(self, param: str) -> bool:

        if param in self.SYMBOLS:
            return True
        else:
            return False

    with warnings.catch_warnings():
        warnings.simplefilter("once")

    def update_param(self, *args):

        if isinstance(args[0], str):
            # Update specified params
            param: str = args[0]
            value: Union[np.ndarray, list, Number] = args[1]
            try:
                self.PARAM[param].v = value
            except KeyError:
                warnings.warn(f'Equations have no parameter: {param}')
        elif isinstance(args[0], Vars):
            # Update params with Vars. For example, to update x0 in trapezoid rules.
            vars_: Vars = args[0]
            for param_name in self.PARAM.keys():
                if param_name in vars_.v.keys():
                    self.PARAM[param_name].v = vars_.v[param_name]
        elif isinstance(args[0], Event):
            # Update params with Event object
            event = args[0]
            t = args[1]
            for param_name in event.var_value.keys():
                self.PARAM[param_name].v[event.index[param_name]] = event.interpolate(param_name, t)

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
        for symbol in eqn.SYMBOLS:
            value_obtained = False
            if symbol.name in self.PARAM:
                if self.PARAM[symbol.name].triggerable:
                    for y in xys:
                        if self.PARAM[symbol.name].trigger_var in y.v:
                            self.PARAM[symbol.name].v = self.PARAM[symbol.name].trigger_fun(
                                y[self.PARAM[symbol.name].trigger_var])
                args = [*args, self.PARAM[symbol.name].v]
                value_obtained = True
            else:
                for y in xys:
                    if symbol.name in y.v:
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
        return self.eqn_diffs[eqn_name][var_name].NUM_EQN(*args)


class AE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 param: Union[List[Param], Param] = None):
        super().__init__(eqn, name, param)

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
            self.a[eqn_name] = [temp, temp + self.g(y, eqn_name).shape[0] - 1]
            temp = temp + self.g(y, eqn_name).shape[0]
            self.size[eqn_name] = self.g(y, eqn_name).shape[0]
        self.var_size = y.total_size

    def g(self, y: Vars, eqn: str = None) -> np.ndarray:
        """

        :param y:
        :param eqn:
        :return:
        """
        # FIXME: deprecate np.concatenate() here. The functions should be concatenated/combined first and then lambdified
        if not eqn:
            temp = np.array([])
            for eqn_name, eqn_ in self.EQNs.items():
                args = self.obtain_eqn_args(eqn_, y)
                temp = np.concatenate([temp, self.eval(eqn_name, *args).reshape(-1, )])
            return temp
        else:
            args = self.obtain_eqn_args(self.EQNs[eqn], y)
            return self.eval(eqn, *args)

    def g_y(self, y: Vars, eqn: List[str] = None, var: List[str] = None) -> List[Tuple[str, str, np.ndarray]]:
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
            var = list(y.v.keys())

        gy: List[Tuple[str, str, np.ndarray]] = []

        for eqn_name in eqn:
            for var_name in var:
                if var_name in self.eqn_diffs[eqn_name]:
                    args = self.obtain_eqn_args(self.eqn_diffs[eqn_name][var_name], y)
                    temp = np.array(self.eval_diffs(eqn_name, var_name, *args))
                    if temp.ndim > 1:
                        #  matrix or row vector
                        gy = [*gy, (eqn_name, var_name, temp)]
                    elif temp.ndim == 1 and temp.shape[0] > 1:
                        # column vector
                        gy = [*gy, (eqn_name, var_name, np.diag(temp))]
                    else:
                        # [number]
                        gy = [*gy, (eqn_name, var_name, temp * np.identity(y.var_size[var_name]))]
        return gy

    def j(self, y: Vars) -> np.ndarray:
        if not self.eqn_size:
            self.assign_equation_address(y)
        gy = self.g_y(y)
        j = np.zeros((self.eqn_size, y.total_size))

        for gy_tuple in gy:
            j[self.a[gy_tuple[0]][0]:(self.a[gy_tuple[0]][-1] + 1), y.a[gy_tuple[1]][0]:(y.a[gy_tuple[1]][-1] + 1)] = \
                gy_tuple[2]

        return j

    def __setitem__(self, key, value):
        """
        hai mei xiang hao ke yi yong lai gan ma
        :param key:
        :param value:
        :return:
        """
        pass

    def __repr__(self):
        if not self.eqn_size:
            return f"Algebraic equation {self.name} with addresses uninitialized"
        else:
            return f"Algebraic equation {self.name} ({self.eqn_size}×{self.var_size})"


class DAE(Equations):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 param: Union[List[Param], Param] = None,
                 const: Union[List[Param], Param] = None
                 ):

        self.f_dict: Dict[str, Ode] = dict()  # dict of state equations
        self.g_dict: Dict[str, Eqn] = dict()  # dict of algebraic equations
        self.var_address: Dict[str, List[int]] = dict()

        super().__init__(eqn, name, param, const)

        self.state_num: int = 0  # number of state variables
        self.algebra_num: int = 0  # number of algebraic variables

        # Check if some equation in self.eqn is Ode.
        # If not, raise error
        if not self.f_dict:
            raise ValueError(f'No ODE found. You should initialise AE instead!')

    def add_eqn(self, eqn: Union[Eqn, Ode]):
        self.EQNs[eqn.name] = eqn
        for symbol_ in self.EQNs[eqn.name].SYMBOLS:
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
            self.size[eqn_name] = eqn_size

        self.state_num = temp

        for eqn_name in self.g_dict.keys():
            eqn_size = self.g(eqn_name, *xys).shape[0]
            self.a[eqn_name] = [temp, temp + eqn_size - 1]
            temp = temp + eqn_size
            self.size[eqn_name] = eqn_size

        self.algebra_num = temp - self.state_num

        temp = 0
        for xy in xys:
            for var_name, a in xy.a.items():
                if var_name in self.SYMBOLS:
                    self.var_address[var_name] = [temp + xy.a[var_name][0], xy.a[var_name][-1] + temp]
                else:
                    raise ValueError(f"DAE {self.name} has no variable {var_name}")
            temp = temp + xy.total_size

        self.var_size: int = temp

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
