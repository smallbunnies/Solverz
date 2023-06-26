from __future__ import annotations

from typing import Union, Dict, List, Callable

import numpy as np
from sympy import cos, sin, Derivative, symbols

from Solverz.eqn import Eqn
from Solverz.equations import DAE
from Solverz.param import Param
from Solverz.sas.sas_alg import dtify, psi, phi
from Solverz.var import Var, TimeVar
from Solverz.variables import VarsBasic, Vars, TimeVars
from Solverz.num.num_interface import lambdify, numerical_interface


class DTeqn(DAE):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 param: Union[List[Param], Param] = None
                 ):

        super().__init__(eqn, name, param)

        self.F_lhs: Dict[str, Dict[str, Eqn]] = dict()
        self.G_lhs: Dict[str, Dict[str, Eqn]] = dict()
        self.F_rhs: Dict[str, Eqn] = dict()
        self.G_rhs: Dict[str, Eqn] = dict()
        self.intermediate_vars: Dict[str, DTcache] = dict()
        self.int_v_temp = list()
        self._lamb_F_list: List[Callable] = list()
        self._lamb_G_list: List[Callable] = list()

        self.dtify()

        # The following must be placed after self.dtify() because we do not derive the DT of parameters but 't' is
        # assumed to be a parameter in DTeqn object.
        if not self.is_autonomous:
            self.PARAM['t'] = Param('t', value=[0])

    def F(self, k):
        temp_F = np.array([])
        for func in self._lamb_F_list:
            temp = func(k)
            if temp.ndim == 1:
                temp_F = np.concatenate([temp_F, temp])
            elif temp.shape[1] > 1:
                raise ValueError(f'The array concatenated should be vector!')
            else:
                temp_F = np.concatenate([temp_F, temp.reshape(-1,)])
        return temp_F

    def G(self, k):
        temp_G = np.array([])
        for func in self._lamb_G_list:
            temp = func(k)
            if temp.ndim == 1:
                temp_G = np.concatenate([temp_G, temp])
            elif temp.shape[1] > 1:
                raise ValueError(f'The array concatenated should be vector!')
            else:
                temp_G = np.concatenate([temp_G, temp.reshape(-1, )])
        return temp_G

    def update_Ainv(self, x, y):
        pass

    @property
    def Ainv(self) -> np.ndarray:
        pass

    def dtify(self):

        for eqn_name, eqn in self.g_dict.items():
            DT_eqn = dtify(eqn.EQN, etf=True, eut=True, constants=list(self.PARAM.keys()))
            if isinstance(DT_eqn, tuple):
                for coeff_var_pair in DT_eqn[0]:
                    coeff = coeff_var_pair[0]
                    var = coeff_var_pair[1]
                    self.G_lhs[eqn_name][var.symbol_name] = \
                        Eqn(name=f"Coeff of {var.__str__()}", eqn=coeff, commutative=coeff_var_pair[0].is_commutative)
                self.G_rhs[eqn_name] = Eqn(name=eqn_name, eqn=DT_eqn[1], commutative=DT_eqn[1].is_commutative)
            elif isinstance(DT_eqn, list):
                # for coeff_var_pair in DT_eqn[0]:
                #     coeff = coeff_var_pair[0]
                #     var = coeff_var_pair[1]
                #     self.G_lhs[eqn_name][var.symbol_name] = \
                #         Eqn(name=f"Coeff of {var.__str__()}", eqn=coeff, commutative=coeff_var_pair[0].is_commutative)
                self.G_rhs[eqn_name] = Eqn(name=eqn_name, eqn=DT_eqn[0][0], commutative=DT_eqn[0][0].is_commutative)
                for i in range(1, len(DT_eqn)):
                    var = DT_eqn[i][0]
                    # add intermediate variables
                    if isinstance(var.symbol, psi):
                        var_name = psi(var.symbol.eqn).__str__()
                        self.SYMBOLS[var_name] = psi(var.symbol.eqn)
                        self.int_v_temp += [var_name]
                        self.add_eqn(Eqn(name=var_name,
                                         eqn=cos(var.symbol.eqn),
                                         commutative=var.symbol.is_commutative))
                    else:
                        var_name = phi(var.symbol.eqn).__str__()
                        self.SYMBOLS[var_name] = phi(var.symbol.eqn)
                        self.int_v_temp += [var_name]
                        self.add_eqn(Eqn(name=var_name,
                                         eqn=sin(var.symbol.eqn),
                                         commutative=var.symbol.is_commutative))
                    self.G_rhs[var_name] = Eqn(name=var_name, eqn=DT_eqn[i][1],
                                               commutative=DT_eqn[i][1].is_commutative)
                    self.G_lhs[var_name] = Eqn(name=var_name, eqn=DT_eqn[i][2],
                                               commutative=DT_eqn[i][2].is_commutative)

        # dtify() of f should be placed after g because we do not want to derive the DT of equations of intermediate
        # variables extracted to g_dict. For example, if intermediate variable psi is extracted then cos function is
        #         # added to g_dict, but we should not derive the DT of this cos function.
        # TODO: 严格意义上g_dict不应该放中间变量的初始化函数

        for eqn_name, eqn in self.f_dict.items():
            DT_eqn = dtify(Derivative(symbols(eqn.diff_var, commutative=eqn.commutative), symbols('t')) - eqn.EQN,
                           etf=True,
                           eut=True,
                           constants=list(self.PARAM.keys()))
            if isinstance(DT_eqn, tuple):  # No variables extracted
                # self.F_lhs[eqn_name] = Eqn(name=eqn_name, eqn=DT_eqn[0], commutative=DT_eqn[0].is_commutative)
                self.F_rhs[eqn_name] = Eqn(name=eqn_name, eqn=DT_eqn[1], commutative=DT_eqn[1].is_commutative)
            elif isinstance(DT_eqn, list):  # Variables extracted
                # self.F_lhs[eqn_name] = Eqn(name=eqn_name, eqn=DT_eqn[0][0], commutative=DT_eqn[0][0].is_commutative)
                self.F_rhs[eqn_name] = Eqn(name=eqn_name, eqn=DT_eqn[0][1], commutative=DT_eqn[0][1].is_commutative)
                for i in range(1, len(DT_eqn)):
                    var = DT_eqn[i][0]
                    # add intermediate variables
                    if isinstance(var.symbol, psi):
                        var_name = psi(var.symbol.eqn).__str__()
                        self.SYMBOLS[var_name] = psi(var.symbol.eqn)
                        self.int_v_temp += [var_name]
                        self.add_eqn(Eqn(name=var_name,
                                         eqn=cos(var.symbol.eqn),
                                         commutative=var.symbol.is_commutative))
                    else:
                        var_name = phi(var.symbol.eqn).__str__()
                        self.SYMBOLS[var_name] = phi(var.symbol.eqn)
                        self.int_v_temp += [var_name]
                        self.add_eqn(Eqn(name=var_name,
                                         eqn=sin(var.symbol.eqn),
                                         commutative=var.symbol.is_commutative))
                    # self.G_lhs[var_name] = Eqn(name=var_name, eqn=DT_eqn[i][1],
                    #                            commutative=DT_eqn[i][1].is_commutative)
                    self.G_rhs[var_name] = Eqn(name=var_name, eqn=DT_eqn[i][2],
                                               commutative=DT_eqn[i][2].is_commutative)

    def lambdify(self, x: DTcache, y: DTcache = None):
        # initialize intermediate variables
        if y is not None:
            temp_y = [TimeVar(name, value=value) for name, value in y[0].v.items()]
            xy = (x[0], y[0])
        else:
            temp_y = []
            xy = (x[0],)

        if len(self.int_v_temp) > 0:
            for DT_name in self.int_v_temp:
                temp_y += [TimeVar(DT_name, value=self.g(DT_name, *xy))]
                y = DTcache(TimeVars(temp_y), x.K)
                xy = (x[0], y[0])
        # allocate addresses for variables and equations
        self.assign_eqn_var_address(*xy)
        # perform lambdify using DTcache.v
        temp_t = np.zeros((x.K, 1))
        temp_t[1] = 1
        t = {'t': temp_t}
        temp_param = self.PARAM
        del temp_param['t']
        self._lamb_F_list = [lambdify(eqn.EQN, modules=[x.v, y.v, t, temp_param, numerical_interface, 'numpy']) for eqn
                             in
                             self.F_rhs.values()]
        self._lamb_G_list = [lambdify(eqn.EQN, modules=[x.v, y.v, t, temp_param, numerical_interface, 'numpy']) for eqn
                             in
                             self.G_rhs.values()]
        return x, y

    def __repr__(self):
        return f'DT of DAE: {self.name}'


class DTcache(TimeVars):

    def __init__(self, timevars: TimeVars, K: int):
        super().__init__([TimeVar(var_name, value=value[:, 0]) for var_name, value in timevars.v.items()], length=K)
        # For time vars, column number denote different time nodes while row number denote different variables
        # For array x, x[0] returns the first row, but we want it to return the 0th order DT of different variables,
        # so we have to transpose the array of DTcache by overriding self.link_var_and_array().
        self.K = K

    def link_var_and_array(self):
        self.array = np.zeros((self.len, self.total_size))
        for var_name in self.var_size.keys():
            self.array[:, self.a[var_name][0]:self.a[var_name][-1] + 1] = self.v[var_name].T
            self.v[var_name] = self.array[:, self.a[var_name][0]:self.a[var_name][-1] + 1]

    def __getitem__(self, item):

        if isinstance(item, int):
            if item > self.len:
                raise ValueError(f'Exceed maximum indices of Time-series Variables')
            else:
                temp_vars: List[Var] = []
                for var_name in self.var_size.keys():
                    temp_vars = [*temp_vars, Var(var_name, value=self.v[var_name][item, :])]
                return Vars(temp_vars)
        elif isinstance(item, str):
            return self.v[item]
        else:
            return self.array[item]

    def __repr__(self):
        return f'DT-series (order {self.len}) {list(self.v.keys())}'


class DTvar(VarsBasic):

    def __init__(self,
                 dtvar: DTcache,
                 N=100):
        super().__init__([Var(var_name, value=value[:, 0]) for var_name, value in dtvar.v.items()])

        self.len = N
        self.frame_shape = dtvar.array.shape
        self.link_var_and_array()

    def link_var_and_array(self):
        self.array = np.zeros((self.len, *self.frame_shape))
        for var_name in self.var_size.keys():
            self.array[0, self.a[var_name][0]:self.a[var_name][-1] + 1, 0] = self.v[var_name]
            self.v[var_name] = self.array[:, self.a[var_name][0]:self.a[var_name][-1] + 1, :]

    def __getitem__(self, item):

        if isinstance(item, str):
            return self.v[item]
        else:
            return self.array[item]

    def __repr__(self):
        return f'DT-storage object (shape: {self.array.shape}) {list(self.v.keys())}'
