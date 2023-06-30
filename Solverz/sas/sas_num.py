from __future__ import annotations

from typing import Union, Dict, List, Callable
import numpy as np
import sympy as sp

from Solverz.num.num_interface import numerical_interface
from Solverz.eqn import Eqn
from Solverz.equations import DAE
from Solverz.param import Param
from Solverz.sas.sas_alg import dtify, k_eqn, DT
from Solverz.var import Var, TimeVar
from Solverz.variables import VarsBasic, Vars, TimeVars


class DTeqn(DAE):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 param: Union[List[Param], Param] = None,
                 const: Union[List[Param], Param] = None
                 ):

        super().__init__(eqn, name, param, const)

        self.F_dict: Dict[str, k_eqn] = dict()
        self.G_dict: Dict[str, k_eqn] = dict()

        self.dtify()
        self.__Acache: np.ndarray = np.array([])

    def F(self, *eqnk):
        eqn = None
        if isinstance(eqnk[0], str):
            eqn = eqnk[0]
            k = eqnk[1]
        else:
            k = eqnk[0]

        temp_F = np.array([])
        if eqn:
            if eqn in self.F_dict.keys():
                return self.F_dict[eqn].RHS_NUM_FUNC(k)
        else:
            for keqn in self.F_dict.values():
                temp = keqn.RHS_NUM_FUNC(k)
                if temp.ndim == 1:
                    temp_F = np.concatenate([temp_F, temp])
                elif temp.shape[1] > 1:
                    raise ValueError(f'The array concatenated should be vector!')
                else:
                    temp_F = np.concatenate([temp_F, temp.reshape(-1, )])
            return temp_F

    def G(self, *eqnk):
        eqn = None
        if isinstance(eqnk[0], str):
            eqn = eqnk[0]
            k = eqnk[1]
        else:
            k = eqnk[0]

        temp_G = np.array([])
        if eqn:
            if eqn in self.G_dict.keys():
                return self.G_dict[eqn].RHS_NUM_FUNC(k)
        else:
            for keqn in self.G_dict.values():
                temp = keqn.RHS_NUM_FUNC(k)
                if temp.ndim == 1:
                    temp_G = np.concatenate([temp_G, temp])
                elif temp.shape[1] > 1:
                    raise ValueError(f'The array concatenated should be vector!')
                else:
                    temp_G = np.concatenate([temp_G, temp.reshape(-1, )])
            return temp_G

    @property
    def DTs(self) -> Dict[str, sp.Symbol]:
        symbol_dict = dict()
        for keqn in self.F_dict.values():
            for symbol_ in keqn.SYMBOLS.values():
                if isinstance(symbol_, DT):
                    symbol_dict[symbol_.symbol_name] = symbol_.symbol
        for keqn in self.G_dict.values():
            for symbol_ in keqn.SYMBOLS.values():
                if isinstance(symbol_, DT):
                    symbol_dict[symbol_.symbol_name] = symbol_.symbol
        return symbol_dict

    def assign_eqn_var_address(self, *xys: DTcache):
        """
        ASSIGN ADDRESSES TO EQUATIONS f and g
        """

        temp = 0
        for eqn_name in self.F_dict.keys():
            eqn_size = self.F(eqn_name, 0).shape[0]
            self.a[eqn_name] = [temp, temp + eqn_size - 1]
            temp = temp + eqn_size
            self.size[eqn_name] = eqn_size

        self.state_num = temp

        for eqn_name in self.G_dict.keys():
            eqn_size = self.G(eqn_name, 0).shape[0]
            self.a[eqn_name] = [temp, temp + eqn_size - 1]
            temp = temp + eqn_size
            self.size[eqn_name] = eqn_size

        self.algebra_num = temp - self.state_num

        temp = 0
        for xy in xys:
            for var_name, a in xy.a.items():
                if var_name in self.DTs.keys():
                    self.var_address[var_name] = [temp + xy.a[var_name][0], xy.a[var_name][-1] + temp]
                else:
                    raise ValueError(f"DAE {self.name} has no variable {var_name}")
            temp = temp + xy.total_size

        self.var_size: int = temp

    def A(self, *xy: DTcache) -> np.ndarray:
        # allocate addresses for variables and equations
        if not self.eqn_size:
            self.assign_eqn_var_address(*xy)
            self.__Acache = np.zeros((self.algebra_num, self.algebra_num))

        for eqn_name, eqn_a in self.G_dict.items():
            eqn_a = [a - self.state_num for a in self.a[eqn_name]]
            for var_name, var_a in self.var_address.items():
                var_a = [a - self.state_num for a in var_a]
                if var_name in self.G_dict[eqn_name].COEFF.NUM_FUNC.keys():
                    temp = np.array(self.G_dict[eqn_name].COEFF[var_name])
                    if temp.ndim > 1:
                        self.__Acache[eqn_a[0]:eqn_a[1] + 1, var_a[0]:var_a[1] + 1] = temp
                    elif temp.ndim == 1 and temp.shape[0] > 1:
                        self.__Acache[eqn_a[0]:eqn_a[1] + 1, var_a[0]:var_a[1] + 1] = np.diag(temp)
                    else:
                        self.__Acache[eqn_a[0]:eqn_a[1] + 1, var_a[0]:var_a[1] + 1] = temp * np.identity(
                            self.var_address[var_name][1]-self.var_address[var_name][0]+1)

        return self.__Acache

    def dtify(self):

        for eqn_name, eqn in self.g_dict.items():
            DT_eqn = dtify(eqn.expr, etf=True, eut=True, constants=list(self.CONST.keys()))
            if isinstance(DT_eqn, k_eqn):
                self.G_dict[eqn_name] = DT_eqn
            elif isinstance(DT_eqn, list):
                self.G_dict[eqn_name] = DT_eqn[0]
                for eqn_ in DT_eqn[1:]:
                    self.G_dict[eqn_.int['var'].__str__()] = eqn_
                    # add initialize equation of intermediate variables
                    self.add_eqn(Eqn(name=eqn_.int['var'].__str__(), eqn=eqn_.int['func']))

        for eqn_name, eqn in self.f_dict.items():
            DT_eqn = dtify(eqn.expr, etf=True, eut=True, constants=list(self.CONST.keys()))
            if isinstance(DT_eqn, k_eqn):
                self.F_dict[eqn_name] = DT_eqn
            elif isinstance(DT_eqn, list):
                self.F_dict[eqn_name] = DT_eqn[0]
                for eqn_ in DT_eqn[1:]:
                    self.G_dict[eqn_.int['var'].__str__()] = eqn_
                    # add initialize equation of intermediate variables
                    self.add_eqn(Eqn(name=eqn_.int['var'].__str__(), eqn=eqn_.int['func']))

    def lambdify(self, x: DTcache, y: DTcache = None):
        # initialize intermediate variables
        if y is not None:
            temp_y = [TimeVar(name, value=value) for name, value in y[0].v.items()]
            xy = (x[0], y[0])
        else:
            temp_y = []
            xy = (x[0],)

        for var_name, keqn in self.G_dict.items():
            if keqn.int is not None:
                # initialize intermediate variables
                temp_y += [TimeVar(var_name, value=self.g(var_name, *xy))]
        if len(temp_y) > 0:
            y = DTcache(TimeVars(temp_y), x.K)

        # perform lambdify using DTcache.v
        if y is not None:
            modules = [x.v, y.v, self.CONST, numerical_interface, 'numpy']
        else:
            modules = [x.v, self.CONST, numerical_interface, 'numpy']

        if not self.is_autonomous:
            temp_t = np.zeros((x.K+1, 1))
            temp_t[1] = 1
            t = {'t': temp_t}
            modules = [t] + modules
        for eqn in self.F_dict.values():
            eqn.lambdify(modules=modules)
        for eqn in self.G_dict.values():
            eqn.lambdify(modules=modules)
        return x, y

    def __repr__(self):
        return f'DT of DAE: {self.name}'


class DTcache(TimeVars):

    def __init__(self, timevars: TimeVars, K: int):
        self.K = K
        super().__init__([TimeVar(var_name, value=value[:, 0]) for var_name, value in timevars.v.items()], length=self.K+1)
        # For time vars, column number denote different time nodes while row number denote different variables
        # For array x, x[0] returns the first row, but we want it to return the 0th order DT of different variables,
        # so we have to transpose the array of DTcache by overriding self.link_var_and_array().

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

    def __setitem__(self, key, value):
        self.array[key] = value

    def __repr__(self):
        return f'DT-series (order {self.len}) {list(self.v.keys())}'


class DTvar:

    def __init__(self,
                 cache: DTcache,
                 N=100):

        self.cache = cache
        self.len = N
        self.frame_shape = cache.array.shape
        self.array: np.ndarray = np.array([])
        self.v: Dict[str, np.ndarray] = dict()
        self.link_var_and_array()

    def link_var_and_array(self):
        self.array = np.zeros((self.len, *self.frame_shape))
        for var_name in self.cache.var_size.keys():
            self.array[0, :, self.cache.a[var_name][0]:self.cache.a[var_name][-1] + 1] = self.cache.v[var_name]
            self.v[var_name] = self.array[:, :, self.cache.a[var_name][0]:self.cache.a[var_name][-1] + 1]

    def __getitem__(self, item):

        if isinstance(item, str):
            return self.v[item]
        else:
            return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = np.array(value)

    def __repr__(self):
        return f'DT-storage object (shape: {self.array.shape}) {list(self.v.keys())}'
