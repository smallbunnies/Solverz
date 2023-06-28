from __future__ import annotations

from typing import Union, Dict, List, Callable

import numpy as np

from Solverz.num.num_interface import numerical_interface
from Solverz.eqn import Eqn
from Solverz.equations import DAE
from Solverz.param import Param
from Solverz.sas.sas_alg import dtify, k_eqn
from Solverz.var import Var, TimeVar
from Solverz.variables import VarsBasic, Vars, TimeVars


class DTeqn(DAE):

    def __init__(self,
                 eqn: Union[List[Eqn], Eqn],
                 name: str = None,
                 param: Union[List[Param], Param] = None
                 ):

        super().__init__(eqn, name, param)

        self.F_dict: Dict[str, k_eqn] = dict()
        self.G_dict: Dict[str, k_eqn] = dict()

        self.dtify()

        # The following must be placed after self.dtify() because we do not derive the DT of parameters but 't' is
        # assumed to be a parameter in DTeqn object.
        if not self.is_autonomous:
            self.PARAM['t'] = Param('t', value=[0])

    def F(self, k):
        temp_F = np.array([])
        for keqn in self.F_dict.values():
            temp = keqn.RHS_NUM_FUNC(k)
            if temp.ndim == 1:
                temp_F = np.concatenate([temp_F, temp])
            elif temp.shape[1] > 1:
                raise ValueError(f'The array concatenated should be vector!')
            else:
                temp_F = np.concatenate([temp_F, temp.reshape(-1, )])
        return temp_F

    def G(self, k):
        temp_G = np.array([])
        for keqn in self.G_dict.values():
            temp = keqn.RHS_NUM_FUNC(k)
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
    def A(self, *xy) -> np.ndarray:
        # allocate addresses for variables and equations
        self.assign_eqn_var_address(*xy)
        pass

    def dtify(self):

        for eqn_name, eqn in self.g_dict.items():
            DT_eqn = dtify(eqn.expr, etf=True, eut=True, constants=list(self.PARAM.keys()))
            if isinstance(DT_eqn, k_eqn):
                self.G_dict[eqn_name] = DT_eqn
            elif isinstance(DT_eqn, list):
                self.G_dict[eqn_name] = DT_eqn[0]
                for eqn_ in DT_eqn[1:]:
                    self.G_dict[eqn_.int['var'].__str__()] = eqn_
                    # add initialize equation of intermediate variables
                    self.add_eqn(Eqn(name=eqn_.int['var'].__str__(), eqn=eqn_.int['func']))

        for eqn_name, eqn in self.f_dict.items():
            DT_eqn = dtify(eqn.expr, etf=True, eut=True, constants=list(self.PARAM.keys()))
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
            y = DTcache(TimeVars(temp_y), x.K)
            xy = (x[0], y[0])

        # perform lambdify using DTcache.v
        temp_t = np.zeros((x.K, 1))
        temp_t[1] = 1
        t = {'t': temp_t}
        temp_param = self.PARAM
        del temp_param['t']
        for eqn in self.F_dict.values():
            eqn.lambdify(modules=[x.v, y.v, t, temp_param, numerical_interface, 'numpy'])
        for eqn in self.G_dict.values():
            eqn.lambdify(modules=[x.v, y.v, t, temp_param, numerical_interface, 'numpy'])
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
