from typing import Callable, Optional, Union, List
from numbers import Number

import numpy as np
from scipy.interpolate import interp1d

from Solverz.num_api.Array import Array
from Solverz.variable.ssymbol import sSymBasic


class ParamBase:
    def __init__(self,
                 name: str,
                 value: Union[np.ndarray, list, Number] = None,
                 triggerable: bool = False,
                 trigger_var: Union[str, List[str]] = None,
                 trigger_fun: Callable = None,
                 dim: int = 1,
                 dtype=float,
                 sparse=False,
                 is_alias=False):
        self.name = name
        self.triggerable = triggerable
        self.trigger_var = [trigger_var] if isinstance(trigger_var, str) else trigger_var
        self.trigger_fun = trigger_fun
        self.dim = dim
        self.dtype = dtype
        self.sparse = sparse
        self.__v = None
        self.v = value
        self.is_alias = is_alias  # if the Param is an alias var

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, value):

        if value is None:
            self.__v = None
        else:
            self.__v = Array(value, dim=self.dim, sparse=self.sparse, dtype=self.dtype)

    def get_v_t(self, t):
        return self.v

    def __repr__(self):
        return f"Param: {self.name} value: {self.v}"


class Param(ParamBase, sSymBasic):

    def __init__(self,
                 name: str,
                 value: Union[np.ndarray, list, Number] = None,
                 triggerable: bool = False,
                 trigger_var: Union[str, List[str]] = None,
                 trigger_fun: Callable = None,
                 dim: int = 1,
                 dtype=float,
                 sparse=False,
                 is_alias=False
                 ):
        ParamBase.__init__(self,
                           name,
                           value,
                           triggerable,
                           trigger_var,
                           trigger_fun,
                           dim,
                           dtype,
                           sparse,
                           is_alias)
        sSymBasic.__init__(self, name=name, Type='Para', value=value, dim=dim)


class IdxParam(ParamBase, sSymBasic):

    def __init__(self,
                 name: str,
                 value: Union[np.ndarray, list, Number] = None,
                 triggerable: bool = False,
                 trigger_var: str = None,
                 trigger_fun: Callable = None
                 ):
        ParamBase.__init__(self,
                           name,
                           value,
                           triggerable,
                           trigger_var,
                           trigger_fun,
                           dim=1,
                           dtype=int,
                           sparse=False)
        sSymBasic.__init__(self,
                           name=name,
                           Type='idx',
                           value=value,
                           dim=1)


class TimeSeriesParam(Param):
    def __init__(self,
                 name: str,
                 v_series,
                 time_series,
                 index=None,
                 value: Union[np.ndarray, list, Number] = None,
                 dim=1,
                 dtype=float,
                 sparse=False
                 ):
        if value is None:
            value = v_series[0]
        super().__init__(name,
                         value,
                         triggerable=False,
                         trigger_var=None,
                         trigger_fun=None,
                         dim=dim,
                         dtype=dtype,
                         sparse=sparse)
        self.v_series = Array(v_series, dim=1)
        self.time_series = Array(time_series, dim=1)
        self.tend = self.time_series[-1]

        if len(self.v_series) != len(self.time_series):
            raise ValueError("Incompatible length between value series and time series!")
        if not np.all(np.diff(self.time_series) > 0):
            raise ValueError("Time stamp should be strictly monotonically increasing!")
        self.index = index
        self.vt = interp1d(self.time_series, self.v_series, kind='linear')

    def get_v_t(self, t):
        if t is None:
            return self.v

        if t < self.tend:
            # input of interp1d is zero-dimensional, we need to reshape
            # [0] is to eliminate the numpy DeprecationWarning in m3b9 test: Conversion of an array with ndim > 0 to a scalar is
            # deprecated, and will error in the future, which should be resolved.
            vt = self.vt(t).reshape((-1,))
        else:
            vt = self.v_series[-1:]

        if self.index is not None:
            temp = self.v.copy()

            temp[self.index] = vt

            return temp
        else:
            return vt

    def __repr__(self):
        return f"TimeSeriesParam: {self.name} value: {self.v}"
