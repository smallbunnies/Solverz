from typing import Callable
from typing import Optional, Union
from numbers import Number

import numpy as np
from scipy.interpolate import interp1d

from Solverz.numerical_interface.Array import Array


class Param:

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 info: Optional[str] = None,
                 value: Union[np.ndarray, list, Number] = None,
                 triggerable: bool = False,
                 trigger_var: str = None,
                 trigger_fun: Callable = None,
                 dim: int = 1,
                 dtype=float,
                 sparse=False
                 ):
        self.name = name
        self.unit = unit
        self.info = info
        self.triggerable = triggerable
        self.trigger_var = trigger_var
        self.trigger_fun = trigger_fun
        self.dim = dim
        self.dtype = dtype
        self.sparse = sparse
        self.__v = None
        self.v = value

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


class IdxParam(Param):

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 info: Optional[str] = None,
                 value: Union[np.ndarray, list, Number] = None,
                 triggerable: bool = False,
                 trigger_var: str = None,
                 trigger_fun: Callable = None
                 ):
        super().__init__(name,
                         unit,
                         info,
                         value,
                         triggerable,
                         trigger_var,
                         trigger_fun,
                         dim=1,
                         dtype=int,
                         sparse=False)


class TimeSeriesParam(Param):
    def __init__(self,
                 name: str,
                 v_series,
                 time_series,
                 index=None,
                 unit: Optional[str] = None,
                 info: Optional[str] = None,
                 value: Union[np.ndarray, list, Number] = None,
                 dim=1,
                 dtype=float,
                 sparse=False
                 ):
        super().__init__(name,
                         unit,
                         info,
                         value,
                         triggerable=False,
                         trigger_var=None,
                         trigger_fun=None,
                         dim=dim,
                         dtype=dtype,
                         sparse=sparse)
        self.v_series = Array(v_series, dim=1)
        self.v_series = np.append(self.v_series, np.array(self.v_series[-1]))
        self.time_series = Array(time_series, dim=1)
        self.time_series = np.append(self.time_series, self.time_series[-1]+0.00001)
        if len(self.v_series) != len(self.time_series):
            raise ValueError("Incompatible length between value series and time series!")
        if not np.all(np.diff(self.time_series) > 0):
            raise ValueError("Time stamp should be strictly monotonically increasing!")
        self.index = index
        self.vt = interp1d(self.time_series, self.v_series, kind='linear')

    def get_v_t(self, t):
        if t is None:
            return self.v

        if self.index is not None:
            temp = self.v.copy()
            temp[self.index] = self.vt(t)
            return temp
        else:
            return self.vt(t).reshape((-1,))  # input of interp1d is zero-dimensional, we need to reshape
