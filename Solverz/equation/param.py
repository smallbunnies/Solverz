from typing import Optional, Union

import numpy as np
from scipy.sparse import csc_array
from typing import Callable
from Solverz.numerical_interface.Array import Array

from numbers import Number


class Param:

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 info: Optional[str] = None,
                 value: Union[np.ndarray, list] = None,
                 triggerable: bool = False,
                 trigger_var: str = None,
                 trigger_fun: Callable = None,
                 dim=1,
                 dtype=float,
                 event=None,
                 sparse=False
                 ):
        self.name = name
        self.unit = unit
        self.info = info
        self.triggerable = triggerable
        self.trigger_var = trigger_var
        self.trigger_fun = trigger_fun
        self.dtype = dtype
        self.dim = dim
        self.event = event
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

    def __repr__(self):
        return f"Param: {self.name} value: {self.v}"


class IdxParam(Param):

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 info: Optional[str] = None,
                 value: Union[np.ndarray, list] = None,
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
