from typing import Optional, Union

import numpy as np
from scipy.sparse import csc_array
from typing import Callable

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
                 dtype=float
                 ):
        self.name = name
        self.unit = unit
        self.info = info
        self.triggerable = triggerable
        self.trigger_var = trigger_var
        self.trigger_fun = trigger_fun
        self.dtype = dtype
        self.dim = dim
        self.__v = value

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, value: Union[np.ndarray, list, Number]):

        if self.dim == 1:
            self.__v = np.array(value, dtype=self.dtype)
        else:
            self.__v = csc_array(value)

    def __repr__(self):
        return f"Param: {self.name} value: {self.v}"
