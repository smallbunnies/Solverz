from typing import Optional, Union

import numpy as np

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
                 dtype=float
                 ):
        self.name = name
        self.unit = unit
        self.info = info
        self.triggerable = triggerable
        self.trigger_var = trigger_var
        self.trigger_fun = trigger_fun
        self.dtype = dtype
        self.v = value

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, value: Union[np.ndarray, list, Number]):

        if isinstance(value, np.ndarray):
            self.__v = np.array(value, dtype=self.dtype)
        else:
            self.__v = np.array(value, dtype=self.dtype)
        if self.v.ndim < 2 and self.dtype == int:
            # idx param has only one dim
            self.__v = self.v.reshape((-1, ))
        elif self.v.ndim == 2:
            # matrix-like param
            pass
        else:
            # vector-like param
            self.__v = self.v.reshape((-1, 1))

    def __repr__(self):
        return f"Param: {self.name} value: {self.v}"
