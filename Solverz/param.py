from typing import Optional, Union

import numpy as np

from typing import Callable

from numbers import Number

from Solverz.solverz_array import SolverzArray


class Param:

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 info: Optional[str] = None,
                 value: Union[SolverzArray, np.ndarray, list] = None,
                 triggerable: bool = False,
                 trigger_var: str = None,
                 trigger_fun: Callable = None,
                 ):
        self.name = name
        self.unit = unit
        self.info = info
        self.triggerable = triggerable
        self.trigger_var = trigger_var
        self.trigger_fun = trigger_fun
        self.v = value

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, value: Union[np.ndarray, list, Number]):

        if isinstance(value, np.ndarray):
            self.__v = value
        else:
            self.__v = np.array(value)

    def __repr__(self):
        return f"Param: {self.name} value: {self.v}"
