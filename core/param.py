from typing import Optional, Union

import numpy as np

from typing import Callable

from .solverz_array import SolverzArray
from .var import Var


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
    def v(self, value: Union[SolverzArray, np.ndarray, list]):

        if isinstance(value, np.ndarray) or isinstance(value, list):
            self.__v = SolverzArray(value)
        else:
            self.__v = value

    def __repr__(self):
        return f"Param: {self.name} value: {self.v}"
