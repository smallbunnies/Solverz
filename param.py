from solverz_array import SolverzArray
from typing import Callable, Iterable, List, Optional, Tuple, Type, Union
import numpy as np


class Param:

    def __init__(self,
                 name: Optional[str] = None,
                 unit: Optional[str] = None,
                 triggerable=False,
                 trigger=None,
                 trigger_fun=None,
                 ):
        self.name = name
        self.unit = unit
        self.__v = None
        self.triggerable = False
        self.trigger = None
        self.trigger_fun = None

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, value: Union[SolverzArray, np.ndarray, list]):

        if isinstance(value, np.ndarray) or isinstance(value, SolverzArray) or isinstance(value, list):
            self.__v = SolverzArray(value)
        else:
            self.__v = value

    def __repr__(self):
        return f"Param {self.name}"
