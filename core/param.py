from typing import Optional, Union

import numpy as np

from .solverz_array import SolverzArray


class Param:

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 info: Optional[str] = None,
                 trigger_able=False,
                 trigger=None,
                 trigger_fun=None,
                 ):
        self.name = name
        self.unit = unit
        self.info = info
        self.__v = None
        self.trigger_able = False
        self.trigger = None
        self.trigger_fun = None

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
        return f"Param: {self.name} value: {np.transpose(self.v)}"
