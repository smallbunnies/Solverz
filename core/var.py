from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Optional, Union, List, Dict

import numpy as np

from .solverz_array import SolverzArray


class Var:

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None
                 ):
        self.name = name
        self.unit = unit
        self.__v = None
        self.initialized = False
        self.linked = False  # if self.__v is a view of some array

    @property
    def v(self) -> SolverzArray:
        return self.__v

    @v.setter
    def v(self, value: Union[SolverzArray, np.ndarray, list]):

        if not self.initialized:
            self.initialized = True

        if isinstance(value, np.ndarray) or isinstance(value, list):
            self.__v = SolverzArray(value)
        else:
            self.__v = value

    def link_external(self):
        pass

    def __repr__(self):
        return f"Var: {self.name} value: {np.transpose(self.v)}"



