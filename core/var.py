from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Optional, Union, List, Dict

import numpy as np

from .solverz_array import SolverzArray, zeros


class Var:

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 value: Union[SolverzArray, np.ndarray, list] = None
                 ):
        self.name = name
        self.unit = unit
        self.__v = None
        self.initialized = False
        self.linked = False  # if self.__v is a view of some array
        self.v = value

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

    def __repr__(self):
        return f"Var: {self.name}\nvalue: {self.v}"


class TimeVar(Var):

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 value: Union[SolverzArray, np.ndarray, list] = None,
                 length: int = 100
                 ):
        """

        :param name:
        :param unit:
        :param value:
        :param length: length of time-series variables
        """
        super().__init__(name, unit, value)
        self.len = length

    @property
    def v(self) -> SolverzArray:
        return self.__v

    @v.setter
    def v(self, value: Union[SolverzArray, np.ndarray, list], initial=True):

        if not self.initialized:
            self.initialized = True

        if value is not None:
            if isinstance(value, np.ndarray) or isinstance(value, list):
                temp = SolverzArray(value)
                if temp.column_size == 1:
                    self.__v = zeros((temp.row_size, self.len))
                    self.__v[:, 0] = temp.array.reshape((-1,))
                else:
                    raise ValueError(f'column size of input value > 1')
            else:
                if value.column_size > 1:
                    self.__v = zeros((value.row_size, self.len))
                    self.__v[:, 0] = value.array.reshape((-1,))
                else:
                    raise ValueError(f'column size of input value > 1')

