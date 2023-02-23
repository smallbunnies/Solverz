from __future__ import annotations

from typing import Optional, Union

import numpy as np


class Var:

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 value: Union[np.ndarray, list] = None
                 ):
        self.name = name
        self.unit = unit
        self.array = None
        self.initialized = False
        self.v = value

    @property
    def v(self) -> np.ndarray:
        return self.array

    @v.setter
    def v(self, value: Union[np.ndarray, list]):

        if isinstance(value, list):
            self.array = np.array(value)
        else:
            self.array = value
        if not self.initialized:
            self.initialized = True

    def __array__(self):
        return self.array

    def __repr__(self):
        return f"Var: {self.name}\nvalue: {self.v}"


class TimeVar(Var):

    def __init__(self,
                 name: str,
                 unit: Optional[str] = None,
                 value: Union[np.ndarray, list] = None,
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
    def v0(self) -> np.ndarray:
        if self.initialized:
            if self.array.ndim > 1:
                return self.array[:, 0]
            else:
                return self.array[:1]  # while self.array[0] returns float instead of np.ndarray
        else:
            raise ValueError('Uninitialized')

    @v0.setter  # initializer
    def v0(self, value: Union[np.ndarray, list, int, float]):

        # the case of setting initial conditions
        if value is not None:
            if isinstance(value, list) or isinstance(value, int) or isinstance(value, float):
                temp = np.array(value).reshape(-1, )
            else:
                temp = value.reshape(-1, )
            # initialize self.__v array with the shape of initial conditions
            if temp.shape[0] > 1:
                self.array = np.zeros((temp.shape[0], self.len))
                self.array[:, 0] = temp
            else:
                self.array = np.zeros((self.len,))
                self.array[0] = temp
            if not self.initialized:
                self.initialized = True
        else:
            raise TypeError(f'Input value should not be of None type!')

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, item, value):
        self.array.__setitem__(item, value)

    def extend(self, length=50):
        self.array = np.concatenate((self.array, np.zeros((self.array.shape[0], length))), axis=1)

    def __repr__(self):
        return f"TimeVar: {self.name}\n size: {self.array.shape} value: {self.array}"
