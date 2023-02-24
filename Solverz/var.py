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

        if value is not None:  # not None
            self.initialized = True
        if isinstance(value, list):
            self.array = np.array(value)
        else:
            self.array = value

    def __array__(self):
        return self.array

    def __repr__(self):
        return f"Var: {self.name}\nvalue: {self.v}"


class TimeVar:

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
        self.name = name
        self.unit = unit
        self.initialized = False
        self.len = length
        self.array = np.zeros((self.len,))  # set default array

        if isinstance(value, list):
            self.v0 = value
        elif isinstance(value, np.ndarray):  # initialize with given value
            if value.ndim == 1:
                self.len = value.shape[0]
                self.v0 = value[0]
            else:
                if self.len == value.shape[0]:
                    value = value.T
                    self.array = value
                    self.initialized = True
                elif self.len == value.shape[1]:
                    self.array = value
                    self.initialized = True
                else:
                    raise ValueError(f"Incompatible Input shape: {value.shape} and TimeVar length: {self.len}")

    @property
    def v0(self) -> np.ndarray:
        if self.initialized:
            if self.array.ndim > 1:
                return self.array[:, 0]
            else:
                return self.array[:1]  # because self.array[0] returns float instead of np.ndarray
        else:
            raise ValueError('Uninitialized')

    @v0.setter  # initializer
    def v0(self, value: Union[np.ndarray, list, int, float]):

        if value is not None:
            if isinstance(value, list) or isinstance(value, int) or isinstance(value, float):
                temp = np.array(value).reshape(-1, )
            else:
                temp = value.reshape(-1, )
            if temp.shape[0] > 1:  # TimeVar "self.name" contains more than one variable
                self.array = np.zeros((temp.shape[0], self.len))
                self.array[:, 0] = temp
            else:  # TimeVar "self.name" contains only one variable
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

    def __array__(self):
        return self.array

    def extend(self, length=50):
        self.array = np.concatenate((self.array, np.zeros((self.array.shape[0], length))), axis=1)

    def __repr__(self):
        return f"TimeVar: {self.name}\n size: {self.array.shape} value: {self.array}"
