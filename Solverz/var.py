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
        self.__v = None
        self.initialized = False
        self.v = value

    @property
    def v(self) -> np.ndarray:
        return self.__v

    @v.setter
    def v(self, value: Union[np.ndarray, list]):

        if isinstance(value, list):
            self.__v = np.array(value)
        else:
            self.__v = value
        if not self.initialized:
            self.initialized = True

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
    def v(self) -> np.ndarray:
        return self.__v

    @v.setter
    def v(self, value: Union[np.ndarray, list, int, float], initial_condition=True):
        if initial_condition:
            # the case of setting initial conditions
            if value is not None:
                if isinstance(value, list) or isinstance(value, int) or isinstance(value, float):
                    temp = np.array(value).reshape(-1,)
                else:
                    temp = value.reshape(-1,)
                # initialize self.__v array with the shape of initial conditions
                self.__v = np.zeros((temp.shape[0], self.len))
                self.__v[:, 0] = temp
                if not self.initialized:
                    self.initialized = True
            # else:
            #     raise TypeError(f'Input value should not be of None type!')
        else:
            # TODO: the case of setting non-initial conditions
            pass


    def __getitem__(self, item):
        if isinstance(item, int):
            if item > self.len:
                raise ValueError(f'Exceed the maximum index, which is {self.len}')
            elif not self.initialized:
                raise NotImplementedError(f'TimeVar {self.name} Uninitialised')
            else:
                return Var(self.name, self.unit, self.v[:, item])
        else:
            raise NotImplementedError(f'Unsupported indices')
