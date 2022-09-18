from __future__ import annotations
from solverz_array import SolverzArray
import numpy as np
from typing import Optional, Union, List
from numbers import Number
from copy import deepcopy


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
    def v(self):
        return self.__v

    @v.setter
    def v(self, value: Union[SolverzArray, np.ndarray, list]):

        if not self.initialized:
            self.initialized = True

        if isinstance(value, np.ndarray) or isinstance(value, SolverzArray) or isinstance(value, list):
            self.__v = SolverzArray(value)
        else:
            self.__v = value

    def link_external(self):
        pass

    def __repr__(self):
        return f"Var: {self.name} value: {np.transpose(self.v)}"


class Vars:

    def __init__(self,
                 var: Union[List[Var], Var]):
        if isinstance(var, Var):
            self.VARS = [var]
        else:
            self.VARS = var

        self.__v = {}
        self.__size = {}
        for var_ in self.VARS:
            self.__v[var_.name] = var_.v
            self.__size[var_.name] = var_.v.row_size

        self.array = np.zeros((self.total_size,1))
        self.link_var_and_array()

    @property
    def v(self):
        return self.__v

    @property
    def size(self):
        return self.__size

    @property
    def total_size(self):
        return np.sum(list(self.__size.values()))

    def link_var_and_array(self):
        temp = 0
        for key in self.size:
            # temp=self.v[key]
            self.array[temp:temp+self.size[key]] = self.v[key].array
            self.v[key].array = self.array[temp:temp+self.size[key]]
            temp = temp + self.size[key]

    def __getitem__(self, item):
        return self.v[item]

    def __setitem__(self, key, value):
        if key in self.v:
            self.v[key] = value
        else:
            raise ValueError(f'There is no variable {key}!')

    def __repr__(self):
        return f'variables {list(self.v.keys())}'

    def __mul__(self, other: Union[Number, Vars]) -> Vars:
        if isinstance(other, Number):
            self.array[:] = self.array * other
            return self
        elif isinstance(other, Vars):
            self.array[:] = other.array * self.array
            return self
        else:
            raise TypeError(f'Input type {type(other)} invalid')

    def __add__(self, other: Union[Number, Vars, SolverzArray, np.ndarray]) -> Vars:
        if isinstance(other, Number):
            self.array[:] = self.array + other
            return self
        elif isinstance(other, Vars):
            self.array[:] = self.array+other.array
            return self
        elif isinstance(other, SolverzArray) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                other = SolverzArray(other)
            if self.total_size != other.row_size:
                raise ValueError('Incompatible array size')
            else:
                self.array[:] = self.array+other.array
                return self

    def __truediv__(self, other: Union[Number, Vars, SolverzArray, np.ndarray]) -> Vars:
        if isinstance(other, Number):
            self.array[:] = self.array / other
            return self
        elif isinstance(other, Vars):
            self.array[:] = self.array/other.array
            return self
        elif isinstance(other, SolverzArray) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                other = SolverzArray(other)
            if self.total_size != other.row_size:
                raise ValueError('Incompatible array size')
            else:
                self.array[:] = self.array/other.array
                return self

    def copy(self) -> Vars:
        return deepcopy(self)
