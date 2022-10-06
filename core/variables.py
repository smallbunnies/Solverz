from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Optional, Union, List, Dict

import numpy as np

from .solverz_array import SolverzArray
from .var import Var


class Vars:

    def __init__(self,
                 var: Union[List[Var], Var]):
        if isinstance(var, Var):
            var = [var]

        self.__v: Dict[str, SolverzArray] = {}
        self.__var_size: Dict[str, int] = {}
        self.__a: Dict[str, List[int]] = {}
        temp = 0
        for var_ in var:
            self.__v[var_.name] = var_.v
            self.__var_size[var_.name] = var_.v.row_size
            self.__a[var_.name] = [temp, temp + self.__var_size[var_.name] - 1]
            temp = temp + self.__var_size[var_.name]

        self.array = np.zeros((self.total_size, 1))
        self.link_var_and_array()

    @property
    def v(self):
        return self.__v

    @property
    def var_size(self):
        return self.__var_size

    @property
    def a(self):
        return self.__a

    @property
    def total_size(self):
        return np.sum(list(self.__var_size.values()))

    def link_var_and_array(self):
        for var_name in self.var_size.keys():
            self.array[self.a[var_name][0]:self.a[var_name][-1] + 1] = self.v[var_name].array
            self.v[var_name].array = self.array[self.a[var_name][0]:self.a[var_name][-1] + 1]

    def __getitem__(self, item):
        return self.v[item]

    def __setitem__(self, key, value):
        if key in self.v:
            self.v[key] = value
        else:
            raise ValueError(f'There is no variable {key}!')

    def __array__(self):
        return self.array

    def __repr__(self):
        return f'variables {list(self.v.keys())}'

    def __mul__(self, other: Union[Number, Vars]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, Number):
            new_vars.array[:] = new_vars.array * other
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = other.array * new_vars.array
            return new_vars
        else:
            raise TypeError(f'Input type {type(other)} invalid')

    def __rmul__(self, other: Union[Number, Vars]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, Number):
            new_vars.array[:] = other * new_vars.array
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = new_vars.array * other.array
            return new_vars
        else:
            raise TypeError(f'Input type {type(other)} invalid')

    def __add__(self, other: Union[Number, Vars, SolverzArray, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, Number):
            new_vars.array[:] = new_vars.array + other
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = new_vars.array + other.array
            return new_vars
        elif isinstance(other, SolverzArray) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                other = SolverzArray(other)
            if new_vars.total_size != other.row_size:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = new_vars.array + other.array
                return new_vars

    def __radd__(self, other: Union[Number, Vars, SolverzArray, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, Number):
            new_vars.array[:] = other + new_vars.array
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = other.array + new_vars.array
            return new_vars
        elif isinstance(other, SolverzArray) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                other = SolverzArray(other)
            if new_vars.total_size != other.row_size:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = other.array + new_vars.array
                return new_vars

    def __sub__(self, other: Union[Number, Vars, SolverzArray, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, Number):
            new_vars.array[:] = new_vars.array - other
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = new_vars.array - other.array
            return new_vars
        elif isinstance(other, SolverzArray) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                other = SolverzArray(other)
            if new_vars.total_size != other.row_size:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = new_vars.array - other.array
                return new_vars

    def __rsub__(self, other: Union[Number, Vars, SolverzArray, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, Number):
            new_vars.array[:] = other - new_vars.array
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = other.array - new_vars.array
            return new_vars
        elif isinstance(other, SolverzArray) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                other = SolverzArray(other)
            if new_vars.total_size != other.row_size:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = other.array - new_vars.array
                return new_vars

    def __truediv__(self, other: Union[Number, Vars, SolverzArray, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, Number):
            new_vars.array[:] = new_vars.array / other
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = new_vars.array / other.array
            return new_vars
        elif isinstance(other, SolverzArray) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                other = SolverzArray(other)
            if new_vars.total_size != other.row_size:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = new_vars.array / other.array
                return new_vars

    def copy(self) -> Vars:
        return deepcopy(self)
