from __future__ import annotations

from copy import deepcopy
from typing import Union, List, Dict

import numpy as np

from .var import Var, TimeVar


class VarsBasic:

    def __init__(self,
                 var: Union[List[Var], Var]):

        if isinstance(var, Var):
            var = [var]

        self.__v: Dict[str, np.ndarray] = {}
        self.__var_size: Dict[str, int] = {}
        self.__a: Dict[str, List[int]] = {}

        temp = 0
        for var_ in var:
            self.__v[var_.name] = var_.v
            self.__var_size[var_.name] = var_.v.shape[0]
            self.__a[var_.name] = [temp, temp + self.__var_size[var_.name] - 1]
            temp = temp + self.__var_size[var_.name]

        self.array = None

    def link_var_and_array(self):
        self.array = np.zeros((self.total_size, 1))
        for var_name in self.var_size.keys():
            self.array[self.a[var_name][0]:self.a[var_name][-1] + 1, 0] = self.v[var_name]
            self.v[var_name] = self.array[self.a[var_name][0]:self.a[var_name][-1] + 1, 0]

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


class Vars(VarsBasic):

    def __init__(self,
                 var: Union[List[Var], Var]):
        super().__init__(var)
        self.link_var_and_array()

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

    def __mul__(self, other: Union[int, float, Vars]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = new_vars.array * other
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = new_vars.array * other.array
            return new_vars
        else:
            raise TypeError(f'Input type {type(other)} invalid')

    def __rmul__(self, other: Union[int, float, Vars]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = other * new_vars.array
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = other.array * new_vars.array
            return new_vars
        else:
            raise TypeError(f'Input type {type(other)} invalid')

    def __add__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = new_vars.array + other
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = new_vars.array + other.array
            return new_vars
        elif isinstance(other, np.ndarray):
            if new_vars.total_size != other.reshape(-1, ).shape[0]:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = new_vars.array + other.reshape(-1, 1)
                return new_vars

    def __radd__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = other + new_vars.array
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = other.array + new_vars.array
            return new_vars
        elif isinstance(other, np.ndarray):
            if new_vars.total_size != other.reshape(-1, ).shape[0]:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = other.reshape(-1, 1) + new_vars.array
                return new_vars

    def __sub__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = new_vars.array - other
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = new_vars.array - other.array
            return new_vars
        elif isinstance(other, np.ndarray):
            if new_vars.total_size != other.reshape(-1, ).shape[0]:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = new_vars.array - other.reshape(-1, 1)
                return new_vars

    def __rsub__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = other - new_vars.array
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = other.array - new_vars.array
            return new_vars
        elif isinstance(other, np.ndarray):
            if new_vars.total_size != other.reshape(-1, ).shape[0]:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = other.reshape(-1, 1) - new_vars.array
                return new_vars

    def __truediv__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        new_vars.link_var_and_array()
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = new_vars.array / other
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = new_vars.array / other.array
            return new_vars
        elif isinstance(other, np.ndarray):
            if new_vars.total_size != other.reshape(-1, ).shape[0]:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = new_vars.array / other.reshape(-1, )
                return new_vars

    def derive_alias(self, suffix: str):
        """
        Derive a new Vars object with names of variables altered
        :return:
        """
        var_ = []
        for var_name in self.var_size.keys():
            var_ = [*var_, Var(var_name + suffix, value=self.v[var_name])]
        return Vars(var_)


class TimeVars(VarsBasic):

    def __init__(self,
                 time_var: Union[List[TimeVar], TimeVar],
                 length: int = 100
                 ):

        if not isinstance(time_var, list):
            time_var = [time_var]

        var = []

        for time_var_ in time_var:
            var = [*var, time_var_[0]]

        super().__init__(var)
        self.len = length
        self.link_var_and_array()

    def link_var_and_array(self):
        self.array = np.zeros((self.total_size, self.len))
        for var_name in self.var_size.keys():
            self.array[self.a[var_name][0]:self.a[var_name][-1] + 1, 0] = self.v[var_name]
            self.v[var_name] = self.array[self.a[var_name][0]:self.a[var_name][-1] + 1, :]

    def __getitem__(self, item):
        """
        case item==str
        case item==number
        case item=[str, number]
        :param item:
        :return: Time series frame, a Vars object
        """
        if isinstance(item, int):
            if item > self.len:
                raise ValueError(f'Exceed maximum indices of Time-series Variables')
            else:
                temp_vars: List[Var] = []
                for var_name in self.var_size.keys():
                    temp_vars = [*temp_vars, Var(var_name, value=self.v[var_name][:, item])]
                return Vars(temp_vars)
        elif isinstance(item, str):
            return self.v[item]
        else:
            # not implemented
            raise NotImplementedError(f'Unsupported indices')

    def __setitem__(self, key: int, value: Vars):
        """
        assign the Vars object to time series frame
        :param key:
        :param value:
        :return: none
        """
        if isinstance(key, int):
            if key > self.len:
                raise ValueError(f'Exceed the maximum index, which is {self.len}')
            else:
                for var_name in self.v.keys():
                    self.v[var_name][:, key] = value.v[var_name].copy()
        else:
            raise NotImplementedError(f'Unsupported indices')

    @classmethod
    def time_vars(cls):
        """
        constructor of TimeVars
        :return:
        """
        pass

    def __repr__(self):
        return f'Time-series (size {self.len}) {list(self.v.keys())}'
