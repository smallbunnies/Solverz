from __future__ import annotations

from copy import deepcopy
from typing import Union, List, Dict
import warnings

import numpy as np

from Solverz.sym_algebra.symbols import iVar
from Solverz.utilities.address import Address, combine_Address
from Solverz.num_api.Array import Array


def as_Vars(var: Union[iVar, List[iVar]]) -> Vars:
    var = [var] if isinstance(var, iVar) else var
    a = Address()

    for var_ in var:
        if not isinstance(var_, iVar):
            raise TypeError(f'Type {type(var_)} cannot be parsed as iVar object')
        if var_.initialized:
            a.add(var_.name, var_.value.shape[0])
        else:
            warnings.warn(f"Variable {var_.name} not initialized, set to zero")
            a.add(var_.name, 1)

    array = np.zeros((a.total_size, ))
    for var_ in var:
        if var_.initialized:
            array[a[var_.name]] = var_.value
        else:
            array[a[var_.name]] = 0

    return Vars(a, array)


class VarsBasic:

    def __init__(self):
        self.a: Address = Address()
        self.array = None

    def __array__(self):
        return self.array

    @property
    def total_size(self):
        return self.a.total_size

    @property
    def var_size(self) -> Dict[str, int]:
        return self.a.size

    @property
    def var_list(self):
        return list(self.a.v.keys())


def combine_Vars(vars1: Vars, vars2: Vars):
    a = combine_Address(vars1.a, vars2.a)
    array = np.concatenate([vars1.array, vars2.array], axis=0)
    return Vars(a, array)


class Vars(VarsBasic):

    def __init__(self,
                 a: Address,
                 array: np.ndarray):
        super().__init__()
        self.a = a
        array = Array(array, dim=1)
        if array.shape[0] != self.a.total_size:
            raise ValueError("Unequal address size and array size!")
        else:
            self.array = array

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.array[self.a[item]]
        else:
            return self.array[item]

    def __setitem__(self, key, value):
        if key in self.var_list:
            temp = Array(value, dim=1)
            if temp.shape[0] == self.array[self.a[key]].shape[0]:
                self.array[self.a[key]] = temp
            else:
                raise ValueError(f'Incompatible input array shape!')
        else:
            raise ValueError(f'There is no variable {key}!')

    def __repr__(self):
        return f'Variables (size {self.total_size}) {list(self.var_list)}'

    def __mul__(self, other: Union[int, float, Vars]) -> Vars:
        new_vars = deepcopy(self)
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = new_vars.array * other
            return new_vars
        elif isinstance(other, Vars):
            if self.a == other.a:
                new_vars.array[:] = new_vars.array * other.array
                return new_vars
            else:
                raise ValueError('Cannot multiply Vars object by another one with different variables!')
        else:
            raise TypeError(f'Input type {type(other)} invalid')

    def __rmul__(self, other: Union[int, float, Vars]) -> Vars:
        new_vars = deepcopy(self)
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = other * new_vars.array
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = other.array * new_vars.array
            return new_vars
        else:
            raise TypeError(f'Input type {type(other)} invalid')

    def __add__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        if isinstance(other, int) or isinstance(other, float):
            return Vars(self.a, self.array+other)
        elif isinstance(other, Vars):
            return Vars(self.a, self.array+other.array)
        elif isinstance(other, np.ndarray):
            return Vars(self.a, self.array+other)

    def __radd__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
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
                new_vars.array[:] = other.reshape(-1, ) + new_vars.array
                return new_vars

    def __sub__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
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
                new_vars.array[:] = new_vars.array - other.reshape(-1, )
                return new_vars

    def __rsub__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
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
                new_vars.array[:] = other.reshape(-1, ) - new_vars.array
                return new_vars

    def __truediv__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
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

    def __rtruediv__(self, other: Union[int, float, Vars, np.ndarray]) -> Vars:
        new_vars = deepcopy(self)
        if isinstance(other, int) or isinstance(other, float):
            new_vars.array[:] = other / new_vars.array
            return new_vars
        elif isinstance(other, Vars):
            new_vars.array[:] = other.array / new_vars.array
            return new_vars
        elif isinstance(other, np.ndarray):
            if new_vars.total_size != other.reshape(-1, ).shape[0]:
                raise ValueError('Incompatible array size')
            else:
                new_vars.array[:] = other.reshape(-1, ) / new_vars.array
                return new_vars

    def derive_alias(self, suffix: str):

        return Vars(self.a.derive_alias(suffix), self.array)


class TimeVars(VarsBasic):

    def __init__(self,
                 Vars_: Vars,
                 length: int = 1001
                 ):

        super().__init__()

        self.a = Vars_.a

        self.array = np.zeros((length, self.total_size))
        if length > 0:
            self.array[0, :] = Vars_.array

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
                return Vars(self.a, self.array[item, 0:])
        elif isinstance(item, str):
            return self.array[:, self.a[item]]
        elif isinstance(item, slice):
            temp = TimeVars(self[item.start], length=0)
            temp.array = self.array[item, 0:]
            return temp
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
                self.array[key:key + 1, :] = value.array[:].T
        else:
            raise NotImplementedError(f'Unsupported indices')

    @property
    def T(self):
        """
        Transpose of TimeVars
        :return:
        """
        return self.array.T

    @property
    def len(self):
        return self.array.shape[0]

    def __repr__(self):
        return f'Time-series (size {self.len}×{self.total_size}) {list(self.var_list)}'

    def append(self, other: TimeVars):
        if self.a == other.a:
            self.array = np.concatenate([self.array, other.array], axis=0)
        else:
            raise ValueError("Cannot concatenate two TimeVars with different variable addresses!")
